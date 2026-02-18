import argparse
import asyncio
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import aiohttp
import numpy as np
from tqdm import tqdm
import imageio.v3 as iio

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from color_extractor.skin import Skin
from color_extractor.resize import Resize
from color_extractor.back import Back

def parse_args():
    parser = argparse.ArgumentParser(
        description="Build a color_names.npz training file from an enriched JSON dataset."
    )
    parser.add_argument(
        "--input-json",
        required=True,
        help="Path to JSON array produced by the agent (color-tags-*.json).",
    )
    parser.add_argument(
        "--output-npz",
        required=True,
        help="Path to output npz file (e.g. color_names_apparel_v001.npz).",
    )
    parser.add_argument(
        "--image-field",
        default="primary_image",
        help="Field name containing the image URL/path.",
    )
    parser.add_argument(
        "--labels-field",
        default="_color_tags",
        help="Field name containing the list of color labels.",
    )
    parser.add_argument(
        "--max-pixels-per-image",
        type=int,
        default=2000,
        help="Max foreground pixels to sample per image (0 = no cap).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for sampling.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=max(1, (os.cpu_count() or 2) - 1),
        help="Number of worker processes to use for CPU work.",
    )
    parser.add_argument(
        "--download-concurrency",
        type=int,
        default=32,
        help="Number of concurrent HTTP downloads.",
    )
    parser.add_argument(
        "--http-timeout",
        type=float,
        default=10.0,
        help="HTTP timeout in seconds per request.",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=2,
        help="Number of retries for failed downloads.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print URI and error for failed images.",
    )
    return parser.parse_args()


def decode_image(data: bytes):
    img = iio.imread(data)
    if len(img.shape) == 2:
        img = np.stack([img, img, img], axis=-1)
    else:
        img = img[:, :, :3]
    if img.shape[0] == 0 or img.shape[1] == 0:
        raise ValueError("empty image")
    return img


def process_record_bytes(data, label, max_pixels, seed, idx):
    try:
        img = decode_image(data)
    except Exception:
        return None

    resize = Resize({})
    back = Back({})
    skin = Skin({})

    resized = resize.get(img)
    back_mask = back.get(resized)
    skin_mask = skin.get(resized)
    mask = back_mask | skin_mask

    pixels = resized[~mask]
    if pixels.size == 0:
        return None

    if max_pixels and pixels.shape[0] > max_pixels:
        rng = np.random.default_rng(seed + idx)
        idxs = rng.choice(pixels.shape[0], max_pixels, replace=False)
        pixels = pixels[idxs]

    pixels_uint8 = np.clip(pixels * 255.0, 0, 255).astype(np.uint8)
    labels_arr = np.full((pixels_uint8.shape[0],), label, dtype="U64")
    return pixels_uint8, labels_arr


async def fetch_bytes(session, url, timeout, retries):
    last_err = None
    for attempt in range(retries + 1):
        try:
            async with session.get(url, timeout=timeout) as resp:
                resp.raise_for_status()
                return await resp.read()
        except Exception as e:
            last_err = e
            await asyncio.sleep(0.5 * (2 ** attempt))
    raise last_err


async def main_async():
    args = parse_args()

    input_path = Path(args.input_json)
    output_path = Path(args.output_npz)

    with input_path.open("r", encoding="utf-8") as f:
        records = json.load(f)

    tasks = []
    skipped = 0
    for idx, record in enumerate(records, start=1):
        uri = record.get(args.image_field)
        labels = record.get(args.labels_field) or []
        if not uri or not labels:
            skipped += 1
            continue
        tasks.append((idx, uri, labels[0]))

    samples_list = []
    labels_list = []

    connector = aiohttp.TCPConnector(limit=args.download_concurrency)
    timeout = aiohttp.ClientTimeout(total=args.http_timeout)

    async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
        loop = asyncio.get_running_loop()
        sem = asyncio.Semaphore(args.download_concurrency)
        progress = tqdm(total=len(tasks), desc="Building NPZ")

        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            async def handle_one(idx, uri, label):
                nonlocal skipped
                async with sem:
                    try:
                        data = await fetch_bytes(session, uri, args.http_timeout, args.retries)
                    except Exception as e:
                        skipped += 1
                        if args.verbose:
                            print(f"skip: {uri} (download_failed: {e})")
                        progress.update(1)
                        return

                try:
                    result = await loop.run_in_executor(
                        executor,
                        process_record_bytes,
                        data,
                        label,
                        args.max_pixels_per_image,
                        args.seed,
                        idx,
                    )
                except Exception as e:
                    skipped += 1
                    if args.verbose:
                        print(f"skip: {uri} (process_failed: {e})")
                    progress.update(1)
                    return

                if result is None:
                    skipped += 1
                    if args.verbose:
                        print(f"skip: {uri} (no_foreground_pixels)")
                    progress.update(1)
                    return

                pixels_uint8, labels_arr = result
                samples_list.append(pixels_uint8)
                labels_list.append(labels_arr)
                progress.update(1)

            await asyncio.gather(*(handle_one(*t) for t in tasks))
            progress.close()

    if not samples_list:
        raise SystemExit(
            "No samples collected. Check input JSON and image access.")

    samples = np.vstack(samples_list)
    labels = np.concatenate(labels_list)

    np.savez(output_path, samples=samples, labels=labels)

    print(
        f"Saved {output_path} with {samples.shape[0]} samples "
        f"from {len(records) - skipped} items (skipped {skipped})."
    )


def main():
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
