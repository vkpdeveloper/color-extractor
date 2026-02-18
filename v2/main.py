from __future__ import annotations

import argparse
import json
from pathlib import Path

from v2.src.color_pipeline.localize import CombinedMaskProvider, HeuristicMaskProvider
from v2.src.color_pipeline.pipeline import ColorExtractionPipeline


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="v2-color-pipeline",
        description="No-training color extraction with Pantone-style mapping.",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    extract = subparsers.add_parser(
        "extract",
        help="Extract dominant garment colors and map them to palette names.",
    )
    extract.add_argument(
        "--image", required=True, help="Path or URL to the input image."
    )
    extract.add_argument(
        "--title", required=True, help="Product title used for garment detection."
    )
    extract.add_argument(
        "--palette",
        default=None,
        help="Path to a user-provided Pantone-style palette (.csv/.json).",
    )
    extract.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="Maximum number of dominant colors to return.",
    )
    extract.add_argument(
        "--out",
        default=None,
        help="Optional JSON output path. If omitted, prints JSON to stdout.",
    )
    extract.add_argument(
        "--debug-mask-out",
        default=None,
        help="Optional path to save the masked output image.",
    )
    extract.add_argument(
        "--external-id",
        default=None,
        help="Optional external id for masked image naming.",
    )
    extract.add_argument(
        "--use-skin-mask",
        action="store_true",
        help="Enable skin suppression. Off by default because it can hide skin-toned garments.",
    )

    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    if args.command == "extract":
        debug_mask_out = args.debug_mask_out
        if not debug_mask_out and args.external_id:
            root_dir = Path(__file__).resolve().parents[1]
            safe_external_id = args.external_id.replace("/", "_").replace("\\", "_")
            debug_mask_out = str(root_dir / "masked_images" / f"{safe_external_id}.jpg")
        mask_provider = CombinedMaskProvider(
            title=args.title,
            heuristic=HeuristicMaskProvider(use_skin_mask=bool(args.use_skin_mask)),
        )
        pipeline = ColorExtractionPipeline(mask_provider=mask_provider)
        result = pipeline.run(
            image_path=args.image,
            title=args.title,
            palette_path=args.palette,
            top_k=args.top_k,
            debug_mask_out=debug_mask_out,
        )

        payload = json.dumps(result.to_dict(), indent=2)

        if args.out:
            output_path = Path(args.out)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(payload + "\n", encoding="utf-8")
        else:
            print(payload)
        return

    parser.error("unknown command")


if __name__ == "__main__":
    main()
