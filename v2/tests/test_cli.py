from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image


def test_cli_extract_smoke(tmp_path):
    image = np.zeros((80, 80, 3), dtype=np.uint8)
    image[:, :] = [30, 120, 210]

    image_path = tmp_path / "cli.png"
    Image.fromarray(image, mode="RGB").save(image_path)

    palette_path = tmp_path / "palette.csv"
    palette_path.write_text(
        "name,code,hex\nAzure,PMS-AZURE,#1E78D2\n", encoding="utf-8"
    )

    out_path = tmp_path / "result.json"
    mask_path = tmp_path / "mask.png"

    repo_root = Path(__file__).resolve().parents[2]
    cmd = [
        sys.executable,
        "-m",
        "v2.main",
        "extract",
        "--image",
        str(image_path),
        "--title",
        "azure shirt",
        "--palette",
        str(palette_path),
        "--top-k",
        "1",
        "--out",
        str(out_path),
        "--debug-mask-out",
        str(mask_path),
    ]

    completed = subprocess.run(
        cmd, cwd=repo_root, check=True, capture_output=True, text=True
    )

    assert completed.returncode == 0
    assert out_path.exists()
    assert mask_path.exists()

    payload = json.loads(out_path.read_text(encoding="utf-8"))
    assert payload["primary"]["matched_name"] == "Azure"
    assert "percentage" in payload["primary"]
    assert payload["primary"]["percentage"] == 100.0
    assert "percentage" in payload["dominant_colors"][0]
    assert payload["palette_source"] == "pantone_user"
