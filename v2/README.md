# v2 Color Extraction Pipeline

No-training garment color extraction pipeline with Pantone-style naming support.

The pipeline is built around three steps:

1. Foreground localization using combined heuristic + SegFormer masking.
2. Dominant color extraction in Lab space with highlight filtering.
3. Palette mapping with CIEDE2000 nearest-neighbor matching.

When no user palette is provided, it falls back to `/Users/vaibhav/Developer/test/color-extractor/v2/data/open_palette.csv`.

## Install and run (uv)

From repository root:

```bash
uv sync --project v2 --extra dev
uv run --project v2 --extra dev pytest
```

Extract colors from an image:

```bash
uv run python -m v2.main extract \
  --image /absolute/path/to/image.jpg \
  --title "Blue cotton shirt" \
  --palette /absolute/path/to/pantone_like_palette.csv \
  --top-k 3 \
  --out /absolute/path/to/result.json \
  --debug-mask-out /absolute/path/to/masked.png
```

If `--palette` is omitted, the fallback open palette is used.
Use `--use-skin-mask` only when needed; it is disabled by default to avoid removing skin-toned garments.

Run the FastAPI server:

```bash
uv run --project v2 uvicorn v2.api:app --host 0.0.0.0 --port 8000
```

Example API request:

```bash
curl -X POST http://127.0.0.1:8000/extract \
  -H "content-type: application/json" \
  -d '{"image_url":"https://example.com/shirt.jpg","title":"Blue cotton shirt","top_k":3}'
```

## Palette formats

### CSV format

Required fields:

- `name`
- either `hex` OR `l,a,b`

Optional fields:

- `code`

Example (`.csv`):

```csv
name,code,hex
Signal Red,PMS-186,#C8102E
Deep Navy,PMS-2965,#003A70
```

Example Lab-based (`.csv`):

```csv
name,code,l,a,b
Custom Cool Gray,PMS-CG,56.2,0.4,-2.8
```

### JSON format

Accepts either a list of objects or `{ "colors": [...] }`.

```json
{
  "colors": [
    { "name": "Signal Red", "code": "PMS-186", "hex": "#C8102E" },
    {
      "name": "Custom Cool Gray",
      "code": "PMS-CG",
      "l": 56.2,
      "a": 0.4,
      "b": -2.8
    }
  ]
}
```

## Output JSON

The command returns:

- `primary`: top dominant color and nearest palette match.
- `dominant_colors`: up to `top-k` colors with proportions.
- `palette_source`: `pantone_user` or `open_fallback`.
- `mask_coverage`: foreground mask coverage ratio.
- `warnings`: extraction warnings.

Each color item includes:

- `matched_name`: output label. For fallback palette this is humanized/common (for example `Navy Blue`).
- `matched_palette_name`: precise nearest palette swatch name.
- `matched_code`: palette code (for example Pantone TCX).
- `delta_e`: CIEDE2000 distance to the selected precise swatch.
