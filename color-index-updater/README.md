# Color Index Updater

Bun + TypeScript pipeline to backfill product colors in Meilisearch indexes by calling the color extractor inference API.

It processes these indexes by default:
- `myntra_products`
- `flipkart_products`
- `amazon_products`

## Runtime output
- Live terminal progress bars via `cli-progress`.
- No `console.log` output for operational logs.
- Structured logs are written via `winston` to `LOG_DIR`.

## What it writes per product
- `primary_color`: string | null
- `color_names`: string[]
- `color_info`: object

`color_info` shape:
- `primary_color`
- `colors`: array of `{ matched_name, matched_code, hex, proportion, percentage }`
- `palette_source`
- `mask_coverage`
- `warnings`
- `updated_at`
- `source`

## Setup
```bash
cd /Users/vaibhav/Developer/FashionAI/color-extractor/color-index-updater
cp .env.example .env
```

Update `.env` values as needed.

## Run
Dry run first:
```bash
bun run start
```

Then set `DRY_RUN=false` in `.env` and run again.

## Notes
- Uses `meilisearch` package for reads and partial updates.
- Calls `COLOR_EXTRACTOR_URL` (`/extract`) for each product using title + first image URL.
- Retries failed inference requests up to `MAX_RETRIES`.
- Optionally updates filterable attributes with:
  - `primary_color`
  - `color_names`
  - `color_info.colors.matched_name`
  - `color_info.colors.matched_code`
