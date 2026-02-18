# Apparel Color Labeling Agent

This Bun + TypeScript agent pulls products from Meilisearch, calls a vision model via the Vercel AI SDK, and writes JSON output that matches the `FromJson` color-extractor format (each record includes `_color_tags`).

## Output Format
The output file is a JSON array. Each record is the original Meilisearch document (or the selected fields) plus:
- `_color_tags`: array of color labels (currently one label)
- `model_confidence`: numeric confidence from the model
- `error`: present only when a record fails

This format mirrors the `FromJson` enrichment behavior in the Python pipeline.

## Setup
1. Copy `.env.example` to `.env` and fill in values.
2. Update `data/color-labels.json` with your canonical label list.
3. Run:

```bash
bun run index.ts
```

## Key Environment Variables
- `GOOGLE_GENERATIVE_AI_API_KEY`: Google AI Studio API key.
- `MEILI_HOST`, `MEILI_API_KEY`, `MEILI_INDEX`: Meilisearch connection.
- `MEILI_IMAGE_FIELD`, `MEILI_ID_FIELD`: field names in your index.
- `MEILI_TITLE_FIELD`: product title field (used as a weak hint).
- `LABELS_FILE`: JSON array of allowed labels.
- `OUTPUT_DIR`, `OUTPUT_VERSION`: versioned output files.
- `LOG_DIR`: directory for structured logs.
- `MODEL_ID`: defaults to `gemini-2.5-flash`.
- `SAMPLE_RATE`: fraction of items to process (e.g. `0.2` = 20%, `0.3` = 30%). Default `0.30`.
  You can override this with `--sample-rate` at runtime.
- `LABEL_SHARE`: fraction of the dataset to pull via label-based search (BM25). Default `0.50`.

## CLI Options
- `--limit <n>`: process exactly `n` items (takes precedence over sampling).
- `--sample-rate <n>`: override sampling rate (0-1).
- `--label-share <n>`: override label-based share (0-1).

## Logging
All item-level results and token usage are written to a JSON log file in `LOG_DIR`.

## Notes
- This uses the Google provider directly, so set `GOOGLE_GENERATIVE_AI_API_KEY` in `.env`.
- If images require auth, the model call will fail. Use public image URLs or extend the agent to fetch and pass image bytes.
- Versioning is handled by `OUTPUT_VERSION` in the filename.
