import { config as loadEnv } from "dotenv";
import { MeiliSearch } from "meilisearch";
import cliProgress from "cli-progress";
import winston from "winston";
import path from "node:path";
import { promises as fs } from "node:fs";

loadEnv();

type Primitive = string | number | boolean | null;
type ProductDoc = Record<string, unknown>;

type ExtractColor = {
  matched_name: string | null;
  matched_code: string | null;
  hex: string;
  proportion: number;
  percentage: number;
};

type ExtractResponse = {
  primary_color: string | null;
  colors: ExtractColor[];
  palette_source: string;
  mask_coverage: number;
  warnings: string[];
  debug_mask_out?: string | null;
};

type IndexRunStats = {
  scanned: number;
  attempted: number;
  updated: number;
  failed: number;
  skippedNoImage: number;
  skippedNoTitle: number;
  skippedNoId: number;
};

type PendingUpdate = {
  payload: Record<string, unknown>;
  docId: Primitive;
  externalId?: Primitive;
};

function env(name: string, fallback?: string): string {
  const value = process.env[name] ?? fallback;
  if (value === undefined) {
    throw new Error(`Missing required env var: ${name}`);
  }
  return value;
}

function envInt(name: string, fallback: number): number {
  const value = process.env[name];
  if (!value) return fallback;
  const parsed = Number.parseInt(value, 10);
  return Number.isNaN(parsed) ? fallback : parsed;
}

function envBool(name: string, fallback: boolean): boolean {
  const value = process.env[name];
  if (!value) return fallback;
  return ["1", "true", "yes", "y", "on"].includes(value.trim().toLowerCase());
}

const cfg = {
  meiliHost: env("MEILI_HOST", "http://localhost:7700"),
  meiliApiKey: process.env.MEILI_API_KEY ?? process.env.MEILI_MASTER_KEY,
  indexes: env(
    "TARGET_INDEXES",
    "myntra_products,flipkart_products,amazon_products",
  )
    .split(",")
    .map((x) => x.trim())
    .filter(Boolean),
  inferenceUrl: env("COLOR_EXTRACTOR_URL", "http://localhost:8000/extract"),
  topK: envInt("COLOR_TOP_K", 6),
  fetchBatchSize: envInt("FETCH_BATCH_SIZE", 500),
  updateBatchSize: envInt("UPDATE_BATCH_SIZE", 100),
  concurrency: envInt("CONCURRENCY", 6),
  maxRetries: envInt("MAX_RETRIES", 2),
  requestTimeoutMs: envInt("REQUEST_TIMEOUT_MS", 30000),
  dryRun: envBool("DRY_RUN", false),
  solidGarmentMode: envBool("SOLID_GARMENT_MODE", false),
  updateFilterable: envBool("UPDATE_FILTERABLE_ATTRIBUTES", true),
  logDir: env("LOG_DIR", "logs"),
  logLevel: env("LOG_LEVEL", "info"),
};

const client = new MeiliSearch({
  host: cfg.meiliHost,
  apiKey: cfg.meiliApiKey,
});

function uniqueStrings(values: Array<string | null | undefined>): string[] {
  const seen = new Set<string>();
  const out: string[] = [];
  for (const value of values) {
    if (!value) continue;
    const normalized = value.trim();
    if (!normalized) continue;
    const key = normalized.toLowerCase();
    if (seen.has(key)) continue;
    seen.add(key);
    out.push(normalized);
  }
  return out;
}

function pickPrimaryKey(doc: ProductDoc): Primitive | undefined {
  const candidates = ["id", "external_id", "externalId", "sku"];
  for (const key of candidates) {
    const value = doc[key];
    if (
      typeof value === "string" ||
      typeof value === "number" ||
      typeof value === "boolean" ||
      value === null
    ) {
      return value;
    }
  }
  return undefined;
}

function pickTitle(doc: ProductDoc): string | undefined {
  const candidates = ["title", "name", "product_title", "productName"];
  for (const key of candidates) {
    const raw = doc[key];
    if (typeof raw === "string" && raw.trim()) {
      return raw.trim();
    }
  }
  return undefined;
}

function pickImageUrl(doc: ProductDoc): string | undefined {
  const value = doc["primary_image"];
  if (typeof value !== "string" || !value.trim()) return undefined;
  if (value.startsWith("http")) return value;
  return undefined;
}

function pickExternalId(doc: ProductDoc): Primitive | undefined {
  const value = doc["external_id"] ?? doc["externalId"];
  if (
    typeof value === "string" ||
    typeof value === "number" ||
    typeof value === "boolean" ||
    value === null
  ) {
    return value;
  }
  return undefined;
}

function skippedCount(stats: IndexRunStats): number {
  return stats.skippedNoId + stats.skippedNoTitle + stats.skippedNoImage;
}

async function sleep(ms: number): Promise<void> {
  await new Promise((resolve) => setTimeout(resolve, ms));
}

async function runWithRetry<T>(fn: () => Promise<T>, retries: number): Promise<T> {
  let attempt = 0;
  while (true) {
    try {
      return await fn();
    } catch (error) {
      if (attempt >= retries) throw error;
      const wait = 300 * (attempt + 1);
      await sleep(wait);
      attempt += 1;
    }
  }
}

async function extractColors(imageUrl: string, title: string): Promise<ExtractResponse> {
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), cfg.requestTimeoutMs);

  try {
    const res = await fetch(cfg.inferenceUrl, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({
        image_url: imageUrl,
        title,
        top_k: cfg.topK,
        solid_garment_mode: cfg.solidGarmentMode,
      }),
      signal: controller.signal,
    });

    if (!res.ok) {
      const text = await res.text();
      throw new Error(`inference_${res.status}: ${text}`);
    }

    return (await res.json()) as ExtractResponse;
  } finally {
    clearTimeout(timeout);
  }
}

async function mapWithConcurrency<T, R>(
  items: T[],
  limit: number,
  worker: (item: T, index: number) => Promise<R>,
): Promise<R[]> {
  const results: R[] = new Array(items.length);
  let next = 0;

  async function runner(): Promise<void> {
    while (true) {
      const current = next;
      next += 1;
      if (current >= items.length) return;
      const item = items[current] as T;
      results[current] = await worker(item, current);
    }
  }

  const workers = Array.from({ length: Math.min(limit, items.length) }, () => runner());
  await Promise.all(workers);
  return results;
}

async function ensureFilterableAttributes(
  indexName: string,
  logger: winston.Logger,
): Promise<void> {
  if (!cfg.updateFilterable || cfg.dryRun) return;

  const index = client.index(indexName);
  const current = (await index.getFilterableAttributes()) ?? [];
  const desired = new Set(current);
  desired.add("primary_color");
  desired.add("color_names");
  desired.add("color_info.colors.matched_name");
  desired.add("color_info.colors.matched_code");

  if (desired.size === current.length) return;

  const next = Array.from(desired);
  await index.updateFilterableAttributes(next);
  logger.info("updated filterable attributes", { index: indexName, attributes: next });
}

function updateBar(bar: cliProgress.SingleBar, stats: IndexRunStats): void {
  bar.update(stats.scanned, {
    attempted: stats.attempted,
    updated: stats.updated,
    failed: stats.failed,
    skipped: skippedCount(stats),
  });
}

async function processIndex(
  indexName: string,
  logger: winston.Logger,
  multibar: cliProgress.MultiBar,
): Promise<IndexRunStats> {
  const stats: IndexRunStats = {
    scanned: 0,
    attempted: 0,
    updated: 0,
    failed: 0,
    skippedNoImage: 0,
    skippedNoTitle: 0,
    skippedNoId: 0,
  };

  const index = client.index(indexName);
  const info = await index.getRawInfo();
  const primaryKey = info.primaryKey || undefined;

  await ensureFilterableAttributes(indexName, logger);

  const indexStats = await index.getStats();
  const totalDocs = indexStats.numberOfDocuments;

  logger.info("index processing started", {
    index: indexName,
    totalDocs,
    primaryKey: primaryKey ?? "id",
  });

  const bar = multibar.create(totalDocs || 1, 0, {
    index: indexName,
    attempted: 0,
    updated: 0,
    failed: 0,
    skipped: 0,
  });

  const fields = uniqueStrings([
    primaryKey,
    "id",
    "external_id",
    "title",
    "name",
    "product_title",
    "primary_image",
  ]);

  let offset = 0;

  try {
    while (true) {
      const page = await index.getDocuments({
        limit: cfg.fetchBatchSize,
        offset,
        fields,
      });

      const docs = page.results as ProductDoc[];
      if (!docs.length) break;

      stats.scanned += docs.length;
      updateBar(bar, stats);

      const updates = await mapWithConcurrency(docs, cfg.concurrency, async (doc) => {
        const id =
          (primaryKey ? (doc[primaryKey] as Primitive | undefined) : undefined) ??
          pickPrimaryKey(doc);
        const externalId = pickExternalId(doc);

        if (id === undefined) {
          stats.skippedNoId += 1;
          updateBar(bar, stats);
          return null;
        }

        const title = pickTitle(doc);
        if (!title) {
          stats.skippedNoTitle += 1;
          updateBar(bar, stats);
          return null;
        }

        const imageUrl = pickImageUrl(doc);
        if (!imageUrl) {
          stats.skippedNoImage += 1;
          updateBar(bar, stats);
          return null;
        }

        stats.attempted += 1;
        updateBar(bar, stats);

        try {
          const response = await runWithRetry(() => extractColors(imageUrl, title), cfg.maxRetries);
          const colorNames = uniqueStrings(response.colors.map((c) => c.matched_name));

          return {
            payload: {
              [primaryKey ?? "id"]: id,
              primary_color: response.primary_color,
              color_names: colorNames,
              color_info: response.colors,
            },
            docId: id,
            externalId,
          } as PendingUpdate;
        } catch (error) {
          stats.failed += 1;
          updateBar(bar, stats);

          const message = error instanceof Error ? error.message : String(error);
          logger.error("doc processing failed", {
            index: indexName,
            id,
            title,
            imageUrl,
            error: message,
          });

          return null;
        }
      });

      const updateRecords = updates.filter((x): x is PendingUpdate => Boolean(x));

      if (cfg.dryRun) {
        stats.updated += updateRecords.length;
        updateBar(bar, stats);
      } else {
        for (let i = 0; i < updateRecords.length; i += cfg.updateBatchSize) {
          const batch = updateRecords.slice(i, i + cfg.updateBatchSize);
          await index.updateDocuments(batch.map((record) => record.payload));
          stats.updated += batch.length;
          for (const record of batch) {
            logger.info("document updated", {
              index: indexName,
              external_id: record.externalId,
              id: record.docId,
            });
          }
          updateBar(bar, stats);
        }
      }

      offset += docs.length;
      if (docs.length < cfg.fetchBatchSize) break;
    }

    bar.update(totalDocs || 1, {
      attempted: stats.attempted,
      updated: stats.updated,
      failed: stats.failed,
      skipped: skippedCount(stats),
    });

    logger.info("index processing completed", { index: indexName, ...stats });
    return stats;
  } finally {
    bar.stop();
  }
}

async function createLogger(): Promise<winston.Logger> {
  await fs.mkdir(cfg.logDir, { recursive: true });
  const dayStamp = new Date().toISOString().slice(0, 10);
  const logFile = path.join(cfg.logDir, `color-index-updater-${dayStamp}.log`);

  const logger = winston.createLogger({
    level: cfg.logLevel,
    format: winston.format.combine(
      winston.format.timestamp(),
      winston.format.errors({ stack: true }),
      winston.format.json(),
    ),
    transports: [new winston.transports.File({ filename: logFile })],
  });

  logger.info("logger initialized", {
    logFile,
    config: {
      meiliHost: cfg.meiliHost,
      indexes: cfg.indexes,
      inferenceUrl: cfg.inferenceUrl,
      dryRun: cfg.dryRun,
      topK: cfg.topK,
      fetchBatchSize: cfg.fetchBatchSize,
      updateBatchSize: cfg.updateBatchSize,
      concurrency: cfg.concurrency,
      requestTimeoutMs: cfg.requestTimeoutMs,
      maxRetries: cfg.maxRetries,
      solidGarmentMode: cfg.solidGarmentMode,
      updateFilterable: cfg.updateFilterable,
    },
  });

  return logger;
}

async function main(): Promise<void> {
  const logger = await createLogger();

  const multibar = new cliProgress.MultiBar(
    {
      clearOnComplete: false,
      hideCursor: true,
      format:
        "[{index}] |{bar}| {percentage}% | {value}/{total} | ETA:{eta_formatted} | attempted:{attempted} updated:{updated} failed:{failed} skipped:{skipped}",
    },
    cliProgress.Presets.shades_classic,
  );

  try {
    for (const indexName of cfg.indexes) {
      try {
        await processIndex(indexName, logger, multibar);
      } catch (error) {
        const message = error instanceof Error ? error.message : String(error);
        logger.error("index fatal error", { index: indexName, error: message });
      }
    }
  } finally {
    multibar.stop();
  }

  logger.info("color index updater finished");
}

await main();
