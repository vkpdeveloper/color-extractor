import { config } from "dotenv";
import { generateText, Output } from "ai";
import { google } from "@ai-sdk/google";
import { z } from "zod";
import { MeiliSearch } from "meilisearch";
import { promises as fs } from "node:fs";
import path from "node:path";
import cliProgress from "cli-progress";

config();

type Env = {
  MEILI_HOST: string;
  MEILI_API_KEY?: string;
  MEILI_INDEX: string;
  MEILI_IMAGE_FIELD: string;
  MEILI_ID_FIELD: string;
  MEILI_TITLE_FIELD?: string;
  MEILI_FIELDS?: string;
  EXTRACTOR_API_URL?: string;
};

type ExtractorColor = {
  hex: string;
  name: string;
  matched_name: string | null;
  matched_palette_name: string | null;
  matched_code: string | null;
  proportion: number;
  percentage: number;
};

type LlmCheck = {
  name: string;
  match: boolean;
  confidence: number;
  reason: string;
};

type EvalRow = {
  id: unknown;
  title: unknown;
  imageUrl: string | null;
  extractedColorNames: string;
  extractedColorCodes: string;
  extractorPaletteNames: string;
  extractorProportions: string;
  supportedColorNames: string;
  unsupportedColorNames: string;
  avgMatchConfidence: string;
  supportedColors: number | "";
  totalColors: number | "";
  supportPercent: string;
  remark: string;
  promptTokens: number | "";
  error: string;
};

const env = process.env as Partial<Env>;

function requireEnv(name: keyof Env): string {
  const value = env[name];
  if (!value) throw new Error(`Missing required env var: ${name}`);
  return value;
}

function parseArgs(args: string[]) {
  const options: {
    query?: string;
    limit?: number;
    concurrency?: number;
    extractorApiUrl?: string;
    extractorPalettePath?: string;
    timeoutMs?: number;
    outCsv?: string;
    idsPath?: string;
  } = {};

  for (let i = 0; i < args.length; i += 1) {
    const arg = args[i];
    if (arg === "--query") {
      options.query = args[i + 1];
      i += 1;
      continue;
    }
    if (arg === "--limit") {
      const value = Number.parseInt(args[i + 1] ?? "", 10);
      if (!Number.isNaN(value) && value > 0) options.limit = value;
      i += 1;
      continue;
    }
    if (arg === "--concurrency") {
      const value = Number.parseInt(args[i + 1] ?? "", 10);
      if (!Number.isNaN(value) && value > 0) options.concurrency = value;
      i += 1;
      continue;
    }
    if (arg === "--extractor-api-url") {
      options.extractorApiUrl = args[i + 1];
      i += 1;
      continue;
    }
    if (arg === "--extractor-palette") {
      options.extractorPalettePath = args[i + 1];
      i += 1;
      continue;
    }
    if (arg === "--timeout-ms") {
      const value = Number.parseInt(args[i + 1] ?? "", 10);
      if (!Number.isNaN(value) && value > 0) options.timeoutMs = value;
      i += 1;
      continue;
    }
    if (arg === "--out-csv") {
      options.outCsv = args[i + 1];
      i += 1;
      continue;
    }
    if (arg === "--ids-path") {
      options.idsPath = args[i + 1];
      i += 1;
      continue;
    }
    if (arg.startsWith("--ids-path=")) {
      options.idsPath = arg.split("=").slice(1).join("=");
      continue;
    }
  }

  return options;
}

const cli = parseArgs(process.argv.slice(2));

const meiliHost = requireEnv("MEILI_HOST");
const meiliIndex = requireEnv("MEILI_INDEX");
const imageField = env.MEILI_IMAGE_FIELD ?? "image";
const idField = env.MEILI_ID_FIELD ?? "id";
const titleField = env.MEILI_TITLE_FIELD ?? "title";
const modelId = "gemini-3-flash-preview";
const extractorApiUrl =
  cli.extractorApiUrl ??
  env.EXTRACTOR_API_URL ??
  "http://127.0.0.1:8000/extract";

const query = cli.query ?? "tshirt";
let limit = cli.limit ?? 100;
const concurrency = Math.max(10, cli.concurrency ?? 10);
const extractorTopK = 3;
const timeoutMs = cli.timeoutMs ?? 30000;
const outCsv = cli.outCsv ?? "eval_results.csv";
const minCombinedProportion = 0.2;
const idsPath = cli.idsPath ?? null;
const pipelineDir = path.resolve(process.cwd(), "v2", "src", "color_pipeline");
let pipelineSource = "";
const maxRetries = 2;
const retryDelayMs = 2000;
const timeoutBackoff = 2;

const extractorPalettePath = cli.extractorPalettePath ?? null;

const client = new MeiliSearch({
  host: meiliHost,
  apiKey: env.MEILI_API_KEY,
});

function buildVerificationSchema() {
  return z.object({
    colors: z.array(
      z.object({
        name: z.string(),
        match: z.boolean(),
        confidence: z.number().min(0).max(1),
        reason: z.string(),
      }),
    ),
    notes: z.string().optional(),
    remark: z.string().optional(),
  });
}

function buildVerificationPrompt(
  extractedColors: ExtractorColor[],
  title?: string,
) {
  const hint = title ? `\nTitle hint (may be wrong): ${title}` : "";
  const payload = extractedColors.map((color) => ({
    name: color.name,
    hex: color.hex,
    matched_palette_name: color.matched_palette_name,
    matched_code: color.matched_code,
    proportion: color.proportion,
  }));
  return (
    `You are verifying extracted clothing colors against the image.\n` +
    `You are given the full pipeline source code for reference. Use it to understand likely failure modes.\n` +
    `Focus only on the target clothing/apparel item. Ignore background, skin, props, shadows, logos, and reflections.${hint}\n` +
    `For each input color, decide if that color is visibly present on the garment.\n` +
    `If the match is not 100%, include a technical remark about what likely went wrong (e.g., mask leakage, segmentation miss, palette mapping ambiguity, specular highlight filtering, or category mismatch).\n` +
    `Keep remarks specific and actionable, grounded in the pipeline steps.\n` +
    `Return one output entry per input color and keep the input color names unchanged.\n` +
    `Input colors JSON: ${JSON.stringify(payload)}\n\n` +
    `Pipeline source code:\n${pipelineSource}`
  );
}

async function fetchDocs() {
  const index = client.index(meiliIndex);
  const fields = (env.MEILI_FIELDS ?? "")
    .split(",")
    .map((f) => f.trim())
    .filter(Boolean);

  const attributesToRetrieve = Array.from(
    new Set([idField, imageField, titleField, ...fields].filter(Boolean)),
  ) as string[];

  const docs: any[] = [];
  let offset = 0;
  while (docs.length < limit) {
    const res = await index.search(query, {
      limit: Math.min(1000, limit - docs.length),
      offset,
      attributesToRetrieve,
    });
    if (res.hits.length === 0) break;
    docs.push(...res.hits);
    offset += res.hits.length;
  }

  return docs.slice(0, limit);
}

async function fetchDocsByIds(ids: string[]) {
  const index = client.index(meiliIndex);

  const results = await mapLimit(ids, Math.min(10, concurrency), async (id) => {
    try {
      return await index.getDocument(id);
    } catch {
      return { __missingId: id };
    }
  });

  return results;
}

async function runExtractor(
  imageUrl: string,
  title: string,
  externalId: string,
): Promise<ExtractorColor[]> {
  const response = await fetch(extractorApiUrl, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({
      image_url: imageUrl,
      title,
      top_k: extractorTopK,
      palette_path: extractorPalettePath,
      external_id: externalId,
    }),
  });

  if (!response.ok) {
    const detail = await response.text();
    throw new Error(`extractor_http_${response.status}: ${detail}`);
  }

  const payload = (await response.json()) as {
    colors?: Array<{
      hex: string;
      matched_name: string | null;
      matched_palette_name: string | null;
      matched_code: string | null;
      proportion: number;
      percentage?: number;
    }>;
  };

  const colors = payload.colors ?? [];
  if (!Array.isArray(colors)) return [];

  const extracted = colors
    .map((color) => {
      const name = color.matched_name ?? color.matched_palette_name;
      if (!name) return null;
      return {
        hex: color.hex,
        name,
        matched_name: color.matched_name,
        matched_palette_name: color.matched_palette_name,
        matched_code: color.matched_code,
        proportion: color.proportion,
        percentage: color.percentage ?? color.proportion * 100,
      } satisfies ExtractorColor;
    })
    .filter((color): color is ExtractorColor => color !== null);

  const combined = new Map<string, ExtractorColor>();
  for (const color of extracted) {
    const key = color.name.trim().toLowerCase();
    const existing = combined.get(key);
    if (!existing) {
      combined.set(key, { ...color });
      continue;
    }
    const mergedProportion = existing.proportion + color.proportion;
    existing.proportion = mergedProportion;
    existing.percentage = mergedProportion * 100;
    if (!existing.matched_palette_name && color.matched_palette_name) {
      existing.matched_palette_name = color.matched_palette_name;
    }
    if (!existing.matched_code && color.matched_code) {
      existing.matched_code = color.matched_code;
    }
    if (!existing.hex && color.hex) {
      existing.hex = color.hex;
    }
  }

  return Array.from(combined.values())
    .filter((color) => color.proportion >= minCombinedProportion)
    .sort((a, b) => b.proportion - a.proportion);
}

async function runLlmVerification(
  extractedColors: ExtractorColor[],
  imageUrl: string,
  title?: string,
) {
  const schema = buildVerificationSchema();
  const promptText = buildVerificationPrompt(extractedColors, title);
  const promptTokens = estimateTokens(promptText);
  const { output } = await generateText({
    model: google(modelId),
    output: Output.object({ schema }),
    messages: [
      {
        role: "user",
        content: [
          {
            type: "text",
            text: promptText,
          },
          { type: "image", image: imageUrl },
        ],
      },
    ],
  });
  return { output, promptTokens };
}

function alignChecks(
  extractedColors: ExtractorColor[],
  llmOutput: Awaited<ReturnType<typeof runLlmVerification>>["output"],
): LlmCheck[] {
  const llmByName = new Map(
    llmOutput.colors.map((item) => [item.name.trim().toLowerCase(), item]),
  );

  return extractedColors.map((color) => {
    const key = color.name.trim().toLowerCase();
    const hit = llmByName.get(key);
    if (!hit) {
      return {
        name: color.name,
        match: false,
        confidence: 0,
        reason: "No model verdict returned for this color.",
      };
    }
    return {
      name: color.name,
      match: hit.match,
      confidence: hit.confidence,
      reason: hit.reason,
    };
  });
}

function buildRemark(
  checks: LlmCheck[],
  llmOutput: Awaited<ReturnType<typeof runLlmVerification>>["output"],
) {
  if (checks.length === 0) return "";
  const supported = checks.filter((check) => check.match).length;
  const total = checks.length;
  if (supported === total) return "";
  if (llmOutput.remark) return llmOutput.remark;
  if (llmOutput.notes) return llmOutput.notes;
  const misses = checks
    .filter((check) => !check.match)
    .map((check) => check.name)
    .join(", ");
  return misses
    ? `Unmatched colors: ${misses}. Likely mask/segmentation miss or palette mapping ambiguity.`
    : "Not a full match; likely mask leakage or segmentation miss.";
}

function estimateTokens(text: string) {
  return Math.max(1, Math.ceil(text.length / 4));
}

function sleep(ms: number) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function isTimeoutError(error: unknown) {
  const message = (error as Error)?.message ?? "";
  return message.endsWith("_timeout") || message.includes("_timeout");
}

async function retryWithBackoff<T>(
  task: (attempt: number) => Promise<T>,
  label: string,
) {
  let lastError: unknown = null;
  for (let attempt = 0; attempt <= maxRetries; attempt += 1) {
    try {
      return await task(attempt);
    } catch (error) {
      lastError = error;
      if (!isTimeoutError(error) || attempt >= maxRetries) {
        throw error;
      }
      const delay = retryDelayMs * Math.pow(2, attempt);
      await sleep(delay);
    }
  }
  throw new Error(`${label}_timeout`);
}

async function loadPipelineSource(rootDir: string) {
  const entries: string[] = [];

  async function walk(dir: string) {
    const dirents = await fs.readdir(dir, { withFileTypes: true });
    for (const dirent of dirents) {
      const fullPath = path.join(dir, dirent.name);
      if (dirent.isDirectory()) {
        if (dirent.name.startsWith(".")) continue;
        await walk(fullPath);
        continue;
      }
      if (!dirent.isFile()) continue;
      if (!dirent.name.endsWith(".py")) continue;
      entries.push(fullPath);
    }
  }

  await walk(rootDir);
  entries.sort();
  const filePayloads = await Promise.all(
    entries.map(async (filePath) => {
      const contents = await fs.readFile(filePath, "utf8");
      const rel = path.relative(process.cwd(), filePath);
      return `\n# File: ${rel}\n${contents}`;
    }),
  );
  return filePayloads.join("\n");
}

function withTimeout<T>(
  promise: Promise<T>,
  ms: number,
  label: string,
): Promise<T> {
  let timeoutId: NodeJS.Timeout | null = null;
  const timeout = new Promise<never>((_, reject) => {
    timeoutId = setTimeout(() => {
      reject(new Error(`${label}_timeout`));
    }, ms);
  });
  return Promise.race([promise, timeout]).finally(() => {
    if (timeoutId) clearTimeout(timeoutId);
  }) as Promise<T>;
}

function csvEscape(value: unknown) {
  const text = String(value ?? "");
  if (text.includes(",") || text.includes('"') || text.includes("\n")) {
    return `"${text.replace(/"/g, '""')}"`;
  }
  return text;
}

async function mapLimit<T, R>(
  items: T[],
  limitNum: number,
  worker: (item: T) => Promise<R>,
) {
  const results: R[] = [];
  let index = 0;

  async function next() {
    const current = index++;
    if (current >= items.length) return;
    const item = items[current];
    if (item === undefined) return;
    const result = await worker(item);
    results[current] = result;
    await next();
  }

  const workers = Array.from({ length: Math.min(limitNum, items.length) }, () =>
    next(),
  );
  await Promise.all(workers);
  return results;
}

async function main() {
  pipelineSource = await loadPipelineSource(pipelineDir);
  let orderedDocs: any[] = [];
  if (idsPath) {
    const idFileContents = await fs.readFile(idsPath, "utf8");
    const ids = idFileContents
      .split(/\r?\n/)
      .map((line) => line.trim())
      .filter(Boolean);
    if (ids.length === 0) {
      throw new Error(`No ids found in ${idsPath}`);
    }
    limit = ids.length;

    const docs = await fetchDocsByIds(ids);
    const docsById = new Map<string, any>();
    for (const doc of docs) {
      if (doc?.__missingId) continue;
      const key = String(doc?.[idField] ?? "");
      if (key) docsById.set(key, doc);
    }
    orderedDocs = ids
      .slice(0, limit)
      .map((id) => docsById.get(id) ?? { __missingId: id });
  } else {
    orderedDocs = await fetchDocs();
  }

  const progress = new cliProgress.SingleBar(
    {
      format:
        "progress |{bar}| {percentage}% | {value}/{total} | colors:{supported}/{checked} err:{errors} | last:{last} | ETA: {eta_formatted}",
      hideCursor: true,
    },
    cliProgress.Presets.shades_classic,
  );

  let supportedColorsSoFar = 0;
  let checkedColorsSoFar = 0;
  let errorsSoFar = 0;
  let lastPromptTokens: number | null = null;
  progress.start(orderedDocs.length, 0, {
    supported: 0,
    checked: 0,
    errors: 0,
    last: "-",
  });

  const results = await mapLimit(orderedDocs, concurrency, async (doc: any) => {
    if (doc?.__missingId) {
      errorsSoFar += 1;
      const missingId = doc.__missingId;
      const row = {
        id: missingId,
        title: "",
        imageUrl: null,
        extractedColorNames: "",
        extractedColorCodes: "",
        extractorPaletteNames: "",
        extractorProportions: "",
        supportedColorNames: "",
        unsupportedColorNames: "",
        avgMatchConfidence: "",
        supportedColors: "",
        totalColors: "",
        supportPercent: "",
        remark: "",
        promptTokens: "",
        error: "missing_doc",
      } satisfies EvalRow;
      progress.increment(1, {
        supported: supportedColorsSoFar,
        checked: checkedColorsSoFar,
        errors: errorsSoFar,
        last: String(missingId ?? "-"),
      });
      return row;
    }

    const id = doc[idField];
    const title = doc[titleField];
    const imageUrl = doc[imageField];
    const lastId = String(id ?? "-");

    if (!imageUrl) {
      errorsSoFar += 1;
      const row = {
        id,
        title,
        imageUrl: null,
        extractedColorNames: "",
        extractedColorCodes: "",
        extractorPaletteNames: "",
        extractorProportions: "",
        supportedColorNames: "",
        unsupportedColorNames: "",
        avgMatchConfidence: "",
        supportedColors: "",
        totalColors: "",
        supportPercent: "",
        remark: "",
        promptTokens: "",
        error: "missing_image",
      } satisfies EvalRow;
      progress.increment(1, {
        supported: supportedColorsSoFar,
        checked: checkedColorsSoFar,
        errors: errorsSoFar,
        last: lastId,
      });
      return row;
    }

    if (!title) {
      errorsSoFar += 1;
      const row = {
        id,
        title,
        imageUrl,
        extractedColorNames: "",
        extractedColorCodes: "",
        extractorPaletteNames: "",
        extractorProportions: "",
        supportedColorNames: "",
        unsupportedColorNames: "",
        avgMatchConfidence: "",
        supportedColors: "",
        totalColors: "",
        supportPercent: "",
        remark: "",
        promptTokens: "",
        error: "missing_title",
      } satisfies EvalRow;
      progress.increment(1, {
        supported: supportedColorsSoFar,
        checked: checkedColorsSoFar,
        errors: errorsSoFar,
        last: lastId,
      });
      return row;
    }

    try {
        const extractedColors = await retryWithBackoff(
          (attempt) =>
            withTimeout(
              runExtractor(imageUrl, title, lastId),
              timeoutMs * Math.pow(timeoutBackoff, attempt),
              "extractor",
            ),
          "extractor",
        );
      if (extractedColors.length === 0) {
        throw new Error("extractor_no_colors");
      }

      const llmResult = await retryWithBackoff(
        (attempt) =>
          withTimeout(
            runLlmVerification(extractedColors, imageUrl, title),
            timeoutMs * Math.pow(timeoutBackoff, attempt),
            "llm",
          ),
        "llm",
      );

      const checks = alignChecks(extractedColors, llmResult.output);
      const remark = buildRemark(checks, llmResult.output);
      const checkedForItem = checks.length;
      const supportedForItem = checks.filter((check) => check.match).length;
      lastPromptTokens = llmResult.promptTokens;
      const supportedNames = checks
        .filter((check) => check.match)
        .map((check) => check.name);
      const unsupportedNames = checks
        .filter((check) => !check.match)
        .map((check) => check.name);
      const avgMatchConfidence =
        checks.length > 0
          ? (
              checks.reduce((sum, check) => sum + check.confidence, 0) /
              checks.length
            ).toFixed(2)
          : "";

      checkedColorsSoFar += checkedForItem;
      supportedColorsSoFar += supportedForItem;

      return {
        id,
        title,
        imageUrl,
        extractedColorNames: extractedColors
          .map((color) => color.name)
          .join(" | "),
        extractedColorCodes: extractedColors
          .map((color) => color.hex)
          .join(" | "),
        extractorPaletteNames: extractedColors
          .map((color) => color.matched_palette_name ?? "")
          .filter(Boolean)
          .join(" | "),
        extractorProportions: extractedColors
          .map(
            (color) => `${color.name}:${(color.proportion * 100).toFixed(1)}%`,
          )
          .join(" | "),
        supportedColorNames: supportedNames.join(" | "),
        unsupportedColorNames: unsupportedNames.join(" | "),
        avgMatchConfidence,
        supportedColors: supportedForItem,
        totalColors: checkedForItem,
        supportPercent:
          checkedForItem > 0
            ? `${((supportedForItem / checkedForItem) * 100).toFixed(2)}%`
            : "",
        remark,
        promptTokens: llmResult.promptTokens,
        error: "",
      } satisfies EvalRow;
    } catch (error) {
      errorsSoFar += 1;
      return {
        id,
        title,
        imageUrl,
        extractedColorNames: "",
        extractedColorCodes: "",
        extractorPaletteNames: "",
        extractorProportions: "",
        supportedColorNames: "",
        unsupportedColorNames: "",
        avgMatchConfidence: "",
        supportedColors: "",
        totalColors: "",
        supportPercent: "",
        remark: "",
        promptTokens: "",
        error: (error as Error).message,
      } satisfies EvalRow;
    } finally {
      progress.increment(1, {
        supported: supportedColorsSoFar,
        checked: checkedColorsSoFar,
        errors: errorsSoFar,
        last: lastId,
      });
    }
  });

  progress.stop();

  const checkedColors = checkedColorsSoFar;
  const supportedColors = supportedColorsSoFar;
  const supportRatio =
    checkedColors > 0 ? (supportedColors / checkedColors) * 100 : 0;

  const header = [
    "id",
    "title",
    "image_url",
    "extracted_colors",
    "extracted_color_codes",
    "extractor_palette_names",
    "extractor_proportions",
    "supported_colors",
    "unsupported_colors",
    "avg_match_confidence",
    "supported_count",
    "total_count",
    "support_percent",
    "remark",
    "prompt_tokens",
    "error",
  ];

  const rows = [header.join(",")];
  for (const row of results) {
    rows.push(
      [
        csvEscape(row.id),
        csvEscape(row.title),
        csvEscape(row.imageUrl),
        csvEscape(row.extractedColorNames),
        csvEscape(row.extractedColorCodes),
        csvEscape(row.extractorPaletteNames),
        csvEscape(row.extractorProportions),
        csvEscape(row.supportedColorNames),
        csvEscape(row.unsupportedColorNames),
        csvEscape(row.avgMatchConfidence),
        csvEscape(row.supportedColors),
        csvEscape(row.totalColors),
        csvEscape(row.supportPercent),
        csvEscape(row.remark),
        csvEscape(row.promptTokens),
        csvEscape(row.error),
      ].join(","),
    );
  }

  rows.push("");
  rows.push(`Supported Colors,${supportedColors}`);
  rows.push(`Checked Colors,${checkedColors}`);
  rows.push(`Support Ratio,${supportRatio.toFixed(2)}%`);
  rows.push(`Items,${results.length}`);
  rows.push(`Errors,${errorsSoFar}`);

  await fs.writeFile(outCsv, rows.join("\n"), "utf8");

  console.log(`Supported Colors: ${supportedColors}`);
  console.log(`Checked Colors: ${checkedColors}`);
  console.log(`Support Ratio: ${supportRatio.toFixed(2)}%`);
  console.log(`Items: ${results.length}`);
  console.log(`Errors: ${errorsSoFar}`);
  if (lastPromptTokens !== null) {
    console.log(`Prompt Tokens (last item): ${lastPromptTokens}`);
  }
  console.log(`CSV: ${outCsv}`);
}

main().catch((error) => {
  console.error(error);
  process.exit(1);
});
