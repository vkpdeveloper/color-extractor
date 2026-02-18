import { config } from 'dotenv';
import { generateText, Output } from 'ai';
import { google } from '@ai-sdk/google';
import { z } from 'zod';
import { MeiliSearch } from 'meilisearch';
import { createWriteStream, promises as fs } from 'node:fs';
import path from 'node:path';
import cliProgress from 'cli-progress';
import winston from 'winston';

config();

type Env = {
  MEILI_HOST: string;
  MEILI_API_KEY?: string;
  MEILI_INDEX: string;
  MEILI_IMAGE_FIELD: string;
  MEILI_ID_FIELD: string;
  MEILI_TITLE_FIELD?: string;
  MEILI_FIELDS?: string;
  LABELS_FILE: string;
  OUTPUT_DIR: string;
  OUTPUT_VERSION: string;
  BATCH_SIZE: string;
  CONCURRENCY: string;
  MODEL_ID: string;
  SAMPLE_RATE: string;
  LOG_DIR: string;
  LABEL_SHARE: string;
};

const env = process.env as Partial<Env>;

function requireEnv(name: keyof Env): string {
  const value = env[name];
  if (!value) {
    throw new Error(`Missing required env var: ${name}`);
  }
  return value;
}

function parseIntEnv(name: keyof Env, fallback: number): number {
  const value = env[name];
  if (!value) return fallback;
  const parsed = Number.parseInt(value, 10);
  if (Number.isNaN(parsed) || parsed <= 0) return fallback;
  return parsed;
}

const meiliHost = requireEnv('MEILI_HOST');
const meiliIndex = requireEnv('MEILI_INDEX');
const imageField = env.MEILI_IMAGE_FIELD ?? 'image';
const idField = env.MEILI_ID_FIELD ?? 'id';
const titleField = env.MEILI_TITLE_FIELD ?? 'title';
const labelsFile = requireEnv('LABELS_FILE');
const outputDir = env.OUTPUT_DIR ?? 'output';
const outputVersion = env.OUTPUT_VERSION ?? 'v001';
const logDir = env.LOG_DIR ?? 'logs';
const batchSize = parseIntEnv('BATCH_SIZE', 100);
const concurrency = parseIntEnv('CONCURRENCY', 4);
const modelId = env.MODEL_ID ?? 'gemini-2.5-flash';

type CliOptions = {
  limit?: number;
  sampleRate?: number;
  labelShare?: number;
};

function parseArgs(args: string[]): CliOptions {
  const options: CliOptions = {};
  for (let i = 0; i < args.length; i += 1) {
    const arg = args[i];
    if (arg === '--limit') {
      const value = Number.parseInt(args[i + 1] ?? '', 10);
      if (!Number.isNaN(value) && value > 0) {
        options.limit = value;
      }
      i += 1;
      continue;
    }
    if (arg === '--sample-rate') {
      const value = Number.parseFloat(args[i + 1] ?? '');
      if (!Number.isNaN(value) && value > 0 && value <= 1) {
        options.sampleRate = value;
      }
      i += 1;
      continue;
    }
    if (arg === '--label-share') {
      const value = Number.parseFloat(args[i + 1] ?? '');
      if (!Number.isNaN(value) && value > 0 && value <= 1) {
        options.labelShare = value;
      }
      i += 1;
      continue;
    }
  }
  return options;
}

const cli = parseArgs(process.argv.slice(2));
const sampleRate = (() => {
  if (cli.sampleRate !== undefined) return cli.sampleRate;
  const value = env.SAMPLE_RATE;
  if (!value) return 0.3;
  const parsed = Number.parseFloat(value);
  if (Number.isNaN(parsed) || parsed <= 0 || parsed > 1) return 0.3;
  return parsed;
})();

const labelShare = (() => {
  if (cli.labelShare !== undefined) return cli.labelShare;
  const value = env.LABEL_SHARE;
  if (!value) return 0.5;
  const parsed = Number.parseFloat(value);
  if (Number.isNaN(parsed) || parsed <= 0 || parsed > 1) return 0.5;
  return parsed;
})();

const outputFile = path.join(outputDir, `color-tags-${outputVersion}.json`);
const logFile = path.join(logDir, `color-tags-${outputVersion}.log`);
const fields = (env.MEILI_FIELDS ?? '')
  .split(',')
  .map((f) => f.trim())
  .filter(Boolean);

const client = new MeiliSearch({
  host: meiliHost,
  apiKey: env.MEILI_API_KEY,
});

function buildLabelSchema(labels: string[]) {
  if (labels.length === 0) {
    throw new Error('Labels list is empty. Provide at least one label.');
  }
  return z.object({
    label: z.enum(labels as [string, ...string[]]),
    confidence: z.number().min(0).max(1),
  });
}

async function loadLabels() {
  const raw = await fs.readFile(labelsFile, 'utf8');
  const labels = JSON.parse(raw) as string[];
  if (!Array.isArray(labels) || labels.some((l) => typeof l !== 'string')) {
    throw new Error('LABELS_FILE must be a JSON array of strings.');
  }
  return labels;
}

function buildPrompt(labels: string[], metadataColor?: string) {
  const labelList = labels.join(', ');
  const hint = metadataColor
    ? `\nTitle hint (may be wrong): ${metadataColor}`
    : '';

  return `You are labeling the primary visible color of a clothing/apparel item.\n` +
    `Focus on the specific item named in the title (e.g., tshirt, jeans, dress). ` +
    `Ignore other items and ignore background, skin, props, shadows, and reflections.\n` +
    `Choose exactly one label from this list: ${labelList}.\n` +
    `Respond with a single JSON object matching the required schema.${hint}`;
}

async function classifyColor({
  imageUrl,
  labels,
  metadataColor,
}: {
  imageUrl: string;
  labels: string[];
  metadataColor?: string;
}) {
  const schema = buildLabelSchema(labels);

  const { output, usage } = await generateText({
    model: google(modelId),
    output: Output.object({ schema }),
    messages: [
      {
        role: 'user',
        content: [
          { type: 'text', text: buildPrompt(labels, metadataColor) },
          { type: 'image', image: imageUrl },
        ],
      },
    ],
  });

  return { output, usage };
}

async function mapLimit<T, R>(
  items: T[],
  limit: number,
  worker: (item: T) => Promise<R>,
) {
  const results: R[] = [];
  let index = 0;

  async function next() {
    const current = index++;
    if (current >= items.length) return;
    const result = await worker(items[current]);
    results[current] = result;
    await next();
  }

  const workers = Array.from({ length: Math.min(limit, items.length) }, () => next());
  await Promise.all(workers);
  return results;
}

async function run() {
  const labels = await loadLabels();
  await fs.mkdir(outputDir, { recursive: true });
  await fs.mkdir(logDir, { recursive: true });

  const logger = winston.createLogger({
    level: 'info',
    format: winston.format.json(),
    transports: [new winston.transports.File({ filename: logFile })],
  });

  const index = client.index(meiliIndex);
  const stats = await index.getStats();
  const total = stats.numberOfDocuments;

  const targetTotal = cli.limit ?? Math.max(1, Math.round(total * sampleRate));
  const labelTarget = Math.max(0, Math.round(targetTotal * labelShare));
  const attributesToRetrieve = Array.from(
    new Set([idField, imageField, titleField, ...fields].filter(Boolean)),
  ) as string[];
  const labelPool: any[] = [];
  const labelPoolIds = new Set<string | number>();

  if (labelTarget > 0) {
    const perLabel = Math.max(10, Math.ceil((labelTarget / labels.length) * 2));
    for (const label of labels) {
      const res = await index.search(label, {
        limit: perLabel,
        attributesToRetrieve,
      });
      for (const hit of res.hits) {
        const id = hit[idField];
        if (!labelPoolIds.has(id)) {
          labelPoolIds.add(id);
          labelPool.push(hit);
          if (labelPool.length >= labelTarget) break;
        }
      }
      if (labelPool.length >= labelTarget) break;
    }
  }

  const progress = new cliProgress.SingleBar(
    {
      format: 'progress |{bar}| {percentage}% | {value}/{total} | ETA: {eta_formatted}',
      hideCursor: true,
    },
    cliProgress.Presets.shades_classic,
  );

  const out = createWriteStream(outputFile, { encoding: 'utf8' });
  out.write('[');

  let written = 0;
  let totalInputTokens = 0;
  let totalOutputTokens = 0;
  let usageSamples = 0;
  let processed = 0;

  const maxItems = cli.limit;
  progress.start(targetTotal, 0);
  const selectedDocs: any[] = [];
  if (labelPool.length > 0) {
    selectedDocs.push(...labelPool.slice(0, labelTarget));
  }

  const remainingNeeded = Math.max(0, targetTotal - selectedDocs.length);
  if (remainingNeeded > 0) {
    const remainingProbability = Math.min(
      1,
      remainingNeeded / Math.max(1, total - selectedDocs.length),
    );

    for (let offset = 0; offset < total && selectedDocs.length < targetTotal; offset += batchSize) {
      const docs = await index.getDocuments({
        offset,
        limit: batchSize,
        fields: attributesToRetrieve,
      });

      for (const doc of docs.results) {
        if (selectedDocs.length >= targetTotal) break;
        const id = doc[idField];
        if (labelPoolIds.has(id)) continue;
        if (Math.random() < remainingProbability) {
          selectedDocs.push(doc);
        }
      }
    }
  }

  const finalDocs = maxItems ? selectedDocs.slice(0, maxItems) : selectedDocs;
  const results = await mapLimit(finalDocs, concurrency, async (doc: any) => {
    const id = doc[idField];
    const imageUrl = doc[imageField];
    const metadataColor = titleField ? doc[titleField] : undefined;

    if (!imageUrl) {
      processed += 1;
      progress.update(Math.min(processed, targetTotal));
      return {
        ...doc,
        _color_tags: [],
        model_confidence: 0,
        error: 'missing_image_field',
      };
    }

    try {
      const { output, usage } = await classifyColor({
        imageUrl,
        labels,
        metadataColor,
      });

      logger.info('classified', {
        id,
        imageUrl,
        label: output.label,
        confidence: output.confidence,
        inputTokens: usage?.inputTokens,
        outputTokens: usage?.outputTokens,
      });

      if (usage?.inputTokens !== undefined) {
        totalInputTokens += usage.inputTokens;
      }
      if (usage?.outputTokens !== undefined) {
        totalOutputTokens += usage.outputTokens;
      }
      if (usage?.inputTokens !== undefined || usage?.outputTokens !== undefined) {
        usageSamples += 1;
      }

      return {
        ...doc,
        _color_tags: [output.label],
        model_confidence: output.confidence,
      };
    } catch (error) {
      logger.error('classify_error', {
        id,
        imageUrl,
        error: (error as Error).message,
      });
      return {
        ...doc,
        _color_tags: [],
        model_confidence: 0,
        error: (error as Error).message,
      };
    } finally {
      processed += 1;
      progress.update(Math.min(processed, targetTotal));
    }
  });

  for (const result of results) {
    const prefix = written === 0 ? '' : ',';
    out.write(prefix + JSON.stringify(result));
    written += 1;
  }

  out.write(']');
  out.end();
  progress.stop();

  console.log(`Wrote ${written} records to ${outputFile}`);
  if (usageSamples > 0) {
    const avgInput = totalInputTokens / usageSamples;
    const avgOutput = totalOutputTokens / usageSamples;
    logger.info('usage_summary', {
      samples: usageSamples,
      avgInputTokens: avgInput,
      avgOutputTokens: avgOutput,
    });
    console.log(
      `Avg tokens per item (samples=${usageSamples}): input=${avgInput.toFixed(2)}, output=${avgOutput.toFixed(2)}`,
    );
  } else {
    console.log('No token usage data returned by the model.');
  }

  logger.info('run_summary', {
    totalDocuments: total,
    sampleRate,
    labelShare,
    limit: maxItems ?? null,
    written,
    outputFile,
  });
}

run().catch((error) => {
  console.error(error);
  process.exit(1);
});
