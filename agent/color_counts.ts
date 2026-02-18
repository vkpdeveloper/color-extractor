import { promises as fs } from 'node:fs';
import path from 'node:path';

const args = process.argv.slice(2);
const inputPath = args[0];
const labelsPath = args[1] ?? path.join('data', 'color-labels.json');

if (!inputPath) {
  console.error('Usage: bun run color_counts.ts <color-tags.json> [labels.json]');
  process.exit(1);
}

async function main() {
  const [labelsRaw, dataRaw] = await Promise.all([
    fs.readFile(labelsPath, 'utf8'),
    fs.readFile(inputPath, 'utf8'),
  ]);

  const labels = JSON.parse(labelsRaw) as string[];
  if (!Array.isArray(labels) || labels.some((l) => typeof l !== 'string')) {
    throw new Error('Labels file must be a JSON array of strings.');
  }

  const records = JSON.parse(dataRaw) as Array<Record<string, unknown>>;
  if (!Array.isArray(records)) {
    throw new Error('Input file must be a JSON array.');
  }

  const counts = new Map<string, number>();
  for (const label of labels) {
    counts.set(label, 0);
  }
  const extraCounts = new Map<string, number>();

  for (const record of records) {
    const tags = record._color_tags;
    if (!Array.isArray(tags)) continue;
    for (const tag of tags) {
      if (typeof tag !== 'string') continue;
      if (counts.has(tag)) {
        counts.set(tag, (counts.get(tag) ?? 0) + 1);
      } else {
        extraCounts.set(tag, (extraCounts.get(tag) ?? 0) + 1);
      }
    }
  }

  const sorted = Array.from(counts.entries())
    .filter(([, count]) => count > 0)
    .sort((a, b) => b[1] - a[1]);

  for (const [label, count] of sorted) {
    console.log(`${label} - ${count}`);
  }

  if (extraCounts.size > 0) {
    const extraSorted = Array.from(extraCounts.entries()).sort((a, b) => b[1] - a[1]);
    console.log('\nExtra labels (not in color-labels.json):');
    for (const [label, count] of extraSorted) {
      console.log(`${label} - ${count}`);
    }
  }
}

main().catch((error) => {
  console.error(error);
  process.exit(1);
});
