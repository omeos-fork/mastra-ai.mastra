import { IndexConfig } from './types';

import { PgVector } from '.';

export interface TestResult {
  dimension: number;
  indexConfig: IndexConfig;
  size: number;
  k?: number;
  metrics: {
    recall?: number;
    minRecall?: number;
    maxRecall?: number;
    latency?: {
      p50: number;
      p95: number;
      lists?: number;
      vectorsPerList?: number;
      m?: number;
      ef?: number;
    };
    clustering?: {
      numLists?: number;
      avgVectorsPerList?: number;
    };
  };
}

export const generateRandomVectors = (count: number, dim: number) => {
  return Array.from({ length: count }, () => {
    return Array.from({ length: dim }, () => Math.random() * 2 - 1);
  });
};

export const findNearestBruteForce = (query: number[], vectors: number[][], k: number) => {
  const similarities = vectors.map((vector, idx) => {
    const similarity = cosineSimilarity(query, vector);
    return { idx, dist: similarity };
  });

  const sorted = similarities.sort((a, b) => b.dist - a.dist);
  return sorted.slice(0, k).map(x => x.idx);
};

export const calculateRecall = (actual: number[], expected: number[], k: number): number => {
  let score = 0;
  for (let i = 0; i < k; i++) {
    if (actual[i] === expected[i]) {
      score += 1;
    } else if (expected.includes(actual[i])) {
      score += 0.5;
    }
  }
  return score / k;
};

export function cosineSimilarity(a: number[], b: number[]): number {
  const dotProduct = a.reduce((sum, val, i) => sum + val * b[i], 0);
  const normA = Math.sqrt(a.reduce((sum, val) => sum + val * val, 0));
  const normB = Math.sqrt(b.reduce((sum, val) => sum + val * val, 0));
  return dotProduct / (normA * normB);
}

export const formatTable = (data: any[], columns: string[]) => {
  const colWidths = columns.map(col =>
    Math.max(
      col.length,
      ...data.map(row => {
        const value = row[col];
        return value === undefined || value === null ? '-'.length : value.toString().length;
      }),
    ),
  );

  const topBorder = '┌' + colWidths.map(w => '─'.repeat(w)).join('┬') + '┐';
  const headerSeparator = '├' + colWidths.map(w => '─'.repeat(w)).join('┼') + '┤';
  const bottomBorder = '└' + colWidths.map(w => '─'.repeat(w)).join('┴') + '┘';

  const header = '│' + columns.map((col, i) => col.padEnd(colWidths[i])).join('│') + '│';
  const rows = data.map(
    row =>
      '│' +
      columns
        .map((col, i) => {
          const value = row[col];
          const displayValue = value === undefined || value === null ? '-' : value.toString();
          return displayValue.padEnd(colWidths[i]);
        })
        .join('│') +
      '│',
  );

  return [topBorder, header, headerSeparator, ...rows, bottomBorder].join('\n');
};

export const groupBy = <T, K extends keyof T>(
  array: T[],
  key: K | ((item: T) => string),
  reducer?: (group: T[]) => any,
): Record<string, any> => {
  const grouped = array.reduce(
    (acc, item) => {
      const value = typeof key === 'function' ? key(item) : item[key];
      if (!acc[value as any]) acc[value as any] = [];
      acc[value as any].push(item);
      return acc;
    },
    {} as Record<string, T[]>,
  );

  if (reducer) {
    return Object.fromEntries(Object.entries(grouped).map(([key, group]) => [key, reducer(group)]));
  }

  return grouped;
};

export const mean = (arr: number[]) => arr.reduce((a, b) => a + b, 0) / arr.length;
export const min = (arr: number[]) => Math.min(...arr);
export const max = (arr: number[]) => Math.max(...arr);

export const calculateTimeout = (dimension: number, size: number, k: number) => {
  let timeout = 600000;
  if (dimension >= 1024) timeout *= 3;
  else if (dimension >= 384) timeout *= 1.5;
  if (size >= 10000) timeout *= 2;
  if (k >= 75) timeout *= 1.5;
  return timeout;
};

export const baseTestConfigs = {
  basicTests: {
    dimension: [
      { dimension: 64, size: 10_000, k: 10, queryCount: 15 },
      { dimension: 256, size: 10_000, k: 10, queryCount: 15 },
      { dimension: 384, size: 10_000, k: 10, queryCount: 15 },
      { dimension: 1024, size: 10_000, k: 10, queryCount: 15 },
      { dimension: 768, size: 10_000, k: 10, queryCount: 15 },
    ],

    size: [
      { dimension: 384, size: 500_000, k: 10, queryCount: 10 },
      { dimension: 384, size: 1_000_000, k: 10, queryCount: 5 },
    ],

    k: [
      { dimension: 384, size: 100_000, k: 10, queryCount: 10 },
      { dimension: 384, size: 100_000, k: 25, queryCount: 10 },
      { dimension: 384, size: 100_000, k: 50, queryCount: 10 },
      { dimension: 384, size: 100_000, k: 100, queryCount: 5 },
    ],
  },

  practicalTests: [
    { dimension: 512, size: 100_000, k: 10, queryCount: 10 },

    // Multi-modal
    { dimension: 1024, size: 100_000, k: 20, queryCount: 10 },
  ],

  stressTests: [
    // Maximum load
    { dimension: 512, size: 1_000_000, k: 50, queryCount: 5 },

    // Dense search
    { dimension: 256, size: 1_000_000, k: 100, queryCount: 5 },

    { dimension: 1024, size: 500_000, k: 50, queryCount: 5 },
  ],

  smokeTests: [{ dimension: 384, size: 1_000, k: 10, queryCount: 5 }],
};

export interface TestConfig {
  dimension: number;
  size: number;
  k: number;
  queryCount: number;
}

export async function setupTestDB(indexName: string): Promise<PgVector> {
  const connectionString = process.env.DB_URL || `postgresql://postgres:postgres@localhost:5434/mastra`;

  const vectorDB = new PgVector(connectionString);
  await vectorDB.pool.query('CREATE EXTENSION IF NOT EXISTS vector;');

  // Configure memory settings for the session
  await vectorDB.pool.query(`
    SET maintenance_work_mem = '512MB';
    SET work_mem = '256MB';
    SET temp_buffers = '256MB';
  `);

  return vectorDB;
}

export async function cleanupTestDB(vectorDB: PgVector, indexName: string) {
  await vectorDB.deleteIndex(indexName);
  await vectorDB.pool.end();
}

export async function warmupQuery(vectorDB: PgVector, indexName: string, dimension: number, k: number) {
  const warmupVector = generateRandomVectors(1, dimension)[0];
  await vectorDB.query(indexName, warmupVector, k);
}

export async function measureLatency(fn: () => Promise<any>): Promise<number> {
  const start = process.hrtime.bigint();
  return fn().then(() => {
    const end = process.hrtime.bigint();
    return Number(end - start) / 1e6;
  });
}

export const getListCount = (result: TestResult): number | undefined => {
  if (result.indexConfig.type !== 'ivfflat') return undefined;
  if (result.metrics.latency?.lists) {
    return result.metrics.latency.lists;
  }
  if (typeof result.indexConfig.ivf?.lists === 'function') {
    return result.indexConfig.ivf.lists(result.size);
  }
  return result.indexConfig.ivf?.lists ?? Math.floor(Math.sqrt(result.size));
};

export function getIndexDescription(indexConfig: IndexConfig): string {
  if (indexConfig.type === 'hnsw') {
    return `HNSW(m=${indexConfig.m},ef=${indexConfig.efConstruction})`;
  }

  if (indexConfig.type === 'ivfflat') {
    if (typeof indexConfig.ivf?.lists === 'function') {
      return 'IVF(N/10)';
    }
    if (indexConfig.ivf?.lists) {
      return `IVF(lists=${indexConfig.ivf.lists})`;
    }
    return 'IVF(dynamic)';
  }

  return 'Flat';
}

export function getSearchEf(k: number, m: number) {
  return {
    default: Math.max(k, m * k), // Default calculation
    lower: Math.max(k, (m * k) / 2), // Lower quality, faster
    higher: Math.max(k, m * k * 2), // Higher quality, slower
  };
}
