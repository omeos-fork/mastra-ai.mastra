import { describe, it, expect, beforeAll, afterAll, beforeEach, afterEach } from 'vitest';

import { PgVector } from '.';

const generateRandomVectors = (count: number, dim: number) => {
  return Array.from({ length: count }, () => {
    // Use wider range (-1 to 1) and don't normalize
    // This creates more diverse vectors with different magnitudes
    return Array.from({ length: dim }, () => Math.random() * 2 - 1);
  });
};

const findNearestBruteForce = (query: number[], vectors: number[][], k: number) => {
  const similarities = vectors.map((vector, idx) => {
    const similarity = cosineSimilarity(query, vector);
    return { idx, dist: similarity };
  });

  const sorted = similarities.sort((a, b) => b.dist - a.dist);

  return sorted.slice(0, k).map(x => x.idx);
};

const calculateRecall = (actual: number[], expected: number[], k: number): number => {
  let score = 0;
  for (let i = 0; i < k; i++) {
    if (actual[i] === expected[i]) {
      // Exact position match
      score += 1;
    } else if (expected.includes(actual[i])) {
      // Right item, wrong position
      score += 0.5;
    }
  }
  return score / k;
};

// Utility function for cosine similarity
function cosineSimilarity(a: number[], b: number[]): number {
  const dotProduct = a.reduce((sum, val, i) => sum + val * b[i], 0);
  const normA = Math.sqrt(a.reduce((sum, val) => sum + val * val, 0));
  const normB = Math.sqrt(b.reduce((sum, val) => sum + val * val, 0));
  return dotProduct / (normA * normB);
}

interface IndexConfig {
  type: 'ivfflat';
  lists: number;
}

interface TestResult {
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
      p99: number;
    };
    clustering?: {
      numLists: number;
      avgVectorsPerList: number;
    };
  };
}

describe('PostgreSQL Vector Index Performance', () => {
  const testConfigs = {
    dimensions: [64, 384, 1024],
    sizes: [100, 1000, 10000],
    kValues: [10, 25, 50, 75, 100],
    indexConfigs: [
      // Test auto-configuration (undefined lists = use sqrt(N))
      // { type: 'ivfflat' },
      // Test fixed configurations
      { type: 'ivfflat', lists: 100 },
      // // Test size-proportional configurations
      // { type: 'ivfflat', lists: (size: number) => Math.floor(size / 10) }, // N/10 lists
    ],
  };

  let vectorDB: PgVector;
  const testIndexName = 'test_index_performance';
  const results: TestResult[] = [];
  const connectionString =
    process.env.DB_URL || 'postgresql://postgres:postgres@localhost:5434/mastra?options=--maintenance_work_mem%3D512MB';

  const HOOK_TIMEOUT = 60000;

  beforeAll(async () => {
    vectorDB = new PgVector(connectionString);
    await vectorDB.pool.query('CREATE EXTENSION IF NOT EXISTS vector;');
  }, HOOK_TIMEOUT);

  beforeEach(async () => {
    await vectorDB.deleteIndex(testIndexName);
  });

  afterAll(async () => {
    await vectorDB.deleteIndex(testIndexName);
    await vectorDB.pool.end();
    analyzeResults(results);
  });

  // Calculate timeout based on dimension, size, and k value
  const calculateTimeout = (dimension: number, size: number, k: number) => {
    // Base timeout of 30 seconds
    let timeout = 30000;

    // Add time based on dimension
    if (dimension >= 1024)
      timeout *= 3; // 3x for 1024d
    else if (dimension >= 384) timeout *= 1.5; // 1.5x for 384d

    // Add time based on size
    if (size >= 10000) timeout *= 2; // 2x for 10k vectors

    // Add time based on k
    if (k >= 75) timeout *= 1.5; // 1.5x for k >= 75

    return timeout;
  };

  for (const dimension of testConfigs.dimensions) {
    for (const indexConfig of testConfigs.indexConfigs) {
      describe(`Dimension: ${dimension}, Index: ${indexConfig.type} (lists=${indexConfig.lists})`, () => {
        // Increase timeout for higher dimensions
        const dimensionTimeout = dimension >= 1024 ? 120000 : 60000;

        it(
          'measures recall with different dataset sizes',
          async () => {
            for (const size of testConfigs.sizes) {
              console.log(`Testing recall with size ${size}...`);

              const testVectors = generateRandomVectors(size, dimension);
              const numQueries = 10;
              const queryVectors = generateRandomVectors(numQueries, dimension);

              // Create index and insert vectors
              const lists = typeof indexConfig.lists === 'function' ? indexConfig.lists(size) : indexConfig.lists;
              const config = { type: indexConfig.type, lists };
              await vectorDB.createIndex(testIndexName, dimension, 'cosine', config);

              const vectorIds = testVectors.map((_, idx) => `vec_${idx}`);
              const metadata = testVectors.map((_, idx) => ({ index: idx }));
              await vectorDB.upsert(testIndexName, testVectors, vectorIds, metadata);

              // Test each k value
              for (const k of testConfigs.kValues) {
                const recalls: number[] = [];
                for (const queryVector of queryVectors) {
                  const expectedNeighbors = findNearestBruteForce(queryVector, testVectors, k);
                  const actualResults = await vectorDB.query(testIndexName, queryVector, k);
                  const actualNeighbors = actualResults.map(r => JSON.parse(r.id).index);

                  recalls.push(calculateRecall(actualNeighbors, expectedNeighbors, k));
                }

                results.push({
                  dimension,
                  indexConfig,
                  size,
                  k,
                  metrics: {
                    recall: mean(recalls),
                    minRecall: min(recalls),
                    maxRecall: max(recalls),
                  },
                });
              }
            }
          },
          dimensionTimeout,
        );

        it(
          'measures dimension-specific query latency',
          async () => {
            for (const size of testConfigs.sizes) {
              for (const k of testConfigs.kValues) {
                const timeout = calculateTimeout(dimension, size, k);
                console.log(`Testing latency with size ${size} and k=${k} (timeout: ${timeout}ms)...`);

                const testVectors = generateRandomVectors(size, dimension);
                const queryVectors = generateRandomVectors(50, dimension);

                const lists = typeof indexConfig.lists === 'function' ? indexConfig.lists(size) : indexConfig.lists;
                const config = { type: indexConfig.type, lists };

                await vectorDB.createIndex(testIndexName, dimension, 'cosine', config);
                await vectorDB.upsert(testIndexName, testVectors);

                // Warm up
                await vectorDB.query(testIndexName, queryVectors[0], k);

                const latencies: number[] = [];
                for (const queryVector of queryVectors) {
                  const start = process.hrtime.bigint();
                  await vectorDB.query(testIndexName, queryVector, k);
                  const end = process.hrtime.bigint();
                  latencies.push(Number(end - start) / 1e6);
                }

                const sorted = [...latencies].sort((a, b) => a - b);
                results.push({
                  dimension,
                  indexConfig,
                  size,
                  k,
                  metrics: {
                    latency: {
                      p50: sorted[Math.floor(sorted.length * 0.5)],
                      p95: sorted[Math.floor(sorted.length * 0.95)],
                      p99: sorted[Math.floor(sorted.length * 0.99)],
                    },
                  },
                });
              }
            }
          },
          // Use maximum possible timeout for the whole test
          calculateTimeout(dimension, Math.max(...testConfigs.sizes), Math.max(...testConfigs.kValues)),
        );

        it(
          'measures cluster distribution',
          async () => {
            for (const size of testConfigs.sizes) {
              console.log(`Testing cluster distribution with size ${size}...`);

              const testVectors = generateRandomVectors(size, dimension);
              const lists =
                typeof indexConfig.lists === 'function'
                  ? indexConfig.lists(size)
                  : (indexConfig.lists ?? Math.floor(Math.sqrt(size)));

              const config = { type: indexConfig.type, lists };
              await vectorDB.createIndex(testIndexName, dimension, 'cosine', config);
              await vectorDB.upsert(testIndexName, testVectors);

              results.push({
                dimension,
                indexConfig,
                size,
                k: 10,
                metrics: {
                  clustering: {
                    numLists: lists,
                    avgVectorsPerList: size / lists,
                  },
                },
              });
            }
          },
          dimensionTimeout,
        );
      });
    }
  }
});

const formatTable = (data: any[], columns: string[]) => {
  // Calculate max width for each column including header
  const colWidths = columns.map(col => Math.max(col.length, ...data.map(row => row[col].toString().length)));

  // Create the table components
  const topBorder = '┌' + colWidths.map(w => '─'.repeat(w)).join('┬') + '┐';
  const headerSeparator = '├' + colWidths.map(w => '─'.repeat(w)).join('┼') + '┤';
  const bottomBorder = '└' + colWidths.map(w => '─'.repeat(w)).join('┴') + '┘';

  // Format header and rows with proper borders
  const header = '│' + columns.map((col, i) => col.padEnd(colWidths[i])).join('│') + '│';
  const rows = data.map(row => '│' + columns.map((col, i) => row[col].toString().padEnd(colWidths[i])).join('│') + '│');

  return [topBorder, header, headerSeparator, ...rows, bottomBorder].join('\n');
};

const analyzeResults = (results: TestResult[]) => {
  // Keep the explanations at the top
  console.log('\nRecall Analysis by Dataset Size and K:');
  console.log('- Dataset Size: Number of vectors in the dataset');
  console.log('- K: Number of nearest neighbors requested');
  console.log('- Min Recall: Lowest recall score across all test queries (1.0 = perfect match)');
  console.log('- Avg Recall: Average recall score across all test queries');
  console.log('- Max Recall: Highest recall score across all test queries');
  console.log('- Lists: Number of IVF lists used in the index');
  console.log('- Vectors/List: Average number of vectors per IVF list');

  console.log('\nLatency Analysis by Dataset Size and K:');
  console.log('- Dataset Size: Number of vectors in the dataset');
  console.log('- K: Number of nearest neighbors requested');
  console.log('- P50 (ms): Median query time in milliseconds');
  console.log('- P95 (ms): 95th percentile query time in milliseconds');
  console.log('- Lists: Number of IVF lists used in the index');
  console.log('- Vectors/List: Average number of vectors per IVF list');

  console.log('\nClustering Analysis by Dataset Size:');
  console.log('- Dataset Size: Number of vectors in the dataset');
  console.log('- Vectors/List: Average number of vectors per IVF list');
  console.log('- Recommended Lists: Square root of dataset size (common heuristic)');
  console.log('- Actual Lists: Number of IVF lists used in the index');
  console.log('- Efficiency: Ratio of recommended to actual lists (closer to 1.0 is better)');

  const byDimension = groupBy(results, 'dimension');

  Object.entries(byDimension).forEach(([dimension, dimensionResults]) => {
    console.log(`\n=== Analysis for ${dimension} dimensions ===\n`);

    // Group by size and k for recall analysis
    const recalls = dimensionResults
      .filter(r => r.metrics.recall !== undefined)
      .map(r => ({
        size: r.size,
        k: r.k,
        lists: typeof r.indexConfig.lists === 'function' ? r.indexConfig.lists(r.size) : r.indexConfig.lists,
        recall: r.metrics.recall!,
        minRecall: r.metrics.minRecall!,
        maxRecall: r.metrics.maxRecall!,
      }));

    console.log('Recall Analysis by Dataset Size and K:');
    const recallColumns = ['Dataset Size', 'K', 'Min Recall', 'Avg Recall', 'Max Recall', 'Lists', 'Vectors/List'];
    const recallData = Object.values(
      groupBy(
        recalls,
        r => `${r.size}-${r.k}`,
        result => ({
          'Dataset Size': result[0].size,
          K: result[0].k,
          'Min Recall': result[0].minRecall.toFixed(3),
          'Avg Recall': mean(result.map(r => r.recall)).toFixed(3),
          'Max Recall': result[0].maxRecall.toFixed(3),
          Lists: result[0].lists,
          'Vectors/List': Math.round(result[0].size / result[0].lists),
        }),
      ),
    );
    console.log(formatTable(recallData, recallColumns));

    // Similar fix for latency analysis
    const latencies = dimensionResults
      .filter(r => r.metrics.latency !== undefined)
      .map(r => ({
        size: r.size,
        k: r.k,
        lists: typeof r.indexConfig.lists === 'function' ? r.indexConfig.lists(r.size) : r.indexConfig.lists,
        p50: r.metrics.latency!.p50,
        p95: r.metrics.latency!.p95,
      }));

    const latencyColumns = ['Dataset Size', 'K', 'P50 (ms)', 'P95 (ms)', 'Lists', 'Vectors/List'];
    const latencyData = Object.values(
      groupBy(
        latencies,
        r => `${r.size}-${r.k}`,
        result => ({
          'Dataset Size': result[0].size,
          K: result[0].k,
          'P50 (ms)': result[0].p50.toFixed(2),
          'P95 (ms)': result[0].p95.toFixed(2),
          Lists: result[0].lists,
          'Vectors/List': Math.round(result[0].size / result[0].lists),
        }),
      ),
    );
    console.log(formatTable(latencyData, latencyColumns));

    console.log('Clustering Analysis by Dataset Size:');

    const clustering = dimensionResults
      .filter(r => r.metrics.clustering !== undefined)
      .map(r => ({
        size: r.size,
        vectorsPerList: r.metrics.clustering!.avgVectorsPerList,
        recommendedLists: Math.floor(Math.sqrt(r.size)),
        actualLists: r.metrics.clustering!.numLists,
      }));

    const clusteringColumns = ['Dataset Size', 'Vectors/List', 'Recommended Lists', 'Actual Lists', 'Efficiency'];
    const clusteringData = Object.values(
      groupBy(
        clustering,
        r => r.size.toString(),
        result => ({
          'Dataset Size': result[0].size,
          'Vectors/List': Math.round(result[0].vectorsPerList),
          'Recommended Lists': result[0].recommendedLists,
          'Actual Lists': result[0].actualLists,
          Efficiency: (result[0].recommendedLists / result[0].actualLists).toFixed(2),
        }),
      ),
    );
    console.log(formatTable(clusteringData, clusteringColumns));
  });
};

const groupBy = <T, K extends keyof T>(
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

const mean = (arr: number[]) => arr.reduce((a, b) => a + b, 0) / arr.length;

const min = (arr: number[]) => Math.min(...arr);

const max = (arr: number[]) => Math.max(...arr);
