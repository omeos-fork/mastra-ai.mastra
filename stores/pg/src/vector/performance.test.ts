import { describe, it, expect, beforeAll, afterAll, beforeEach, afterEach } from 'vitest';

import { PgVector } from '.';

const generateRandomVectors = (count: number, dim: number) => {
  return Array.from({ length: count }, () => {
    // Generate truly random vectors
    const vector = Array.from({ length: dim }, () => Math.random() - 0.5);
    // Normalize
    const magnitude = Math.sqrt(vector.reduce((sum, val) => sum + val * val, 0));
    return vector.map(val => val / magnitude);
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

const calculateRecall = (actual: number[], expected: number[]) => {
  const intersection = actual.filter(x => expected.includes(x));
  const recall = intersection.length / expected.length;
  return recall;
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
  metrics: {
    recall?: number;
    latency?: {
      p50: number;
      p95: number;
      p99: number;
    };
    clustering?: {
      numClusters: number;
      minSize: number;
      maxSize: number;
      avgSize: number;
      stdDev: number;
      imbalance: number;
    };
    incrementalRecall?: {
      batchNumber: number;
      totalVectors: number;
      recall: number;
    };
  };
}

describe('PostgreSQL Vector Index Performance', () => {
  const testConfigs = {
    dimensions: [64, 384, 1024],
    sizes: [100, 1000, 10000],
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

  const TEST_TIMEOUT = 30000;
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

  for (const dimension of testConfigs.dimensions) {
    for (const indexConfig of testConfigs.indexConfigs) {
      describe(`Dimension: ${dimension}, Index: ${indexConfig.type} (lists=${indexConfig.lists})`, () => {
        it(
          'measures recall with different dataset sizes',
          async () => {
            for (const size of testConfigs.sizes) {
              console.log(`\nTesting recall with size ${size}...`);

              // Generate test vectors and completely separate query vectors
              const testVectors = generateRandomVectors(size, dimension);
              const numQueries = 10;
              const queryVectors = generateRandomVectors(numQueries, dimension);

              // Create index and insert vectors
              const lists = typeof indexConfig.lists === 'function' ? indexConfig.lists(size) : indexConfig.lists;
              const config = { type: indexConfig.type, lists };
              await vectorDB.createIndex(testIndexName, dimension, 'cosine', config);

              // Use simple string IDs and store index in metadata
              const vectorIds = testVectors.map((_, idx) => `vec_${idx}`);
              const metadata = testVectors.map((_, idx) => ({ index: idx }));
              await vectorDB.upsert(testIndexName, testVectors, vectorIds, metadata);

              // Calculate recall for each query vector
              const recalls: number[] = [];
              for (const queryVector of queryVectors) {
                // Find nearest neighbors using brute force
                const expectedNeighbors = findNearestBruteForce(queryVector, testVectors, 10);

                // Query using index
                const actualResults = await vectorDB.query(testIndexName, queryVector, 10);
                const actualNeighbors = actualResults.map(r => JSON.parse(r.id).index);

                recalls.push(calculateRecall(actualNeighbors, expectedNeighbors));
              }

              // Store average recall across all queries
              results.push({
                dimension,
                indexConfig,
                size,
                metrics: {
                  recall: mean(recalls),
                  minRecall: min(recalls),
                  maxRecall: max(recalls),
                },
              });
            }
          },
          TEST_TIMEOUT,
        );

        it(
          'measures dimension-specific query latency',
          async () => {
            for (const size of testConfigs.sizes) {
              console.log(`\nTesting latency with size ${size}...`);

              const testVectors = generateRandomVectors(size, dimension);
              const queryVectors = generateRandomVectors(50, dimension);

              const lists = typeof indexConfig.lists === 'function' ? indexConfig.lists(size) : indexConfig.lists;
              const config = { type: indexConfig.type, lists };

              await vectorDB.createIndex(testIndexName, dimension, 'cosine', config);
              await vectorDB.upsert(testIndexName, testVectors);

              // Warm up
              await vectorDB.query(testIndexName, queryVectors[0], 10);

              const latencies: number[] = [];
              for (const queryVector of queryVectors) {
                const start = process.hrtime.bigint();
                await vectorDB.query(testIndexName, queryVector, 10);
                const end = process.hrtime.bigint();
                latencies.push(Number(end - start) / 1e6);
              }

              const sorted = [...latencies].sort((a, b) => a - b);
              results.push({
                dimension,
                indexConfig,
                size,
                metrics: {
                  latency: {
                    p50: sorted[Math.floor(sorted.length * 0.5)],
                    p95: sorted[Math.floor(sorted.length * 0.95)],
                    p99: sorted[Math.floor(sorted.length * 0.99)],
                  },
                },
              });
            }
          },
          TEST_TIMEOUT,
        );

        it(
          'measures cluster distribution',
          async () => {
            for (const size of testConfigs.sizes) {
              console.log(`\nTesting cluster distribution with size ${size}...`);

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
                metrics: {
                  clustering: {
                    numLists: lists,
                    avgVectorsPerList: size / lists,
                  },
                },
              });
            }
          },
          TEST_TIMEOUT,
        );
      });
    }
  }
});

const analyzeResults = (results: TestResult[]) => {
  const byDimension = groupBy(results, 'dimension');

  Object.entries(byDimension).forEach(([dimension, dimensionResults]) => {
    console.log(`\n=== Analysis for ${dimension} dimensions ===`);

    // For recall analysis
    const recalls = dimensionResults
      .filter(r => r.metrics.recall !== undefined)
      .map(r => ({
        size: r.size,
        lists: typeof r.indexConfig.lists === 'function' ? r.indexConfig.lists(r.size) : r.indexConfig.lists,
        recall: r.metrics.recall!,
        minRecall: r.metrics.minRecall!,
        maxRecall: r.metrics.maxRecall!,
      }));

    console.log('\nRecall Analysis by Dataset Size:');
    console.table(
      groupBy(recalls, 'size', result => ({
        'Dataset Size': result[0].size,
        'Min Recall': result[0].minRecall.toFixed(3),
        'Avg Recall': mean(result.map(r => r.recall)).toFixed(3),
        'Max Recall': result[0].maxRecall.toFixed(3),
        Lists: result[0].lists,
        'Vectors/List': Math.round(result[0].size / result[0].lists),
      })),
    );

    // For latency analysis
    const latencies = dimensionResults
      .filter(r => r.metrics.latency !== undefined)
      .map(r => ({
        size: r.size,
        lists: typeof r.indexConfig.lists === 'function' ? r.indexConfig.lists(r.size) : r.indexConfig.lists,
        p50: r.metrics.latency!.p50,
        p95: r.metrics.latency!.p95,
      }));

    console.log('\nLatency Analysis by Dataset Size:');
    console.table(
      groupBy(latencies, 'size', result => ({
        'Dataset Size': result[0].size,
        'P50 (ms)': result[0].p50.toFixed(2),
        'P95 (ms)': result[0].p95.toFixed(2),
        Lists: result[0].lists,
        'Vectors/List': Math.round(result[0].size / result[0].lists),
      })),
    );

    // For clustering analysis
    const clustering = dimensionResults
      .filter(r => r.metrics.clustering !== undefined)
      .map(r => ({
        size: r.size,
        lists: typeof r.indexConfig.lists === 'function' ? r.indexConfig.lists(r.size) : r.indexConfig.lists,
        avgVectorsPerList: r.size / r.lists,
      }));

    console.log('\nClustering Analysis by Dataset Size:');
    console.table(
      groupBy(clustering, 'size', result => ({
        'Dataset Size': result[0].size,
        'Vectors/List': Math.round(result[0].size / result[0].lists),
        'Recommended Lists': Math.floor(Math.sqrt(result[0].size)),
        'Actual Lists': result[0].lists,
        Efficiency: (result[0].lists / Math.floor(Math.sqrt(result[0].size))).toFixed(2),
      })),
    );
  });
};

const groupBy = <T, K extends keyof T>(array: T[], key: K, reducer?: (group: T[]) => any): Record<string, any> => {
  const grouped = array.reduce(
    (acc, item) => {
      const value = item[key];
      if (!acc[value as any]) acc[value as any] = [];
      acc[value as any].push(item);
      return acc;
    },
    {} as Record<string, T[]>,
  );

  if (reducer) {
    Object.keys(grouped).forEach(key => {
      grouped[key] = reducer(grouped[key]);
    });
  }

  return grouped;
};

const mean = (arr: number[]) => arr.reduce((a, b) => a + b, 0) / arr.length;

const min = (arr: number[]) => Math.min(...arr);

const max = (arr: number[]) => Math.max(...arr);
