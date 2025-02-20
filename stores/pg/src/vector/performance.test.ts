import { describe, it, expect, beforeAll, afterAll, beforeEach, afterEach } from 'vitest';

import { PgVector } from '.';

const generateRandomVectors = (count: number, dim: number) => {
  // Create a base vector for testing
  const baseVector = Array.from({ length: dim }, () => Math.random() - 0.5);
  const baseMagnitude = Math.sqrt(baseVector.reduce((sum, val) => sum + val * val, 0));
  const normalizedBase = baseVector.map(val => val / baseMagnitude);

  return Array.from({ length: count }, (_, i) => {
    if (i === 0) return normalizedBase;

    // Create variations of the base vector
    const variation = Array.from({ length: dim }, () => (Math.random() - 0.5) * 0.2);
    const vector = normalizedBase.map((val, j) => val + variation[j]);

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
      { type: 'ivfflat' },
      // Test fixed configurations
      { type: 'ivfflat', lists: 100 },
      // Test size-proportional configurations
      { type: 'ivfflat', lists: (size: number) => Math.floor(size / 10) }, // N/10 lists
    ],
  };

  let vectorDB: PgVector;
  const testIndexName = 'test_index_performance';
  const results: TestResult[] = [];
  const connectionString = process.env.DB_URL || 'postgresql://postgres:postgres@localhost:5434/mastra';

  const TEST_TIMEOUT = 30000;
  const HOOK_TIMEOUT = 60000;

  beforeAll(async () => {
    vectorDB = new PgVector(connectionString);
    await vectorDB.pool.query('CREATE EXTENSION IF NOT EXISTS vector;');
    // Increase maintenance_work_mem for index creation
    await vectorDB.pool.query('SET maintenance_work_mem = "256MB";');
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

              // Generate test vectors
              const vectors = generateRandomVectors(size, dimension);
              const queryVector = vectors[0];

              // Create index and insert vectors
              const lists = typeof indexConfig.lists === 'function' ? indexConfig.lists(size) : indexConfig.lists;

              const config = {
                type: indexConfig.type,
                lists,
              };

              await vectorDB.createIndex(testIndexName, dimension, 'cosine', config);

              // Use simple string IDs and store index in metadata
              const vectorIds = vectors.map((_, idx) => `vec_${idx}`);
              const metadata = vectors.map((_, idx) => ({ index: idx }));
              await vectorDB.upsert(testIndexName, vectors, vectorIds, metadata);

              // Find nearest neighbors using brute force
              const expectedNeighbors = findNearestBruteForce(queryVector, vectors, 10);

              // Query using index
              const actualResults = await vectorDB.query(testIndexName, queryVector, 10);

              // Parse the index from the ID field which contains our metadata
              const actualNeighbors = actualResults.map(r => {
                const meta = JSON.parse(r.id);
                return meta.index;
              });

              const recall = calculateRecall(actualNeighbors, expectedNeighbors);

              results.push({
                dimension,
                indexConfig,
                size,
                metrics: { recall },
              });

              console.log(`
                Dimension: ${dimension}
                Config: ${indexConfig.type} (lists=${indexConfig.lists})
                Dataset size: ${size}
                Recall: ${recall.toFixed(2)}
              `);
            }
          },
          TEST_TIMEOUT,
        );

        it(
          'measures dimension-specific query latency',
          async () => {
            const size = testConfigs.sizes[0];
            console.log(`\nTesting latency with size ${size}...`);

            const vectors = generateRandomVectors(size, dimension);
            const queryVector = generateRandomVectors(1, dimension)[0];

            // Create index and insert vectors
            const lists = typeof indexConfig.lists === 'function' ? indexConfig.lists(size) : indexConfig.lists;

            const config = {
              type: indexConfig.type,
              lists,
            };

            await vectorDB.createIndex(testIndexName, dimension, 'cosine', config);
            await vectorDB.upsert(testIndexName, vectors);

            console.log('Warming up with initial query...');
            await vectorDB.query(testIndexName, queryVector, 10);

            const numQueries = 50;
            const latencies: number[] = [];

            for (let i = 0; i < numQueries; i++) {
              const start = process.hrtime.bigint();
              await vectorDB.query(testIndexName, queryVector, 10);
              const end = process.hrtime.bigint();
              latencies.push(Number(end - start) / 1e6);
            }

            const sorted = [...latencies].sort((a, b) => a - b);
            const p50 = sorted[Math.floor(sorted.length * 0.5)];
            const p95 = sorted[Math.floor(sorted.length * 0.95)];
            const p99 = sorted[Math.floor(sorted.length * 0.99)];

            results.push({
              dimension,
              indexConfig,
              size,
              metrics: { latency: { p50, p95, p99 } },
            });

            console.log(`
            Dimension: ${dimension}
            Config: ${indexConfig.type} (lists=${indexConfig.lists})
            Dataset size: ${size}
            P50 latency: ${p50.toFixed(2)}ms
            P95 latency: ${p95.toFixed(2)}ms
            P99 latency: ${p99.toFixed(2)}ms
          `);
          },
          TEST_TIMEOUT,
        );

        it(
          'measures cluster distribution',
          async () => {
            const size = testConfigs.sizes[0];
            console.log(`\nTesting cluster distribution with size ${size}...`);

            const vectors = generateRandomVectors(size, dimension);

            // Create index and insert vectors
            const lists =
              typeof indexConfig.lists === 'function'
                ? indexConfig.lists(size)
                : (indexConfig.lists ?? Math.floor(Math.sqrt(size)));

            const config = {
              type: indexConfig.type,
              lists,
            };

            await vectorDB.createIndex(testIndexName, dimension, 'cosine', config);
            await vectorDB.upsert(testIndexName, vectors);

            const client = await vectorDB.pool.connect();
            try {
              // Run some sample queries to test distribution
              const sampleQueries = 10;
              const querySizes = [1, 10, 50];

              for (const k of querySizes) {
                const latencies: number[] = [];

                for (let i = 0; i < sampleQueries; i++) {
                  const queryVector = generateRandomVectors(1, dimension)[0];
                  const start = process.hrtime.bigint();
                  await vectorDB.query(testIndexName, queryVector, k);
                  const end = process.hrtime.bigint();
                  latencies.push(Number(end - start) / 1e6);
                }

                const avgLatency = latencies.reduce((a, b) => a + b, 0) / latencies.length;
                console.log(`Average latency for k=${k}: ${avgLatency.toFixed(2)}ms`);
              }

              // Store basic index info
              results.push({
                dimension,
                indexConfig: {
                  ...indexConfig,
                  actualLists: lists,
                },
                size,
                metrics: {
                  clustering: {
                    numLists: lists,
                    avgVectorsPerList: size / lists,
                  },
                },
              });

              console.log(`
                Dimension: ${dimension}
                Config: ${indexConfig.type} (lists=${lists})
                Dataset size: ${size}
                Theoretical vectors per list: ${(size / lists).toFixed(2)}
              `);
            } finally {
              client.release();
            }
          },
          TEST_TIMEOUT,
        );
      });
    }
  }
});

const analyzeResults = (results: TestResult[]) => {
  // Group results by dimension
  const byDimension = groupBy(results, 'dimension');

  Object.entries(byDimension).forEach(([dimension, dimensionResults]) => {
    console.log(`\n=== Analysis for ${dimension} dimensions ===`);

    // Analyze recall
    const recalls = dimensionResults
      .filter(r => r.metrics.recall !== undefined)
      .map(r => ({
        size: r.size,
        lists: r.indexConfig.lists,
        recall: r.metrics.recall!,
      }));

    console.log('\nRecall Analysis:');
    console.table(
      groupBy(recalls, 'size', result => ({
        'min recall': min(result.map(r => r.recall)),
        'max recall': max(result.map(r => r.recall)),
        'avg recall': mean(result.map(r => r.recall)),
        'best lists': result.reduce((a, b) => (a.recall > b.recall ? a : b)).lists,
      })),
    );

    // Analyze latency with more detail
    const latencies = dimensionResults
      .filter(r => r.metrics.latency !== undefined)
      .map(r => ({
        size: r.size,
        lists: r.indexConfig.lists,
        p50: r.metrics.latency!.p50,
        p95: r.metrics.latency!.p95,
        p99: r.metrics.latency!.p99,
        listsToVectorRatio: r.indexConfig.lists / r.size,
      }));

    console.log('\nLatency Analysis (ms):');
    console.table(
      groupBy(latencies, 'size', result => ({
        'min p50': min(result.map(r => r.p50)),
        'avg p50': mean(result.map(r => r.p50)),
        'min p95': min(result.map(r => r.p95)),
        'avg p95': mean(result.map(r => r.p95)),
        'lists/vectors ratio': mean(result.map(r => r.listsToVectorRatio)),
        'best lists': result.reduce((a, b) => (a.p95 < b.p95 ? a : b)).lists,
      })),
    );

    // Analyze clustering with focus on list distribution
    const clustering = dimensionResults
      .filter(r => r.metrics.clustering !== undefined)
      .map(r => ({
        size: r.size,
        lists: r.indexConfig.lists,
        avgVectorsPerList: r.size / r.indexConfig.lists,
      }));

    console.log('\nClustering Analysis:');
    console.table(
      groupBy(clustering, 'size', result => ({
        'vectors per list': mean(result.map(r => r.avgVectorsPerList)),
        'recommended lists': Math.floor(Math.sqrt(result[0].size)),
        'actual lists': result[0].lists,
        'efficiency ratio': result[0].lists / Math.floor(Math.sqrt(result[0].size)),
      })),
    );

    // Analyze incremental updates
    const incrementalRecalls = dimensionResults
      .filter(r => r.metrics.incrementalRecall !== undefined)
      .map(r => ({
        lists: r.indexConfig.lists,
        ...r.metrics.incrementalRecall!,
      }));

    if (incrementalRecalls.length > 0) {
      console.log('\nIncremental Update Analysis:');
      console.table(
        groupBy(incrementalRecalls, 'lists', result => ({
          'initial recall': result[0].recall,
          'final recall': result[result.length - 1].recall,
          degradation: result[0].recall - result[result.length - 1].recall,
          'avg recall': mean(result.map(r => r.recall)),
        })),
      );
    }
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
