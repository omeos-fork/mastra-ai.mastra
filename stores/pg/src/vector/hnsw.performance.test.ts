import { describe, it, expect, beforeAll, afterAll, beforeEach } from 'vitest';

import {
  TestResult,
  baseTestConfigs,
  generateRandomVectors,
  findNearestBruteForce,
  calculateRecall,
  formatTable,
  groupBy,
  mean,
  min,
  max,
  calculateTimeout,
  setupTestDB,
  cleanupTestDB,
  HOOK_TIMEOUT,
  getDimensionTimeout,
  warmupQuery,
  measureLatency,
} from './performance.helpers';
import { IndexConfig } from './types';

import { PgVector } from '.';

describe('PostgreSQL HNSW Index Performance', () => {
  const testConfigs = {
    ...baseTestConfigs,
    indexConfigs: [
      // Test HNSW with default settings
      {
        type: 'hnsw',
        hnsw: {
          m: 16, // Default connections
          efConstruction: 64, // Default build complexity
        },
      },
      // Test HNSW with higher quality settings
      {
        type: 'hnsw',
        hnsw: {
          m: 32, // More connections
          efConstruction: 128, // Higher build quality
        },
      },
      // Test HNSW with maximum quality settings
      {
        type: 'hnsw',
        hnsw: {
          m: 64, // Maximum connections
          efConstruction: 256, // Maximum build quality
        },
      },
    ],
  };

  let vectorDB: PgVector;
  const testIndexName = 'test_index_performance_hnsw';
  const results: TestResult[] = [];

  beforeAll(async () => {
    vectorDB = await setupTestDB(testIndexName);
  }, HOOK_TIMEOUT);

  beforeEach(async () => {
    await vectorDB.deleteIndex(testIndexName);
  });

  afterAll(async () => {
    await cleanupTestDB(vectorDB, testIndexName);
    analyzeResults(results);
  });

  // For each test, we'll try different ef values relative to k
  const getSearchEf = (k: number, m: number) => ({
    default: Math.max(k, m * k), // Default calculation
    lower: Math.max(k, (m * k) / 2), // Lower quality, faster
    higher: Math.max(k, m * k * 2), // Higher quality, slower
  });

  for (const dimension of testConfigs.dimensions) {
    for (const indexConfig of testConfigs.indexConfigs) {
      const indexDesc = `HNSW(m=${indexConfig.hnsw?.m},ef=${indexConfig.hnsw?.efConstruction})`;

      describe(`Dimension: ${dimension}, Index: ${indexDesc}`, () => {
        const dimensionTimeout = getDimensionTimeout(dimension);

        it(
          'measures recall with different dataset sizes',
          async () => {
            for (const size of testConfigs.sizes) {
              console.log(`Testing recall with size ${size}...`);

              const testVectors = generateRandomVectors(size, dimension);
              const numQueries = 10;
              const queryVectors = generateRandomVectors(numQueries, dimension);

              // Create index and insert vectors
              await vectorDB.createIndex(testIndexName, dimension, 'cosine', indexConfig);
              const vectorIds = testVectors.map((_, idx) => `vec_${idx}`);
              const metadata = testVectors.map((_, idx) => ({ index: idx }));
              await vectorDB.upsert(testIndexName, testVectors, vectorIds, metadata);

              // Test each k value
              for (const k of testConfigs.kValues) {
                const recalls: number[] = [];

                for (const queryVector of queryVectors) {
                  const expectedNeighbors = findNearestBruteForce(queryVector, testVectors, k);

                  // Test different ef values
                  const efValues = getSearchEf(k, indexConfig.hnsw?.m || 16);
                  for (const [efType, ef] of Object.entries(efValues)) {
                    const actualResults = await vectorDB.query(testIndexName, queryVector, k, undefined, false, 0, ef);
                    const actualNeighbors = actualResults.map(r => JSON.parse(r.id).index);
                    const recall = calculateRecall(actualNeighbors, expectedNeighbors, k);
                    recalls.push(recall);

                    results.push({
                      dimension,
                      indexConfig: {
                        ...indexConfig,
                        hnsw: {
                          ...indexConfig.hnsw,
                          ef,
                          efType,
                        },
                      },
                      size,
                      k,
                      metrics: {
                        recall,
                        minRecall: recall,
                        maxRecall: recall,
                      },
                    });
                  }
                }

                // Add aggregate metrics after all queries for this k value
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
          'measures latency',
          async () => {
            for (const size of testConfigs.sizes) {
              for (const k of testConfigs.kValues) {
                const timeout = calculateTimeout(dimension, size, k);
                console.log(`Testing latency with size ${size} and k=${k} (timeout: ${timeout}ms)...`);

                const testVectors = generateRandomVectors(size, dimension);
                const queryVectors = generateRandomVectors(50, dimension);

                await vectorDB.createIndex(testIndexName, dimension, 'cosine', indexConfig);
                await vectorDB.upsert(testIndexName, testVectors);
                await warmupQuery(vectorDB, testIndexName, dimension, k);

                for (const queryVector of queryVectors) {
                  const efValues = getSearchEf(k, indexConfig.hnsw?.m || 16);

                  for (const [efType, ef] of Object.entries(efValues)) {
                    const latency = await measureLatency(() =>
                      vectorDB.query(testIndexName, queryVector, k, undefined, false, 0, ef),
                    );

                    results.push({
                      dimension,
                      indexConfig: {
                        ...indexConfig,
                        hnsw: {
                          ...indexConfig.hnsw,
                          ef,
                          efType,
                        },
                      },
                      size,
                      k,
                      metrics: {
                        latency: {
                          p50: latency,
                          p95: latency,
                          m: indexConfig.hnsw?.m,
                          ef,
                        },
                      },
                    });
                  }
                }
              }
            }
          },
          calculateTimeout(dimension, Math.max(...testConfigs.sizes), Math.max(...testConfigs.kValues)),
        );
      });
    }
  }
});

function analyzeResults(results: TestResult[]) {
  const byDimension = groupBy(results, 'dimension');

  Object.entries(byDimension).forEach(([dimension, dimensionResults]) => {
    console.log(`\n=== Analysis for ${dimension} dimensions ===\n`);

    // Recall Analysis
    const recalls = dimensionResults
      .filter(r => r.metrics.recall !== undefined)
      .map(r => ({
        size: r.size,
        k: r.k,
        m: r.indexConfig.hnsw?.m,
        ef: r.indexConfig.hnsw?.ef,
        recall: r.metrics.recall!,
        minRecall: r.metrics.minRecall!,
        maxRecall: r.metrics.maxRecall!,
      }));

    console.log('Recall Analysis:');
    const recallColumns = ['Dataset Size', 'K', 'M', 'EF', 'Min Recall', 'Avg Recall', 'Max Recall'];

    const recallData = Object.values(
      groupBy(
        recalls,
        r => `${r.size}-${r.k}-${r.m}-${r.ef}`,
        result => ({
          'Dataset Size': result[0].size,
          K: result[0].k,
          M: result[0].m ?? '-',
          EF: result[0].ef ?? '-',
          'Min Recall': result[0].minRecall.toFixed(3),
          'Avg Recall': mean(result.map(r => r.recall)).toFixed(3),
          'Max Recall': result[0].maxRecall.toFixed(3),
        }),
      ),
    );
    console.log(formatTable(recallData, recallColumns));

    // Latency Analysis
    const latencies = dimensionResults
      .filter(r => r.metrics.latency !== undefined)
      .map(r => ({
        size: r.size,
        k: r.k,
        m: r.metrics.latency?.m,
        ef: r.metrics.latency?.ef,
        p50: r.metrics.latency!.p50,
        p95: r.metrics.latency!.p95,
      }));

    console.log('\nLatency Analysis:');
    const latencyColumns = ['Dataset Size', 'K', 'M', 'EF', 'P50 (ms)', 'P95 (ms)'];

    const latencyData = Object.values(
      groupBy(
        latencies,
        r => `${r.size}-${r.k}-${r.m}-${r.ef}`,
        result => ({
          'Dataset Size': result[0].size,
          K: result[0].k,
          M: result[0].m ?? '-',
          EF: result[0].ef ?? '-',
          'P50 (ms)': mean(result.map(r => r.p50)).toFixed(2),
          'P95 (ms)': mean(result.map(r => r.p95)).toFixed(2),
        }),
      ),
    );
    console.log(formatTable(latencyData, latencyColumns));
  });
}
