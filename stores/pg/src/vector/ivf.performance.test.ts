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

describe('PostgreSQL IVF Index Performance', () => {
  const testConfigs = {
    ...baseTestConfigs,
    indexConfigs: [
      // Test flat/linear search as baseline
      { type: 'flat' },

      // Test IVF with dynamic lists (sqrt(N))
      { type: 'ivfflat' },

      // Test IVF with fixed lists
      { type: 'ivfflat', ivf: { lists: 100 } },

      // Test IVF with size-proportional lists
      { type: 'ivfflat', ivf: { lists: (size: number) => Math.floor(size / 10) } },
    ],
  };

  let vectorDB: PgVector;
  const testIndexName = 'test_index_performance_ivf';
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

  const getListCount = (result: TestResult): number | undefined => {
    if (result.indexConfig.type !== 'ivfflat') return undefined;
    if (result.metrics.latency?.lists) {
      return result.metrics.latency.lists;
    }
    if (typeof result.indexConfig.ivf?.lists === 'function') {
      return result.indexConfig.ivf.lists(result.size);
    }
    return result.indexConfig.ivf?.lists ?? Math.floor(Math.sqrt(result.size));
  };

  for (const dimension of testConfigs.dimensions) {
    for (const indexConfig of testConfigs.indexConfigs) {
      const indexDesc =
        indexConfig.type === 'ivfflat' && typeof indexConfig.ivf?.lists === 'function'
          ? 'IVF(N/10)'
          : indexConfig.type === 'ivfflat' && indexConfig.ivf?.lists
            ? `IVF(lists=${indexConfig.ivf.lists})`
            : indexConfig.type === 'ivfflat'
              ? 'IVF(dynamic)'
              : 'Flat';

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
              const lists = getListCount({ size, indexConfig, metrics: {} } as TestResult);
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
                  indexConfig: {
                    ...indexConfig,
                    lists: getListCount({ size, indexConfig, k, metrics: {} } as TestResult),
                  },
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

                const lists = getListCount({ size, indexConfig, k, metrics: {} } as TestResult);
                const config = { type: indexConfig.type, lists };
                await vectorDB.createIndex(testIndexName, dimension, 'cosine', config);
                await vectorDB.upsert(testIndexName, testVectors);
                await warmupQuery(vectorDB, testIndexName, dimension, k);

                const latencies: number[] = [];
                for (const queryVector of queryVectors) {
                  const latency = await measureLatency(() => vectorDB.query(testIndexName, queryVector, k));
                  latencies.push(latency);
                }

                const sorted = [...latencies].sort((a, b) => a - b);
                results.push({
                  dimension,
                  indexConfig: {
                    type: indexConfig.type,
                    lists,
                  },
                  size,
                  k,
                  metrics: {
                    latency: {
                      p50: sorted[Math.floor(sorted.length * 0.5)],
                      p95: sorted[Math.floor(sorted.length * 0.95)],
                      lists,
                      vectorsPerList: Math.round(size / (lists || 1)),
                    },
                  },
                });
              }
            }
          },
          calculateTimeout(dimension, Math.max(...testConfigs.sizes), Math.max(...testConfigs.kValues)),
        );

        it(
          'measures cluster distribution',
          async () => {
            for (const size of testConfigs.sizes) {
              console.log(`Testing cluster distribution with size ${size}...`);

              const testVectors = generateRandomVectors(size, dimension);
              const lists = getListCount({ size, indexConfig, metrics: {} } as TestResult);

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
                    avgVectorsPerList: size / (lists || 1),
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

function analyzeResults(results: TestResult[]) {
  const byDimension = groupBy(results, 'dimension');

  Object.entries(byDimension).forEach(([dimension, dimensionResults]) => {
    console.log(`\n=== Analysis for ${dimension} dimensions ===\n`);

    const byType = groupBy(dimensionResults, r => r.indexConfig.type);

    Object.entries(byType).forEach(([type, typeResults]) => {
      console.log(`\n--- ${type.toUpperCase()} Index Analysis ---\n`);

      // Recall Analysis
      const recalls = typeResults
        .filter(r => r.metrics.recall !== undefined)
        .map(r => ({
          size: r.size,
          k: r.k,
          lists: getListCount(r),
          recall: r.metrics.recall!,
          minRecall: r.metrics.minRecall!,
          maxRecall: r.metrics.maxRecall!,
          vectorsPerList: Math.round(r.size / (getListCount(r) || 1)),
        }));

      console.log('Recall Analysis:');
      const recallColumns = ['Dataset Size', 'K'];
      if (type === 'ivfflat') {
        recallColumns.push('Lists', 'Vectors/List');
      }
      recallColumns.push('Min Recall', 'Avg Recall', 'Max Recall');

      const recallData = Object.values(
        groupBy(
          recalls,
          r => `${r.size}-${r.k}-${r.lists}`,
          result => ({
            'Dataset Size': result[0].size,
            K: result[0].k,
            ...(type === 'ivfflat'
              ? {
                  Lists: result[0].lists,
                  'Vectors/List': result[0].vectorsPerList,
                }
              : {}),
            'Min Recall': result[0].minRecall.toFixed(3),
            'Avg Recall': mean(result.map(r => r.recall)).toFixed(3),
            'Max Recall': result[0].maxRecall.toFixed(3),
          }),
        ),
      );
      console.log(formatTable(recallData, recallColumns));

      // Latency Analysis
      const latencies = typeResults
        .filter(r => r.metrics.latency !== undefined)
        .map(r => ({
          size: r.size,
          k: r.k,
          lists: getListCount(r),
          p50: r.metrics.latency!.p50,
          p95: r.metrics.latency!.p95,
          vectorsPerList: Math.round(r.size / (getListCount(r) || 1)),
        }));

      console.log('\nLatency Analysis:');
      const latencyColumns = ['Dataset Size', 'K'];
      if (type === 'ivfflat') {
        latencyColumns.push('Lists', 'Vectors/List');
      }
      latencyColumns.push('P50 (ms)', 'P95 (ms)');

      const latencyData = Object.values(
        groupBy(
          latencies,
          r => `${r.size}-${r.k}-${r.lists}`,
          result => ({
            'Dataset Size': result[0].size,
            K: result[0].k,
            ...(type === 'ivfflat'
              ? {
                  Lists: result[0].lists,
                  'Vectors/List': result[0].vectorsPerList,
                }
              : {}),
            'P50 (ms)': mean(result.map(r => r.p50)).toFixed(2),
            'P95 (ms)': mean(result.map(r => r.p95)).toFixed(2),
          }),
        ),
      );
      console.log(formatTable(latencyData, latencyColumns));

      // Clustering Analysis (only for IVF)
      if (type === 'ivfflat') {
        const clustering = typeResults
          .filter(r => r.metrics.clustering !== undefined)
          .map(r => ({
            size: r.size,
            vectorsPerList: r.metrics.clustering!.avgVectorsPerList,
            recommendedLists: Math.floor(Math.sqrt(r.size)),
            actualLists: r.metrics.clustering!.numLists,
          }));

        console.log('\nClustering Analysis:');
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
      }
    });
  });
}
