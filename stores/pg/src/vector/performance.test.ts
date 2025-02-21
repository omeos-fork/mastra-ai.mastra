import { describe, it, beforeAll, afterAll, beforeEach } from 'vitest';

import {
  baseTestConfigs,
  TestConfig,
  TestResult,
  setupTestDB,
  cleanupTestDB,
  calculateTimeout,
  generateRandomVectors,
  findNearestBruteForce,
  calculateRecall,
  formatTable,
  groupBy,
  mean,
  min,
  max,
  warmupQuery,
  measureLatency,
  getIndexDescription,
  getListCount,
  getSearchEf,
} from './performance.helpers';
import { IndexConfig } from './types';

import { PgVector } from '.';

const warmupCache = new Map<string, boolean>();
async function smartWarmup(vectorDB: PgVector, testIndexName: string, dimension: number, k: number) {
  const cacheKey = `${dimension}-${k}-${indexType}`;
  if (!warmupCache.has(cacheKey)) {
    console.log(`Warming up ${indexType} index for ${dimension}d vectors, k=${k}`);
    await warmupQuery(vectorDB, testIndexName, dimension, k);
    warmupCache.set(cacheKey, true);
  }
}

describe('PostgreSQL Index Performance', () => {
  let vectorDB: PgVector;
  const testIndexName = 'test_index_performance';
  const results: TestResult[] = [];

  // IVF and HNSW specific configs
  const indexConfigs = [
    { type: 'flat' }, // Test flat/linear search as baseline
    { type: 'ivfflat', ivf: { lists: 100 } }, // Test IVF with fixed lists
    { type: 'ivfflat', ivf: { lists: (size: number) => Math.sqrt(size) } },
    { type: 'hnsw', m: 16, efConstruction: 64 }, // Default settings
    { type: 'hnsw', m: 64, efConstruction: 256 }, // Maximum quality
  ];

  beforeAll(async () => {
    vectorDB = await setupTestDB(testIndexName);
  });

  beforeEach(async () => {
    await vectorDB.deleteIndex(testIndexName);
  });

  afterAll(async () => {
    await cleanupTestDB(vectorDB, testIndexName);
    analyzeResults(results);
  });

  // Combine all test configs
  const allConfigs: TestConfig[] = [
    ...baseTestConfigs.basicTests.dimension,
    ...baseTestConfigs.basicTests.size,
    ...baseTestConfigs.basicTests.k,
    ...baseTestConfigs.practicalTests,
    ...baseTestConfigs.stressTests,
    ...baseTestConfigs.smokeTests,
  ];

  // For each index config
  for (const indexConfig of indexConfigs) {
    describe(`Index: ${getIndexDescription(indexConfig)}`, () => {
      for (const testConfig of allConfigs) {
        const timeout = calculateTimeout(testConfig.dimension, testConfig.size, testConfig.k);
        const testDesc = `dim=${testConfig.dimension} size=${testConfig.size} k=${testConfig.k}`;

        it(
          testDesc,
          async () => {
            const testVectors = generateRandomVectors(testConfig.size, testConfig.dimension);
            const queryVectors = generateRandomVectors(testConfig.queryCount, testConfig.dimension);
            const vectorIds = testVectors.map((_, idx) => `vec_${idx}`);
            const metadata = testVectors.map((_, idx) => ({ index: idx }));

            // Create index and insert vectors
            const lists =
              indexConfig.type === 'ivfflat'
                ? getListCount({ size: testConfig.size, indexConfig, metrics: {} } as TestResult)
                : undefined;

            const config =
              indexConfig.type === 'hnsw'
                ? { type: 'hnsw', hnsw: { m: indexConfig.m, efConstruction: indexConfig.efConstruction } }
                : { type: indexConfig.type, lists };

            await vectorDB.createIndex(testIndexName, testConfig.dimension, 'cosine', config);
            await vectorDB.upsert(testIndexName, testVectors, vectorIds, metadata);
            await smartWarmup(vectorDB, testIndexName, testConfig.dimension, testConfig.k);

            // For HNSW, test different EF values
            const efValues =
              indexConfig.type === 'hnsw' ? getSearchEf(testConfig.k, indexConfig.m) : { default: undefined };

            for (const [efType, ef] of Object.entries(efValues)) {
              const recalls: number[] = [];
              const latencies: number[] = [];

              for (const queryVector of queryVectors) {
                const expectedNeighbors = findNearestBruteForce(queryVector, testVectors, testConfig.k);
                const actualResults = await vectorDB.query(
                  testIndexName,
                  queryVector,
                  testConfig.k,
                  undefined,
                  false,
                  0,
                  ef, // This will be undefined for non-HNSW indexes
                );

                const actualNeighbors = actualResults.map(r => JSON.parse(r.id).index);
                const recall = calculateRecall(actualNeighbors, expectedNeighbors, testConfig.k);
                recalls.push(recall);
                const latency = await measureLatency(() =>
                  vectorDB.query(testIndexName, queryVector, testConfig.k, undefined, false, 0, ef),
                );
                latencies.push(latency);
              }

              const sorted = [...latencies].sort((a, b) => a - b);
              results.push({
                dimension: testConfig.dimension,
                size: testConfig.size,
                k: testConfig.k,
                indexConfig: {
                  ...indexConfig,
                  ...(indexConfig.type === 'ivfflat' && {
                    lists: getListCount({ size: testConfig.size, indexConfig, metrics: {} } as TestResult),
                  }),
                  ...(indexConfig.type === 'hnsw' && {
                    ef,
                    efType,
                  }),
                },
                metrics: {
                  recall: mean(recalls),
                  minRecall: min(recalls),
                  maxRecall: max(recalls),
                  latency: {
                    p50: sorted[Math.floor(sorted.length * 0.5)],
                    p95: sorted[Math.floor(sorted.length * 0.95)],
                    ...(indexConfig.type === 'ivfflat' && {
                      lists,
                      vectorsPerList: Math.round(testConfig.size / (lists || 1)),
                    }),
                    ...(indexConfig.type === 'hnsw' && {
                      m: indexConfig.m,
                      ef,
                    }),
                  },
                  ...(indexConfig.type === 'ivfflat' && {
                    clustering: {
                      numLists: lists,
                      avgVectorsPerList: testConfig.size / (lists || 1),
                      recommendedLists: Math.floor(Math.sqrt(testConfig.size)),
                    },
                  }),
                },
              });
            }
          },
          timeout,
        );
      }
    });
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
          recall: r.metrics.recall!,
          minRecall: r.metrics.minRecall!,
          maxRecall: r.metrics.maxRecall!,
          ...(type === 'ivfflat' && {
            lists: getListCount(r),
            vectorsPerList: Math.round(r.size / (getListCount(r) || 1)),
          }),
          ...(type === 'hnsw' && {
            m: r.indexConfig.m,
            ef: r.indexConfig.ef,
            efType: r.indexConfig.efType,
          }),
        }));

      console.log('Recall Analysis:');
      const recallColumns = ['Dataset Size', 'K'];
      if (type === 'hnsw') {
        recallColumns.push('M', 'EF', 'EF Type');
      } else if (type === 'ivfflat') {
        recallColumns.push('Lists', 'Vectors/List');
      }
      recallColumns.push('Min Recall', 'Avg Recall', 'Max Recall');

      const recallData = Object.values(
        groupBy(
          recalls,
          r => `${r.size}-${r.k}-${type === 'ivfflat' ? r.lists : `${r.m}-${r.ef}`}`,
          result => ({
            'Dataset Size': result[0].size,
            K: result[0].k,
            ...(type === 'ivfflat'
              ? {
                  Lists: result[0].lists,
                  'Vectors/List': result[0].vectorsPerList,
                }
              : {}),
            ...(type === 'hnsw'
              ? {
                  M: result[0].m,
                  EF: result[0].ef,
                  'EF Type': result[0].efType,
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
          ...(type === 'ivfflat' && {
            lists: getListCount(r),
            vectorsPerList: Math.round(r.size / (getListCount(r) || 1)),
          }),
          ...(type === 'hnsw' && {
            m: r.indexConfig.m,
            ef: r.indexConfig.ef,
            efType: r.indexConfig.efType,
          }),
          p50: r.metrics.latency!.p50,
          p95: r.metrics.latency!.p95,
        }));

      console.log('\nLatency Analysis:');
      const latencyColumns = ['Dataset Size', 'K'];
      if (type === 'hnsw') {
        latencyColumns.push('M', 'EF', 'EF Type');
      } else if (type === 'ivfflat') {
        latencyColumns.push('Lists', 'Vectors/List');
      }
      latencyColumns.push('P50 (ms)', 'P95 (ms)');

      const latencyData = Object.values(
        groupBy(
          latencies,
          r => `${r.size}-${r.k}-${type === 'ivfflat' ? r.lists : `${r.m}-${r.ef}`}`,
          result => ({
            'Dataset Size': result[0].size,
            K: result[0].k,
            ...(type === 'ivfflat'
              ? {
                  Lists: result[0].lists,
                  'Vectors/List': result[0].vectorsPerList,
                }
              : {}),
            ...(type === 'hnsw'
              ? {
                  M: result[0].m,
                  EF: result[0].ef,
                  'EF Type': result[0].efType,
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
