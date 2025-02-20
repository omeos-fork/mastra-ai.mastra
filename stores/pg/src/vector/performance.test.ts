import { describe, it, expect, beforeAll, afterAll, beforeEach, afterEach } from 'vitest';

import { IndexConfig } from './types';

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
      lists?: number; // Only for IVF
      vectorsPerList?: number; // Only for IVF
    };
    clustering?: {
      // Only for IVF
      numLists?: number;
      avgVectorsPerList?: number;
    };
  };
}

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

describe('PostgreSQL Vector Index Performance', () => {
  const testConfigs = {
    dimensions: [64, 384, 1024],
    // sizes: [100, 1000, 10000],
    // kValues: [10, 25, 50, 75, 100],
    sizes: [100],
    kValues: [10],
    indexConfigs: [
      // Test flat/linear search as baseline
      // { type: 'flat' },
      // Test IVF with dynamic lists (sqrt(N))
      // { type: 'ivfflat' },
      // Test IVF with fixed lists
      // { type: 'ivfflat', ivf: { lists: 100 } },
      // Test IVF with size-proportional lists
      // { type: 'ivfflat', ivf: { lists: (size: number) => Math.floor(size / 10) } },
      // Test HNSW with default settings
      {
        type: 'hnsw',
        hnsw: {
          m: 16, // Default connections
          efConstruction: 64, // Default build complexity
        },
      },

      // Test HNSW with higher quality settings
      // {
      //   type: 'hnsw',
      //   hnsw: {
      //     m: 32, // More connections
      //     efConstruction: 128, // Higher build quality
      //   },
      // },

      // // Test HNSW with maximum quality settings
      // {
      //   type: 'hnsw',
      //   hnsw: {
      //     m: 64, // Maximum connections
      //     efConstruction: 256, // Maximum build quality
      //   },
      // },
    ],
  };

  let vectorDB: PgVector;
  const testIndexName = 'test_index_performance';
  const results: TestResult[] = [];
  const connectionString = process.env.DB_URL || `postgresql://postgres:postgres@localhost:5434/mastra`;

  const HOOK_TIMEOUT = 600000;

  beforeAll(async () => {
    vectorDB = new PgVector(connectionString);
    await vectorDB.pool.query('CREATE EXTENSION IF NOT EXISTS vector;');

    // Configure memory settings for the session
    await vectorDB.pool.query(`
      SET maintenance_work_mem = '512MB';
      SET work_mem = '256MB';
      SET temp_buffers = '256MB';
    `);
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
    // Base timeout of 600 seconds
    let timeout = 600000;

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

  // For each test, we'll try different ef values relative to k
  const getSearchEf = (k: number, m: number) => ({
    default: Math.max(k, m * k), // Default calculation
    lower: Math.max(k, (m * k) / 2), // Lower quality, faster
    higher: Math.max(k, m * k * 2), // Higher quality, slower
  });

  for (const dimension of testConfigs.dimensions) {
    for (const indexConfig of testConfigs.indexConfigs) {
      const indexDesc =
        indexConfig.type === 'hnsw'
          ? `HNSW(m=${indexConfig.hnsw?.m},ef=${indexConfig.hnsw?.ef})`
          : indexConfig.type === 'ivfflat' && typeof indexConfig.ivf?.lists === 'function'
            ? 'IVF(N/10)'
            : indexConfig.type === 'ivfflat' && indexConfig.ivf?.lists
              ? `IVF(lists=${indexConfig.ivf.lists})`
              : indexConfig.type === 'ivfflat'
                ? 'IVF(dynamic)'
                : 'Flat';

      describe(`Dimension: ${dimension}, Index: ${indexDesc}`, () => {
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
              const lists = getListCount({ size, indexConfig, metrics: {} } as TestResult);
              const config = { type: indexConfig.type, lists };
              await vectorDB.createIndex(testIndexName, dimension, 'cosine', config);

              const vectorIds = testVectors.map((_, idx) => `vec_${idx}`);
              const metadata = testVectors.map((_, idx) => ({ index: idx }));
              await vectorDB.upsert(testIndexName, testVectors, vectorIds, metadata);

              // Test each k value
              for (const k of testConfigs.kValues) {
                const recalls: number[] = [];
                const lists = getListCount({ size, indexConfig, k, metrics: {} } as TestResult);

                for (const queryVector of queryVectors) {
                  const expectedNeighbors = findNearestBruteForce(queryVector, testVectors, k);

                  // Test different ef values if using HNSW
                  let actualResults;
                  if (indexConfig.type === 'hnsw' && indexConfig.hnsw?.m) {
                    const efValues = getSearchEf(k, indexConfig.hnsw.m);

                    // Test each ef value
                    for (const [efType, ef] of Object.entries(efValues)) {
                      actualResults = await vectorDB.query(testIndexName, queryVector, k, undefined, false, 0, ef);
                      const actualNeighbors = actualResults.map(r => JSON.parse(r.id).index);
                      const recall = calculateRecall(actualNeighbors, expectedNeighbors, k);
                      recalls.push(recall);

                      // Add ef value to metrics with all recall metrics
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
                          minRecall: recall, // For individual queries, min=max=recall
                          maxRecall: recall,
                        },
                      });
                    }
                  } else {
                    // Non-HNSW index
                    actualResults = await vectorDB.query(testIndexName, queryVector, k);
                    const actualNeighbors = actualResults.map(r => JSON.parse(r.id).index);
                    recalls.push(calculateRecall(actualNeighbors, expectedNeighbors, k));
                  }
                }

                // After all queries, add aggregate metrics
                results.push({
                  dimension,
                  indexConfig: {
                    ...indexConfig,
                    lists,
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
          'measures dimension-specific query latency',
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

                // Warm up
                await vectorDB.query(testIndexName, queryVectors[0], k);

                const latencies: { size: number; k: number; m?: number; ef?: number; p50: number; p95: number }[] = [];
                for (const queryVector of queryVectors) {
                  if (indexConfig.type === 'hnsw' && indexConfig.hnsw?.m) {
                    const efValues = getSearchEf(k, indexConfig.hnsw.m);

                    // Test each ef value
                    for (const [efType, ef] of Object.entries(efValues)) {
                      const start = process.hrtime.bigint();
                      await vectorDB.query(testIndexName, queryVector, k, undefined, false, 0, ef);
                      const end = process.hrtime.bigint();
                      const latency = Number(end - start) / 1e6;

                      latencies.push({
                        size,
                        k,
                        m: indexConfig.hnsw.m,
                        ef,
                        p50: latency,
                        p95: latency,
                      });
                    }
                  } else {
                    const start = process.hrtime.bigint();
                    await vectorDB.query(testIndexName, queryVector, k);
                    const end = process.hrtime.bigint();
                    latencies.push({
                      size,
                      k,
                      lists: getListCount({ size, indexConfig, k, metrics: {} } as TestResult),
                      vectorsPerList: Math.round(
                        size / (getListCount({ size, indexConfig, k, metrics: {} } as TestResult) || 1),
                      ),
                      p50: Number(end - start) / 1e6,
                      p95: Number(end - start) / 1e6,
                    });
                  }
                }

                if (indexConfig.type !== 'hnsw') {
                  const sorted = [...latencies].sort((a, b) => a.p50 - b.p50);
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
                        vectorsPerList: Math.round(size / lists),
                      },
                    },
                  });
                }
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
  const colWidths = columns.map(col =>
    Math.max(
      col.length,
      ...data.map(row => {
        const value = row[col];
        return value === undefined || value === null ? '-'.length : value.toString().length;
      }),
    ),
  );

  // Create the table components
  const topBorder = '┌' + colWidths.map(w => '─'.repeat(w)).join('┬') + '┐';
  const headerSeparator = '├' + colWidths.map(w => '─'.repeat(w)).join('┼') + '┤';
  const bottomBorder = '└' + colWidths.map(w => '─'.repeat(w)).join('┴') + '┘';

  // Format header and rows with proper borders
  const header = '│' + columns.map((col, i) => col.padEnd(colWidths[i])).join('│') + '│';
  const rows = data.map(
    row =>
      '│' +
      columns
        .map((col, i) => {
          const value = row[col];
          // Handle undefined/null values
          const displayValue = value === undefined || value === null ? '-' : value.toString();
          return displayValue.padEnd(colWidths[i]);
        })
        .join('│') +
      '│',
  );

  return [topBorder, header, headerSeparator, ...rows, bottomBorder].join('\n');
};

const getGroupKey = (result: any, type: string) => {
  if (type === 'ivfflat') {
    return `${result.size}-${result.k}-${result.lists}`;
  } else if (type === 'hnsw') {
    return `${result.size}-${result.k}-${result.m}-${result.ef}`;
  }
  return `${result.size}-${result.k}`;
};

const analyzeResults = (results: TestResult[]) => {
  // Keep the explanations at the top
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
          vectorsPerList: Math.round(r.size / getListCount(r)),
          m: r.indexConfig.hnsw?.m,
          ef: r.indexConfig.hnsw?.ef,
        }));

      console.log('Recall Analysis:');
      const recallColumns = ['Dataset Size', 'K'];
      if (type === 'ivfflat') {
        recallColumns.push('Lists', 'Vectors/List');
      } else if (type === 'hnsw') {
        recallColumns.push('M', 'EF');
      }
      recallColumns.push('Min Recall', 'Avg Recall', 'Max Recall');

      const recallData = Object.values(
        groupBy(
          recalls,
          r => getGroupKey(r, type),
          result => ({
            'Dataset Size': result[0].size,
            K: result[0].k,
            'Min Recall': result[0].minRecall.toFixed(3),
            'Avg Recall': mean(result.map(r => r.recall)).toFixed(3),
            'Max Recall': result[0].maxRecall.toFixed(3),
            ...(type === 'ivfflat'
              ? {
                  Lists: result[0].lists,
                  'Vectors/List': result[0].vectorsPerList,
                }
              : type === 'hnsw'
                ? {
                    M: result[0].m,
                    EF: result[0].ef,
                  }
                : {}),
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
          vectorsPerList: Math.round(r.size / getListCount(r)),
          m: r.indexConfig.hnsw?.m,
          ef: r.indexConfig.hnsw?.ef,
        }));

      console.log('\nLatency Analysis:');
      const latencyColumns = ['Dataset Size', 'K'];
      if (type === 'ivfflat') {
        latencyColumns.push('Lists', 'Vectors/List');
      } else if (type === 'hnsw') {
        latencyColumns.push('M', 'EF');
      }
      latencyColumns.push('P50 (ms)', 'P95 (ms)');

      const latencyData = Object.values(
        groupBy(
          latencies,
          r => getGroupKey(r, type),
          result => ({
            'Dataset Size': result[0].size,
            K: result[0].k,
            ...(type === 'ivfflat'
              ? {
                  Lists: result[0].lists,
                  'Vectors/List': result[0].vectorsPerList,
                }
              : type === 'hnsw'
                ? {
                    M: result[0].m ?? '-',
                    EF: result[0].ef ?? '-',
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
