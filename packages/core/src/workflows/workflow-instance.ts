import type { Span } from '@opentelemetry/api';
import { get } from 'radash';
import sift from 'sift';
import { createActor, type Snapshot, setup, assign, type MachineContext, fromPromise } from 'xstate';
import type { z } from 'zod';

import type { IAction, MastraPrimitives } from '../action';
import type { Logger } from '../logger';

import { Machine } from './machine';
import { Step } from './step';
import type {
  DependencyCheckOutput,
  ResolverFunctionInput,
  ResolverFunctionOutput,
  RetryConfig,
  StepCondition,
  StepDef,
  StepGraph,
  StepNode,
  StepResult,
  StepVariableType,
  WorkflowActionParams,
  WorkflowActions,
  WorkflowActors,
  WorkflowContext,
  WorkflowEvent,
  WorkflowRunState,
  WorkflowState,
} from './types';
import {
  getActivePathsAndStatus,
  getStepResult,
  getSuspendedPaths,
  isErrorEvent,
  isTransitionEvent,
  mergeChildValue,
  recursivelyCheckForFinalState,
} from './utils';

export interface WorkflowResultReturn<T extends z.ZodType<any>> {
  runId: string;
  start: (props?: { triggerData?: z.infer<T> } | undefined) => Promise<{
    triggerData?: z.infer<T>;
    results: Record<string, StepResult<any>>;
    runId: string;
  }>;
}

export class WorkflowInstance<TSteps extends Step<any, any, any>[] = any, TTriggerSchema extends z.ZodType<any> = any>
  implements WorkflowResultReturn<TTriggerSchema>
{
  name: string;
  #mastra?: MastraPrimitives;
  #machine!: Machine<TSteps, TTriggerSchema>;

  logger: Logger;

  #steps: Record<string, IAction<any, any, any, any>> = {};
  #stepGraph: StepGraph;
  #stepSubscriberGraph: Record<string, StepGraph> = {};

  #retryConfig?: RetryConfig;

  #runId: string;
  #executionSpan: Span | undefined;

  #onStepTransition: Set<(state: WorkflowRunState) => void | Promise<void>> = new Set();
  #onFinish?: () => void;

  constructor({
    name,
    logger,
    steps,
    runId,
    retryConfig,
    mastra,
    stepGraph,
    stepSubscriberGraph,
    onStepTransition,
    onFinish,
  }: {
    name: string;
    logger: Logger;
    steps: Record<string, IAction<any, any, any, any>>;
    mastra?: MastraPrimitives;
    retryConfig?: RetryConfig;
    runId?: string;
    stepGraph: StepGraph;
    stepSubscriberGraph: Record<string, StepGraph>;
    onStepTransition: Set<(state: WorkflowRunState) => void | Promise<void>>;
    onFinish?: () => void;
  }) {
    this.name = name;
    this.logger = logger;

    this.#steps = steps;
    this.#stepGraph = stepGraph;
    this.#stepSubscriberGraph = stepSubscriberGraph;

    this.#retryConfig = retryConfig;
    this.#mastra = mastra;

    this.#runId = runId ?? crypto.randomUUID();
    this.#onStepTransition = onStepTransition;
    this.#onFinish = onFinish;
  }

  get runId() {
    return this.#runId;
  }

  async start({ triggerData }: { triggerData?: z.infer<TTriggerSchema> } = {}) {
    const results = await this.execute({ triggerData });

    if (this.#onFinish) {
      this.#onFinish();
    }

    return {
      ...results,
      runId: this.runId,
    };
  }

  async execute({
    triggerData,
    snapshot,
    stepId,
  }: {
    stepId?: string;
    triggerData?: z.infer<TTriggerSchema>;
    snapshot?: Snapshot<any>;
  } = {}): Promise<{
    triggerData?: z.infer<TTriggerSchema>;
    results: Record<string, StepResult<any>>;
  }> {
    this.#executionSpan = this.#mastra?.telemetry?.tracer.startSpan(`workflow.${this.name}.execute`, {
      attributes: { componentName: this.name, runId: this.runId },
    });

    const machineInput = snapshot
      ? (snapshot as any).context
      : {
          // Maintain the original step results and their output
          steps: {},
          triggerData: triggerData || {},
          attempts: Object.keys(this.#steps).reduce(
            (acc, stepKey) => {
              acc[stepKey] = this.#steps[stepKey]?.retryConfig?.attempts || this.#retryConfig?.attempts || 3;
              return acc;
            },
            {} as Record<string, number>,
          ),
        };

    this.#machine = new Machine({
      logger: this.logger,
      mastra: this.#mastra,
      name: this.name,
      runId: this.runId,
      steps: this.#steps,
      onStepTransition: this.#onStepTransition,
      stepGraph: this.#stepGraph,
      executionSpan: this.#executionSpan,
    });

    const nestedMachines: Promise<any>[] = [];
    this.#machine.on('spawn-subscriber', ({ parentStepId, context }) => {
      console.log('event caught', { parentStepId, context });
      if (this.#stepSubscriberGraph[parentStepId]) {
        console.log('spawning nested machine', parentStepId);
        const machine = new Machine({
          logger: this.logger,
          mastra: this.#mastra,
          name: this.name,
          runId: this.runId,
          steps: this.#steps,
          onStepTransition: this.#onStepTransition,
          stepGraph: this.#stepSubscriberGraph[parentStepId],
          executionSpan: this.#executionSpan,
        });

        nestedMachines.push(machine.execute({ input: context }));
      }
    });

    const { results } = await this.#machine.execute({ snapshot, stepId, input: machineInput });
    const nestedResults = (await Promise.all(nestedMachines)).reduce(
      (acc, { results }) => ({ ...acc, ...results }),
      {},
    );
    const allResults = { ...results, ...nestedResults };
    console.dir({ results, nestedResults, allResults }, { depth: null });
    return { results: allResults };
  }

  /**
   * Persists the workflow state to the database
   */
  async #persistWorkflowSnapshot() {
    return this.#machine.persistMachineSnapshot();
  }

  async getState(): Promise<WorkflowRunState | null> {
    if (!this.#machine) {
      return null;
    }

    const snapshot = this.#machine.getSnapshot();
    if (!snapshot) {
      return null;
    }

    const m = getActivePathsAndStatus(snapshot.value as Record<string, any>);
    return {
      runId: this.runId,
      value: snapshot.value as Record<string, string>,
      context: snapshot.context,
      activePaths: m,
      timestamp: Date.now(),
    };
  }
}
