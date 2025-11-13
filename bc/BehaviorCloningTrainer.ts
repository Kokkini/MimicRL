/**
 * BehaviorCloningTrainer - Trains policy network via supervised learning
 * Game-agnostic: works with normalized observations and actions
 * 
 * Location: src/MimicRL/bc/BehaviorCloningTrainer.ts (library code)
 */

// TensorFlow.js is loaded from CDN as a global 'tf' object
declare const tf: any;

import { PolicyAgent } from '../agents/PolicyAgent.js';
import { DemonstrationDataset } from './Demonstration.js';

export interface BehaviorCloningTrainerConfig {
  /**
   * Learning rate for optimizer
   */
  learningRate?: number;

  /**
   * Batch size for training
   */
  batchSize?: number;

  /**
   * Number of epochs to train
   */
  epochs?: number;

  /**
   * Loss function type
   * - 'mse': Mean squared error (for continuous actions)
   * - 'crossentropy': Cross-entropy (for discrete actions)
   * - 'mixed': Automatically choose based on action spaces
   */
  lossType?: 'mse' | 'crossentropy' | 'mixed';

  /**
   * Weight decay (L2 regularization)
   */
  weightDecay?: number;

  /**
   * Validation split (0-1, fraction of data to use for validation)
   */
  validationSplit?: number;
}

export interface BCTrainingStats {
  trainLoss: number;
  valLoss: number;
  epoch: number;
  step: number;
}

/**
 * BehaviorCloningTrainer - Trains policy network via supervised learning
 */
export class BehaviorCloningTrainer {
  private config: Required<BehaviorCloningTrainerConfig>;
  private optimizer: any; // tf.Optimizer
  private trainingStats: BCTrainingStats;
  private yieldChannel: MessageChannel;
  private yieldChannelResolve: (() => void) | null;

  constructor(config?: BehaviorCloningTrainerConfig) {
    // Create MessageChannel for non-throttled yielding (works in background tabs)
    this.yieldChannel = new MessageChannel();
    this.yieldChannelResolve = null;
    this.yieldChannel.port1.onmessage = () => {
      if (this.yieldChannelResolve) {
        this.yieldChannelResolve();
        this.yieldChannelResolve = null;
      }
    };
    this.yieldChannel.port2.onmessage = () => {}; // Empty handler

    this.config = {
      learningRate: config?.learningRate ?? 0.001,
      batchSize: config?.batchSize ?? 32,
      epochs: config?.epochs ?? 10,
      lossType: config?.lossType ?? 'mixed',
      weightDecay: config?.weightDecay ?? 0.0001,
      validationSplit: config?.validationSplit ?? 0.2
    };

    // Create optimizer
    this.optimizer = tf.train.adam(this.config.learningRate);

    // Training statistics
    this.trainingStats = {
      trainLoss: 0,
      valLoss: 0,
      epoch: 0,
      step: 0
    };
  }

  /**
   * Train policy network on demonstration dataset
   * @param {DemonstrationDataset} dataset - Dataset of expert demonstrations
   * @param {PolicyAgent} policyAgent - Policy agent to train (updates policyNetwork)
   * @param {Function} onProgress - Optional progress callback (epoch, loss, valLoss) => void
   * @returns {Promise<BCTrainingStats>} Final training statistics
   */
  async train(
    dataset: DemonstrationDataset,
    policyAgent: PolicyAgent,
    onProgress?: (epoch: number, loss: number, valLoss?: number) => void
  ): Promise<BCTrainingStats> {
    // Validate dataset matches policy agent
    this.validateDataset(dataset, policyAgent);

    // Prepare training data
    const { trainData, valData } = this.prepareTrainingData(dataset, policyAgent);

    // Train for specified epochs
    for (let epoch = 0; epoch < this.config.epochs; epoch++) {
      const epochStats = await this.trainEpoch(
        trainData,
        policyAgent,
        epoch
      );

      // Validate if validation split > 0
      let valLoss = 0;
      if (valData && valData.observations.shape[0] > 0) {
        valLoss = this.computeValidationLoss(valData, policyAgent);
      }

      this.trainingStats = {
        trainLoss: epochStats.loss,
        valLoss: valLoss,
        epoch: epoch + 1,
        step: (epoch + 1) * this.config.batchSize
      };

      // Call progress callback
      if (onProgress) {
        onProgress(epoch + 1, epochStats.loss, valLoss);
      }

      // Yield to event loop periodically
      await this.yieldToEventLoop();
    }

    // Clean up tensors
    trainData.observations.dispose();
    trainData.actions.dispose();
    if (valData) {
      valData.observations.dispose();
      valData.actions.dispose();
    }

    return this.trainingStats;
  }

  /**
   * Prepare training data from dataset
   */
  private prepareTrainingData(
    dataset: DemonstrationDataset,
    policyAgent: PolicyAgent
  ): {
    trainData: { observations: any; actions: any }; // tf.Tensor
    valData?: { observations: any; actions: any }; // tf.Tensor
  } {
    // Flatten all episodes into single arrays
    const observations: number[][] = [];
    const actions: number[][] = [];

    for (const episode of dataset.episodes) {
      for (const step of episode.steps) {
        observations.push(step.observation);
        actions.push(step.action);
      }
    }

    // Convert to tensors
    const obsTensor = tf.tensor2d(observations, [observations.length, policyAgent.observationSize]);
    const actTensor = tf.tensor2d(actions, [actions.length, policyAgent.actionSize]);

    // Split into train/validation
    if (this.config.validationSplit > 0) {
      const totalSamples = observations.length;
      const valSize = Math.floor(totalSamples * this.config.validationSplit);
      const trainSize = totalSamples - valSize;

      const trainObs = obsTensor.slice([0, 0], [trainSize, -1]);
      const trainAct = actTensor.slice([0, 0], [trainSize, -1]);
      const valObs = obsTensor.slice([trainSize, 0], [valSize, -1]);
      const valAct = actTensor.slice([trainSize, 0], [valSize, -1]);

      // Dispose original tensors
      obsTensor.dispose();
      actTensor.dispose();

      return {
        trainData: { observations: trainObs, actions: trainAct },
        valData: { observations: valObs, actions: valAct }
      };
    } else {
      return {
        trainData: { observations: obsTensor, actions: actTensor }
      };
    }
  }

  /**
   * Train for one epoch
   */
  private async trainEpoch(
    data: { observations: any; actions: any },
    policyAgent: PolicyAgent,
    epoch: number
  ): Promise<{ loss: number }> {
    const batchSize = this.config.batchSize;
    const totalSamples = data.observations.shape[0];
    const numBatches = Math.ceil(totalSamples / batchSize);

    let totalLoss = 0;

    // Shuffle data (create indices and shuffle)
    const indices = Array.from({ length: totalSamples }, (_, i) => i);
    this.shuffleArray(indices);

    for (let i = 0; i < numBatches; i++) {
      const start = i * batchSize;
      const end = Math.min(start + batchSize, totalSamples);

      // Get batch indices
      const batchIndices = indices.slice(start, end);
      const batchIndicesTensor = tf.tensor1d(batchIndices, 'int32');

      // Create batch tensors
      const batchObs = tf.gather(data.observations, batchIndicesTensor);
      const batchAct = tf.gather(data.actions, batchIndicesTensor);

      // Train on batch
      const loss = await this.trainBatch(batchObs, batchAct, policyAgent);
      totalLoss += loss;

      // Clean up
      batchObs.dispose();
      batchAct.dispose();
      batchIndicesTensor.dispose();

      // Yield periodically
      if (i % 10 === 0) {
        await this.yieldToEventLoop();
      }
    }

    return { loss: totalLoss / numBatches };
  }

  /**
   * Train on a single batch
   */
  private async trainBatch(
    observations: any,
    actions: any,
    policyAgent: PolicyAgent
  ): Promise<number> {
    return tf.tidy(() => {
      // Forward pass
      const predictions = policyAgent.policyNetwork.predict(observations);

      // Compute loss based on action spaces
      const loss = this.computeLoss(predictions, actions, policyAgent);

      // Backward pass
      this.optimizer.minimize(() => {
        const pred = policyAgent.policyNetwork.predict(observations);
        return this.computeLoss(pred, actions, policyAgent);
      });

      const lossValue = loss.dataSync()[0];
      predictions.dispose();
      return lossValue;
    });
  }

  /**
   * Compute loss based on action spaces
   */
  private computeLoss(
    predictions: any,
    targets: any,
    policyAgent: PolicyAgent
  ): any {
    if (this.config.lossType === 'mixed') {
      // Compute loss per action index based on action space type
      const losses: any[] = [];

      for (let i = 0; i < policyAgent.actionSize; i++) {
        const actionSpace = policyAgent.actionSpaces[i];
        const predSlice = predictions.slice([0, i], [-1, 1]).squeeze();
        const targetSlice = targets.slice([0, i], [-1, 1]).squeeze();

        if (actionSpace.type === 'discrete') {
          // For discrete: use sigmoid cross-entropy
          // Note: sigmoidCrossEntropy expects logits (raw outputs), not probabilities
          // It applies sigmoid internally
          const loss = tf.losses.sigmoidCrossEntropy(
            targetSlice,
            predSlice  // Pass logits, not probabilities
          );
          losses.push(loss);
        } else {
          // For continuous: use MSE
          const loss = tf.losses.meanSquaredError(targetSlice, predSlice);
          losses.push(loss);
        }

        predSlice.dispose();
        targetSlice.dispose();
      }

      // Average losses
      let totalLoss = losses[0];
      for (let i = 1; i < losses.length; i++) {
        totalLoss = totalLoss.add(losses[i]);
      }
      const avgLoss = totalLoss.div(tf.scalar(losses.length));
      losses.forEach(l => l.dispose());
      totalLoss.dispose();
      return avgLoss;
    } else if (this.config.lossType === 'mse') {
      return tf.losses.meanSquaredError(targets, predictions);
    } else {
      // crossentropy: sigmoidCrossEntropy expects logits (raw outputs), not probabilities
      // It applies sigmoid internally
      return tf.losses.sigmoidCrossEntropy(targets, predictions);
    }
  }

  /**
   * Compute validation loss
   */
  private computeValidationLoss(
    valData: { observations: any; actions: any },
    policyAgent: PolicyAgent
  ): number {
    return tf.tidy(() => {
      const predictions = policyAgent.policyNetwork.predict(valData.observations);
      const loss = this.computeLoss(predictions, valData.actions, policyAgent);
      const lossValue = loss.dataSync()[0];
      predictions.dispose();
      return lossValue;
    });
  }

  /**
   * Validate dataset matches policy agent
   */
  private validateDataset(
    dataset: DemonstrationDataset,
    policyAgent: PolicyAgent
  ): void {
    if (dataset.metadata.gameCoreInfo) {
      const info = dataset.metadata.gameCoreInfo;
      if (info.observationSize !== policyAgent.observationSize) {
        throw new Error(
          `Observation size mismatch: dataset has ${info.observationSize}, ` +
          `policy agent expects ${policyAgent.observationSize}`
        );
      }
      if (info.actionSize !== policyAgent.actionSize) {
        throw new Error(
          `Action size mismatch: dataset has ${info.actionSize}, ` +
          `policy agent expects ${policyAgent.actionSize}`
        );
      }
    }

    // Validate first step if available
    if (dataset.episodes.length > 0 && dataset.episodes[0].steps.length > 0) {
      const firstStep = dataset.episodes[0].steps[0];
      if (firstStep.observation.length !== policyAgent.observationSize) {
        throw new Error('Observation size mismatch in dataset');
      }
      if (firstStep.action.length !== policyAgent.actionSize) {
        throw new Error('Action size mismatch in dataset');
      }
    }
  }

  /**
   * Shuffle array in place
   */
  private shuffleArray<T>(array: T[]): void {
    for (let i = array.length - 1; i > 0; i--) {
      const j = Math.floor(Math.random() * (i + 1));
      [array[i], array[j]] = [array[j], array[i]];
    }
  }

  /**
   * Yield to event loop with smart strategy based on tab visibility
   */
  private async yieldToEventLoop(): Promise<void> {
    // Check if tab is hidden using Page Visibility API
    const isHidden = typeof document !== 'undefined' && 
                     (document.hidden || document.visibilityState === 'hidden');
    
    if (isHidden) {
      // Tab is hidden: use MessageChannel (not throttled)
      return new Promise(resolve => {
        this.yieldChannelResolve = resolve;
        this.yieldChannel.port2.postMessage(null);
      });
    } else {
      // Tab is visible: use setTimeout(0) to allow UI updates
      return new Promise(resolve => setTimeout(resolve, 0));
    }
  }
}

