/**
 * DemonstrationCollector - Collects expert demonstrations during gameplay
 * Game-agnostic: works with normalized observations and actions from GameCore
 * 
 * Location: src/MimicRL/bc/DemonstrationCollector.ts (library code)
 */

import { Action } from '../core/GameCore.js';
import { DemonstrationStep, DemonstrationEpisode } from './Demonstration.js';

export interface DemonstrationCollectorConfig {
  /**
   * Whether to automatically record all steps (default: false)
   * If false, only records when explicitly enabled
   */
  autoRecord?: boolean;

  /**
   * Maximum number of steps to keep in memory before flushing
   */
  maxBufferSize?: number;
}

/**
 * DemonstrationCollector - Collects expert demonstrations during gameplay
 */
export class DemonstrationCollector {
  private autoRecord: boolean;
  private maxBufferSize: number;
  private currentEpisode: DemonstrationEpisode | null;
  private episodeBuffer: DemonstrationEpisode[];
  private isRecording: boolean;

  constructor(config?: DemonstrationCollectorConfig) {
    this.autoRecord = config?.autoRecord ?? false;
    this.maxBufferSize = config?.maxBufferSize ?? 10000;
    this.currentEpisode = null;
    this.episodeBuffer = [];
    this.isRecording = false;
  }

  /**
   * Start recording a new episode
   * @param {string} episodeId - Unique identifier for this episode
   * @param {Object} metadata - Optional episode metadata
   */
  startEpisode(episodeId: string, metadata?: Record<string, any>): void {
    if (this.isRecording) {
      this.endEpisode();
    }
    
    this.currentEpisode = {
      id: episodeId,
      steps: [],
      metadata: {
        timestamp: Date.now(),
        ...metadata
      }
    };
    this.isRecording = true;
  }

  /**
   * Record a single step (observation-action pair)
   * @param {number[]} observation - Normalized observation from GameCore
   * @param {Action} action - Action taken by expert
   * @param {Object} stepMetadata - Optional step metadata
   */
  recordStep(
    observation: number[],
    action: Action,
    stepMetadata?: Record<string, any>
  ): void {
    if (!this.isRecording || !this.currentEpisode) {
      return; // Not recording, ignore
    }

    const step: DemonstrationStep = {
      observation,
      action,
      metadata: {
        stepIndex: this.currentEpisode.steps.length,
        ...stepMetadata
      }
    };

    this.currentEpisode.steps.push(step);

    // Flush to buffer if needed
    if (this.currentEpisode.steps.length >= this.maxBufferSize) {
      this.flushCurrentEpisode();
    }
  }

  /**
   * End the current episode and return the demonstration
   * @param {Object} episodeMetadata - Final episode metadata (e.g., outcome)
   * @returns {DemonstrationEpisode | null} The completed episode, or null if not recording
   */
  endEpisode(episodeMetadata?: Record<string, any>): DemonstrationEpisode | null {
    if (!this.isRecording || !this.currentEpisode) {
      return null;
    }

    // Merge final metadata
    const startTime = this.currentEpisode.metadata.timestamp;
    this.currentEpisode.metadata = {
      ...this.currentEpisode.metadata,
      ...episodeMetadata,
      duration: Date.now() - startTime
    };

    const episode = this.currentEpisode;
    this.episodeBuffer.push(episode);
    this.currentEpisode = null;
    this.isRecording = false;

    return episode;
  }

  /**
   * Discard the current episode without saving
   */
  discardEpisode(): void {
    this.currentEpisode = null;
    this.isRecording = false;
  }

  /**
   * Get all buffered episodes
   * @returns {DemonstrationEpisode[]} Array of completed episodes
   */
  getEpisodes(): DemonstrationEpisode[] {
    return [...this.episodeBuffer];
  }

  /**
   * Clear all buffered episodes
   */
  clearEpisodes(): void {
    this.episodeBuffer = [];
  }

  /**
   * Check if currently recording
   */
  getRecording(): boolean {
    return this.isRecording;
  }

  /**
   * Flush current episode to buffer (for memory management)
   */
  private flushCurrentEpisode(): void {
    if (this.currentEpisode && this.currentEpisode.steps.length > 0) {
      this.episodeBuffer.push(this.currentEpisode);
      // Start new episode with same ID
      const id = this.currentEpisode.id;
      const metadata = this.currentEpisode.metadata;
      this.currentEpisode = {
        id: id,
        steps: [],
        metadata: { ...metadata }
      };
    }
  }
}

