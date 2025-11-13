/**
 * Demonstration Types - Game-agnostic demonstration structures
 * 
 * Location: src/MimicRL/bc/Demonstration.ts (library code)
 */

import { Action, ActionSpace } from '../core/GameCore.js';

/**
 * A single demonstration step (state-action pair)
 * Game-agnostic: uses normalized observations and actions
 */
export interface DemonstrationStep {
  /**
   * Normalized observation array (from GameCore)
   */
  observation: number[];

  /**
   * Action taken by expert (number array)
   * Matches Action type from GameCore interface
   */
  action: Action;

  /**
   * Optional metadata
   */
  metadata?: {
    timestamp?: number;
    episodeId?: string;
    stepIndex?: number;
    [key: string]: any;
  };
}

/**
 * A complete episode demonstration
 */
export interface DemonstrationEpisode {
  /**
   * Unique identifier for this episode
   */
  id: string;

  /**
   * Array of demonstration steps
   */
  steps: DemonstrationStep[];

  /**
   * Episode metadata
   */
  metadata: {
    timestamp: number;
    duration?: number;
    outcome?: ('win' | 'loss' | 'tie')[] | null;
    playerIndex?: number;  // Which player this demonstration is for
    [key: string]: any;
  };
}

/**
 * Collection of demonstration episodes
 */
export interface DemonstrationDataset {
  /**
   * Array of demonstration episodes
   */
  episodes: DemonstrationEpisode[];

  /**
   * Dataset metadata
   */
  metadata: {
    totalSteps: number;
    createdAt: number;
    updatedAt: number;
    gameCoreInfo?: {
      observationSize: number;
      actionSize: number;
      actionSpaces: ActionSpace[];
    };
  };
}

