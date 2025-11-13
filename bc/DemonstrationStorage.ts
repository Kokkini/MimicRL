/**
 * DemonstrationStorage - Handles persistence of demonstrations
 * Game-agnostic: stores normalized observations and actions
 * 
 * Location: src/MimicRL/bc/DemonstrationStorage.ts (library code)
 * 
 * Note: In browser environment, uses localStorage/IndexedDB
 * In Node.js environment, uses filesystem
 */

import { ActionSpace } from '../core/GameCore.js';
import { DemonstrationEpisode, DemonstrationDataset } from './Demonstration.js';

export interface DemonstrationStorageConfig {
  /**
   * Base directory/key for storing demonstrations
   * In browser: localStorage key prefix or IndexedDB database name
   * In Node.js: directory path
   */
  storageKey?: string;

  /**
   * Storage format ('json' | 'binary')
   */
  format?: 'json' | 'binary';
}

/**
 * DemonstrationStorage - Handles persistence of demonstrations
 */
export class DemonstrationStorage {
  private storageKey: string;
  private format: 'json' | 'binary';
  private isBrowser: boolean;

  constructor(config?: DemonstrationStorageConfig) {
    this.storageKey = config?.storageKey || 'mimicrl_demonstrations';
    this.format = config?.format || 'json';
    this.isBrowser = typeof window !== 'undefined' && typeof localStorage !== 'undefined';
  }

  /**
   * Save a demonstration dataset
   * @param {DemonstrationDataset} dataset - Dataset to save
   * @param {string} filename - Optional filename/key (default: auto-generated)
   * @returns {Promise<string>} Key/path to saved dataset
   */
  async saveDataset(
    dataset: DemonstrationDataset,
    filename?: string
  ): Promise<string> {
    // Generate filename if not provided
    if (!filename) {
      const timestamp = Date.now();
      filename = `demonstrations_${timestamp}.${this.format === 'json' ? 'json' : 'bin'}`;
    }

    if (this.isBrowser) {
      // Browser: use localStorage
      const key = `${this.storageKey}_${filename}`;
      const json = JSON.stringify(dataset);
      localStorage.setItem(key, json);
      return key;
    } else {
      // Node.js: use filesystem (would need fs module)
      // For now, throw error - can be implemented if needed
      throw new Error('File system storage not implemented. Use browser environment or implement fs module.');
    }
  }

  /**
   * Load a demonstration dataset
   * @param {string} key - Key/path to dataset
   * @returns {Promise<DemonstrationDataset>} Loaded dataset
   */
  async loadDataset(key: string): Promise<DemonstrationDataset> {
    if (this.isBrowser) {
      const json = localStorage.getItem(key);
      if (!json) {
        throw new Error(`Dataset not found: ${key}`);
      }
      return JSON.parse(json);
    } else {
      throw new Error('File system storage not implemented. Use browser environment or implement fs module.');
    }
  }

  /**
   * List all available datasets
   * @returns {Promise<string[]>} Array of dataset keys
   */
  async listDatasets(): Promise<string[]> {
    if (this.isBrowser) {
      const keys: string[] = [];
      for (let i = 0; i < localStorage.length; i++) {
        const key = localStorage.key(i);
        if (key && key.startsWith(this.storageKey + '_')) {
          keys.push(key);
        }
      }
      return keys;
    } else {
      throw new Error('File system storage not implemented. Use browser environment or implement fs module.');
    }
  }

  /**
   * Delete a dataset
   * @param {string} key - Key/path to dataset
   */
  async deleteDataset(key: string): Promise<void> {
    if (this.isBrowser) {
      localStorage.removeItem(key);
    } else {
      throw new Error('File system storage not implemented. Use browser environment or implement fs module.');
    }
  }

  /**
   * Create a dataset from episodes
   * @param {DemonstrationEpisode[]} episodes - Episodes to include
   * @param {Object} gameCoreInfo - GameCore metadata (observationSize, actionSize, actionSpaces)
   * @returns {DemonstrationDataset} Created dataset
   */
  createDataset(
    episodes: DemonstrationEpisode[],
    gameCoreInfo?: {
      observationSize: number;
      actionSize: number;
      actionSpaces: ActionSpace[];
    }
  ): DemonstrationDataset {
    const totalSteps = episodes.reduce((sum, ep) => sum + ep.steps.length, 0);
    
    return {
      episodes,
      metadata: {
        totalSteps,
        createdAt: Date.now(),
        updatedAt: Date.now(),
        gameCoreInfo
      }
    };
  }

  /**
   * Get storage size (for browser, returns approximate size in bytes)
   */
  getStorageSize(): number {
    if (this.isBrowser) {
      let total = 0;
      for (let i = 0; i < localStorage.length; i++) {
        const key = localStorage.key(i);
        if (key && key.startsWith(this.storageKey + '_')) {
          const value = localStorage.getItem(key);
          if (value) {
            total += value.length;
          }
        }
      }
      return total;
    }
    return 0;
  }
}

