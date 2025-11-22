/**
 * ModelManager - Handles saving and loading of neural network models
 * Uses localStorage for model persistence and IndexedDB for large data
 */

interface ModelManagerConfig {
  storagePrefix?: string;
  maxModels?: number;
  autoSaveInterval?: number;
  compressionEnabled?: boolean;
}

interface ModelData {
  id: string;
  model: any;
  metadata: {
    savedAt: string;
    version: string;
    [key: string]: any;
  };
}

interface ModelInfo {
  id: string;
  metadata: {
    savedAt: string;
    [key: string]: any;
  };
  size: number;
}

interface StorageStats {
  totalSize: number;
  modelCount: number;
  localStorageUsage: number;
  maxStorage: number;
  usagePercentage: number;
}

interface SerializableModel {
  id?: string;
  serialize(): any;
}

export class ModelManager {
  private storagePrefix: string;
  private maxModels: number;
  private autoSaveInterval: number;
  private compressionEnabled: boolean;
  private db: IDBDatabase | null = null;

  constructor(config: ModelManagerConfig = {}) {
    this.storagePrefix = config.storagePrefix || 'saber_rl_';
    this.maxModels = config.maxModels || 10;
    this.autoSaveInterval = config.autoSaveInterval || 50; // Auto-save every N games
    this.compressionEnabled = config.compressionEnabled || false;
    
    this.initializeStorage();
  }

  /**
   * Initialize storage systems
   */
  async initializeStorage(): Promise<void> {
    try {
      // Check localStorage availability
      if (typeof Storage === 'undefined') {
        throw new Error('localStorage not available');
      }
      
      // Initialize IndexedDB for large data
      await this.initializeIndexedDB();
      
      console.log('ModelManager initialized successfully');
    } catch (error) {
      console.error('Failed to initialize ModelManager:', error);
      throw error;
    }
  }

  /**
   * Initialize IndexedDB for large data storage
   */
  async initializeIndexedDB(): Promise<void> {
    return new Promise((resolve, reject) => {
      const request = indexedDB.open('SabeRL_RL_Data', 1);
      
      request.onerror = () => reject(request.error);
      request.onsuccess = () => {
        this.db = request.result;
        resolve();
      };
      
      request.onupgradeneeded = (event) => {
        const db = (event.target as IDBOpenDBRequest).result;
        
        // Create object stores
        if (!db.objectStoreNames.contains('models')) {
          db.createObjectStore('models', { keyPath: 'id' });
        }
        
        if (!db.objectStoreNames.contains('experiences')) {
          db.createObjectStore('experiences', { keyPath: 'id', autoIncrement: true });
        }
        
        if (!db.objectStoreNames.contains('metrics')) {
          db.createObjectStore('metrics', { keyPath: 'id' });
        }
      };
    });
  }

  /**
   * Save a neural network model
   * Only keeps the current/latest model - overwrites previous saves
   * @param model - Neural network model
   * @param metadata - Additional metadata
   * @returns Model ID
   */
  async saveModel(model: SerializableModel, metadata: Record<string, any> = {}): Promise<string> {
    try {
      // Use fixed key for current model - overwrites previous save
      const FIXED_MODEL_KEY = `${this.storagePrefix}current_model`;
      
      // Delete old model data before saving new one (free up space)
      const oldModelData = localStorage.getItem(FIXED_MODEL_KEY);
      if (oldModelData) {
        // Try to parse old model to get its ID for cleanup
        try {
          const oldData = JSON.parse(oldModelData);
          if (oldData.id) {
            // Remove from IndexedDB if it exists
            await this.deleteFromIndexedDB('models', oldData.id);
          }
        } catch (e) {
          // Ignore parse errors for old data
        }
      }
      
      // Delete all old model entries from localStorage
      this.cleanupOldModels();
      
      const modelId = model.id || this.generateModelId();
      const serializedModel = model.serialize();
      
      const modelData: ModelData = {
        id: modelId,
        model: serializedModel,
        metadata: {
          ...metadata,
          savedAt: new Date().toISOString(),
          version: '1.0.0'
        }
      };
      
      // Save to localStorage with fixed key (overwrites previous)
      localStorage.setItem(FIXED_MODEL_KEY, JSON.stringify(modelData));
      
      // Save to IndexedDB for backup (also overwrites with fixed ID)
      const indexedDBData = {
        ...modelData,
        id: 'current_model' // Fixed ID for IndexedDB too (overwrite the id from modelData)
      };
      await this.saveToIndexedDB('models', indexedDBData);
      
      console.log(`Model saved successfully: ${modelId}`);
      return modelId;
    } catch (error) {
      console.error('Failed to save model:', error);
      throw error;
    }
  }

  /**
   * Clean up old model entries from localStorage
   * Removes all model entries except the current one
   */
  cleanupOldModels(): void {
    try {
      const FIXED_MODEL_KEY = `${this.storagePrefix}current_model`;
      const keysToRemove: string[] = [];
      
      for (let i = 0; i < localStorage.length; i++) {
        const key = localStorage.key(i);
        if (key && key.startsWith(this.storagePrefix) && key.includes('model_') && key !== FIXED_MODEL_KEY) {
          keysToRemove.push(key);
        }
      }
      
      keysToRemove.forEach(key => {
        localStorage.removeItem(key);
        console.log(`Removed old model: ${key}`);
      });
      
      // Also clear the model list
      const modelListKey = `${this.storagePrefix}model_list`;
      localStorage.removeItem(modelListKey);
      
    } catch (error) {
      console.error('Failed to cleanup old models:', error);
    }
  }

  /**
   * Load a neural network model
   * @param modelId - Model ID (optional, defaults to current model)
   * @returns Loaded model
   */
  async loadModel(modelId: string | null = null): Promise<any> {
    try {
      // If no modelId provided, load the current model
      const storageKey = modelId 
        ? `${this.storagePrefix}model_${modelId}`
        : `${this.storagePrefix}current_model`;
      
      const storedData = localStorage.getItem(storageKey);
      
      if (storedData) {
        const modelData = JSON.parse(storedData) as ModelData;
        return this.deserializeModel(modelData.model);
      }
      
      // Fallback to IndexedDB (try current_model first, then provided ID)
      const dbId = modelId || 'current_model';
      const modelData = await this.loadFromIndexedDB('models', dbId) as ModelData | null;
      if (modelData) {
        return this.deserializeModel(modelData.model);
      }
      
      throw new Error(`Model not found: ${modelId || 'current_model'}`);
    } catch (error) {
      console.error('Failed to load model:', error);
      throw error;
    }
  }

  /**
   * List all saved models
   * @returns List of model information
   */
  async listModels(): Promise<ModelInfo[]> {
    try {
      const models: ModelInfo[] = [];
      const modelListKey = `${this.storagePrefix}model_list`;
      const modelList = localStorage.getItem(modelListKey);
      
      if (modelList) {
        const list = JSON.parse(modelList) as string[];
        for (const modelId of list) {
          const storageKey = `${this.storagePrefix}model_${modelId}`;
          const storedData = localStorage.getItem(storageKey);
          
          if (storedData) {
            const modelData = JSON.parse(storedData) as ModelData;
            models.push({
              id: modelId,
              metadata: modelData.metadata,
              size: storedData.length
            });
          }
        }
      }
      
      return models.sort((a, b) => new Date(b.metadata.savedAt).getTime() - new Date(a.metadata.savedAt).getTime());
    } catch (error) {
      console.error('Failed to list models:', error);
      return [];
    }
  }

  /**
   * Delete a model
   * @param modelId - Model ID
   * @returns Success status
   */
  async deleteModel(modelId: string): Promise<boolean> {
    try {
      // Remove from localStorage
      const storageKey = `${this.storagePrefix}model_${modelId}`;
      localStorage.removeItem(storageKey);
      
      // Remove from IndexedDB
      await this.deleteFromIndexedDB('models', modelId);
      
      // Update model list
      await this.removeFromModelList(modelId);
      
      console.log(`Model deleted: ${modelId}`);
      return true;
    } catch (error) {
      console.error('Failed to delete model:', error);
      return false;
    }
  }

  /**
   * Save training session data
   * @param sessionData - Training session data
   * @returns Session ID
   */
  async saveTrainingSession(sessionData: Record<string, any>): Promise<string> {
    try {
      const sessionId = sessionData.id || this.generateSessionId();
      const data = {
        ...sessionData,
        id: sessionId,
        savedAt: new Date().toISOString()
      };
      
      // Save to IndexedDB
      await this.saveToIndexedDB('sessions', data);
      
      console.log(`Training session saved: ${sessionId}`);
      return sessionId;
    } catch (error) {
      console.error('Failed to save training session:', error);
      throw error;
    }
  }

  /**
   * Load training session data
   * @param sessionId - Session ID
   * @returns Session data
   */
  async loadTrainingSession(sessionId: string): Promise<Record<string, any>> {
    try {
      const sessionData = await this.loadFromIndexedDB('sessions', sessionId);
      if (!sessionData) {
        throw new Error(`Session not found: ${sessionId}`);
      }
      
      return sessionData as Record<string, any>;
    } catch (error) {
      console.error('Failed to load training session:', error);
      throw error;
    }
  }

  /**
   * Save experience data
   * @param experiences - Experience data
   * @returns Experience ID
   */
  async saveExperiences(experiences: any[]): Promise<string> {
    try {
      const experienceId = this.generateExperienceId();
      const data = {
        id: experienceId,
        experiences: experiences,
        savedAt: new Date().toISOString(),
        count: experiences.length
      };
      
      // Save to IndexedDB
      await this.saveToIndexedDB('experiences', data);
      
      console.log(`Experiences saved: ${experienceId} (${experiences.length} items)`);
      return experienceId;
    } catch (error) {
      console.error('Failed to save experiences:', error);
      throw error;
    }
  }

  /**
   * Load experience data
   * @param experienceId - Experience ID
   * @returns Experience data
   */
  async loadExperiences(experienceId: string): Promise<any[]> {
    try {
      const data = await this.loadFromIndexedDB('experiences', experienceId) as { experiences: any[] } | null;
      if (!data) {
        throw new Error(`Experiences not found: ${experienceId}`);
      }
      
      return data.experiences;
    } catch (error) {
      console.error('Failed to load experiences:', error);
      throw error;
    }
  }

  /**
   * Get storage usage statistics
   * @returns Storage statistics
   */
  getStorageStats(): StorageStats {
    try {
      let totalSize = 0;
      let modelCount = 0;
      
      // Calculate localStorage usage
      for (let i = 0; i < localStorage.length; i++) {
        const key = localStorage.key(i);
        if (key && key.startsWith(this.storagePrefix)) {
          const value = localStorage.getItem(key);
          if (value) {
            totalSize += value.length;
            if (key.includes('model_')) {
              modelCount++;
            }
          }
        }
      }
      
      return {
        totalSize: totalSize,
        modelCount: modelCount,
        localStorageUsage: totalSize,
        maxStorage: 5 * 1024 * 1024, // 5MB typical limit
        usagePercentage: (totalSize / (5 * 1024 * 1024)) * 100
      };
    } catch (error) {
      console.error('Failed to get storage stats:', error);
      return { totalSize: 0, modelCount: 0, localStorageUsage: 0, maxStorage: 0, usagePercentage: 0 };
    }
  }

  /**
   * Clear all stored data
   * @returns Success status
   */
  async clearAllData(): Promise<boolean> {
    try {
      // Clear localStorage
      const keysToRemove: string[] = [];
      for (let i = 0; i < localStorage.length; i++) {
        const key = localStorage.key(i);
        if (key && key.startsWith(this.storagePrefix)) {
          keysToRemove.push(key);
        }
      }
      
      keysToRemove.forEach(key => localStorage.removeItem(key));
      
      // Clear IndexedDB
      if (this.db) {
        const transaction = this.db.transaction(['models', 'experiences', 'metrics', 'sessions'], 'readwrite');
        const stores = ['models', 'experiences', 'metrics', 'sessions'];
        
        for (const storeName of stores) {
          const store = transaction.objectStore(storeName);
          await new Promise<void>((resolve, reject) => {
            const request = store.clear();
            request.onsuccess = () => resolve();
            request.onerror = () => reject(request.error);
          });
        }
      }
      
      console.log('All data cleared successfully');
      return true;
    } catch (error) {
      console.error('Failed to clear data:', error);
      return false;
    }
  }

  /**
   * Deserialize model from stored data
   * @param modelData - Serialized model data
   * @returns Deserialized model
   */
  deserializeModel(modelData: any): any {
    // This would depend on the specific model implementation
    // For now, return the data as-is
    return modelData;
  }

  /**
   * Save data to IndexedDB
   * @param storeName - Store name
   * @param data - Data to save
   */
  async saveToIndexedDB(storeName: string, data: any): Promise<void> {
    if (!this.db) return;
    
    return new Promise<void>((resolve, reject) => {
      const transaction = this.db!.transaction([storeName], 'readwrite');
      const store = transaction.objectStore(storeName);
      const request = store.put(data);
      
      request.onsuccess = () => resolve();
      request.onerror = () => reject(request.error);
    });
  }

  /**
   * Load data from IndexedDB
   * @param storeName - Store name
   * @param key - Data key
   * @returns Loaded data
   */
  async loadFromIndexedDB(storeName: string, key: string): Promise<any> {
    if (!this.db) return null;
    
    return new Promise((resolve, reject) => {
      const transaction = this.db!.transaction([storeName], 'readonly');
      const store = transaction.objectStore(storeName);
      const request = store.get(key);
      
      request.onsuccess = () => resolve(request.result);
      request.onerror = () => reject(request.error);
    });
  }

  /**
   * Delete data from IndexedDB
   * @param storeName - Store name
   * @param key - Data key
   */
  async deleteFromIndexedDB(storeName: string, key: string): Promise<void> {
    if (!this.db) return;
    
    return new Promise<void>((resolve, reject) => {
      const transaction = this.db!.transaction([storeName], 'readwrite');
      const store = transaction.objectStore(storeName);
      const request = store.delete(key);
      
      request.onsuccess = () => resolve();
      request.onerror = () => reject(request.error);
    });
  }

  /**
   * Update model list
   * @param modelId - Model ID
   * @param _modelData - Model data (unused, kept for API compatibility)
   */
  async updateModelList(modelId: string, _modelData: any): Promise<void> {
    const modelListKey = `${this.storagePrefix}model_list`;
    let modelList: string[] = [];
    
    const stored = localStorage.getItem(modelListKey);
    if (stored) {
      modelList = JSON.parse(stored);
    }
    
    if (!modelList.includes(modelId)) {
      modelList.push(modelId);
      
      // Keep only the most recent models
      if (modelList.length > this.maxModels) {
        modelList = modelList.slice(-this.maxModels);
      }
      
      localStorage.setItem(modelListKey, JSON.stringify(modelList));
    }
  }

  /**
   * Remove from model list
   * @param modelId - Model ID
   */
  async removeFromModelList(modelId: string): Promise<void> {
    const modelListKey = `${this.storagePrefix}model_list`;
    const stored = localStorage.getItem(modelListKey);
    
    if (stored) {
      const modelList = JSON.parse(stored) as string[];
      const filteredList = modelList.filter(id => id !== modelId);
      localStorage.setItem(modelListKey, JSON.stringify(filteredList));
    }
  }

  /**
   * Generate unique model ID
   * @returns Model ID
   */
  generateModelId(): string {
    return `model_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
  }

  /**
   * Generate unique session ID
   * @returns Session ID
   */
  generateSessionId(): string {
    return `session_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
  }

  /**
   * Generate unique experience ID
   * @returns Experience ID
   */
  generateExperienceId(): string {
    return `exp_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
  }
}

