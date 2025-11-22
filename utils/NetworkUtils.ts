/**
 * NetworkUtils - Utility functions for network serialization/deserialization
 * Game-agnostic utilities for saving and loading TensorFlow.js models
 */

// TensorFlow.js is loaded from CDN as a global 'tf' object
declare const tf: any;

export interface SerializedNetworkData {
  architecture: {
    inputSize: number;
    hiddenLayers: number[];
    outputSize: number;
    activation: string;
  };
  weights: Array<{
    data: number[];
    shape: number[];
    dtype: string;
  }>;
}

export interface NetworkArchitecture {
  inputSize: number;
  hiddenLayers: number[];
  outputSize: number;
  activation: string;
}

export class NetworkUtils {
  /**
   * Load a tf.LayersModel from serialized weights
   * @param serializedData - Serialized network data
   * @returns Loaded model with weights restored
   */
  static loadNetworkFromSerialized(serializedData: SerializedNetworkData): any {
    const { architecture, weights } = serializedData;
    
    // Create model with same architecture
    const model = tf.sequential();
    
    // Input layer (first hidden layer with input shape)
    model.add(tf.layers.dense({
      units: architecture.hiddenLayers[0],
      inputShape: [architecture.inputSize],
      activation: architecture.activation,
      name: 'input_layer'
    }));
    
    // Additional hidden layers
    for (let i = 1; i < architecture.hiddenLayers.length; i++) {
      model.add(tf.layers.dense({
        units: architecture.hiddenLayers[i],
        activation: architecture.activation,
        name: `hidden_layer_${i}`
      }));
    }
    
    // Output layer (linear activation)
    model.add(tf.layers.dense({
      units: architecture.outputSize,
      activation: 'linear',
      name: 'output_layer'
    }));
    
    // Load weights if provided
    if (weights && weights.length > 0) {
      // Convert serialized weights back to tensors
      const weightTensors = weights.map(w => 
        tf.tensor(w.data, w.shape, w.dtype)
      );
      model.setWeights(weightTensors);
    }
    
    return model;
  }
  
  /**
   * Serialize a tf.LayersModel to a storable format
   * @param model - Model to serialize
   * @param architecture - Architecture config (inputSize, hiddenLayers, outputSize, activation)
   * @returns Serialized model data
   */
  static serializeNetwork(model: any, architecture: NetworkArchitecture): SerializedNetworkData {
    const weights = model.getWeights();
    const serializedWeights = weights.map((w: any) => ({
      data: Array.from(w.dataSync()),
      shape: w.shape,
      dtype: w.dtype
    }));
    
    return {
      architecture,
      weights: serializedWeights
    };
  }
}

