# MimicRL

A game-agnostic reinforcement learning library for training AI agents in any game environment. MimicRL provides a clean, standardized interface that allows you to train policies using Proximal Policy Optimization (PPO) and Behavior Cloning (BC) on any game that implements the `GameCore` interface.

## Features

- **🎮 Game-Agnostic Design**: Works with any game that implements the `GameCore` interface
- **🤖 Reinforcement Learning**: PPO algorithm with support for mixed discrete/continuous actions
- **👥 Behavior Cloning**: Train agents by imitating expert demonstrations
- **🔄 Multiplayer Support**: Train multiple players simultaneously with different controllers
- **📊 Training Infrastructure**: Built-in rollout collection, experience buffers, and training metrics
- **💾 Model Management**: Save/load trained models, checkpoint management
- **🌐 Browser-Based**: Runs entirely in the browser using TensorFlow.js

## Architecture

MimicRL follows a clean separation between game-specific code and the reusable RL library:

```
┌─────────────────────────────────────────────────────────┐
│                    Game Implementation                    │
│  ┌───────────────────────────────────────────────────┐  │
│  │              GameCore (Game-Specific)              │  │
│  │  - Implements standardized GameCore interface     │  │
│  │  - Handles game logic, physics, collisions         │  │
│  │  - Returns normalized observations/rewards         │  │
│  └───────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────┘
                          ▲
                          │
┌─────────────────────────────────────────────────────────┐
│              MimicRL Library (Game-Agnostic)              │
│  ┌───────────────────────────────────────────────────┐  │
│  │         TrainingSession / RLTrainingSystem       │  │
│  │  - Manages training loop                         │  │
│  │  - Collects rollouts                             │  │
│  │  - Updates policies                              │  │
│  └───────────────────────────────────────────────────┘  │
│  ┌───────────────────────────────────────────────────┐  │
│  │         Player Controllers                        │  │
│  │  - PolicyController (RL agent)                   │  │
│  │  - HumanController, RandomController, etc.       │  │
│  └───────────────────────────────────────────────────┘  │
│  ┌───────────────────────────────────────────────────┐  │
│  │         PolicyAgent                               │  │
│  │  - Neural network policy and value functions      │  │
│  │  - Works exclusively with normalized arrays      │  │
│  └───────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────┘
```

## Core Design Principles

### 1. Game-Agnostic RL Components

The RL library works exclusively with **normalized number arrays**:
- **Observations**: `number[]` - normalized feature vectors
- **Actions**: `number[]` - arrays where each element can be discrete (0/1) or continuous
- **No game-specific knowledge**: PolicyAgent, PolicyController, and trainers never see game-specific objects

### 2. GameCore Interface

Any game must implement the `GameCore` interface:

```typescript
interface GameCore {
  reset(): GameState;                    // Reset to initial state
  step(actions: Action[], deltaTime: number): GameState;  // Advance one step
  getNumPlayers(): number;               // Number of players
  getObservationSize(): number;         // Size of observation array
  getActionSize(): number;               // Size of action array
  getActionSpaces(): ActionSpace[];      // Action space for each action index
}
```

The `GameCore` is responsible for:
- Converting internal game state to normalized `number[]` observations
- Applying actions from all players uniformly
- Returning rewards and outcomes for all players

### 3. Player Controllers

Controllers implement the `PlayerController` interface:

```typescript
interface PlayerController {
  decide(observation: number[]): Action;
}
```

Examples:
- `PolicyController` - Uses a trained PolicyAgent (library code)
- `HumanController` - Reads from keyboard/mouse (game-specific)
- `RandomController` - Random actions (game-specific)

## Installation

MimicRL is written in **TypeScript** and designed to work in browser environments with TensorFlow.js.

### Prerequisites

1. **TypeScript**: The library is written in TypeScript. You'll need TypeScript installed:
   ```bash
   npm install --save-dev typescript
   ```

2. **TensorFlow.js**: Include TensorFlow.js in your HTML:
   ```html
   <script src="https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@latest"></script>
   ```

### Setup

1. **Compile TypeScript**: The library source files are in TypeScript. Compile them to JavaScript:
   ```bash
   tsc
   ```
   Or configure your build system to compile TypeScript files.

2. **Import MimicRL components**: 

   **Note on `.js` extensions in TypeScript imports**: When using ES modules with TypeScript, you should use `.js` extensions in import statements even though the source files are `.ts`. This is because:
   - TypeScript doesn't rewrite import paths during compilation
   - The runtime (browser/Node) will resolve these to the compiled `.js` files
   - TypeScript's module resolution will find the `.ts` source files automatically
   
   **TypeScript:**
   ```typescript
   // Use .js extensions even though source files are .ts
   import { PolicyAgent } from './MimicRL/agents/PolicyAgent.js';
   import { TrainingSession } from './MimicRL/training/TrainingSession.js';
   import { PolicyController } from './MimicRL/controllers/PolicyController.js';
   import { GameCore, GameState, Action, ActionSpace } from './MimicRL/core/GameCore.js';
   ```

   **JavaScript (after compilation):**
   ```javascript
   // Same import paths work after TypeScript compilation
   import { PolicyAgent } from './MimicRL/agents/PolicyAgent.js';
   import { TrainingSession } from './MimicRL/training/TrainingSession.js';
   import { PolicyController } from './MimicRL/controllers/PolicyController.js';
   ```

### TypeScript Configuration

If you're using TypeScript in your project, ensure your `tsconfig.json` includes:

```json
{
  "compilerOptions": {
    "target": "ES2020",
    "module": "ES2020",
    "moduleResolution": "node",
    "strict": true,
    "esModuleInterop": true,
    "allowSyntheticDefaultImports": true
  }
}
```

## Quick Start

### Prerequisite: Implement GameCore Interface

Both RL and BC require a game that implements the `GameCore` interface:

```typescript
import { GameCore, GameState, Action, ActionSpace } from './MimicRL/core/GameCore.js';

class MyGameCore implements GameCore {
  reset(): GameState {
    // Initialize game state
    return {
      observations: [
        this.buildObservationFor(0),  // Player 0
        this.buildObservationFor(1)  // Player 1
      ],
      rewards: [0, 0],
      done: false,
      outcome: null
    };
  }

  step(actions: Action[], deltaTime: number): GameState {
    // Apply actions[0] to player 0, actions[1] to player 1, etc.
    // Update game state
    // Return new GameState
  }

  getNumPlayers(): number { return 2; }
  getObservationSize(): number { return 9; }  // Size of normalized observation array
  getActionSize(): number { return 4; }         // Size of action array
  getActionSpaces(): ActionSpace[] {
    return [
      { type: 'discrete' },   // Action 0: button press
      { type: 'discrete' },   // Action 1: button press
      { type: 'continuous' }, // Action 2: continuous value
      { type: 'continuous' }  // Action 3: continuous value
    ];
  }

  private buildObservationFor(playerIndex: number): number[] {
    // Convert game state to normalized number array
    // Example: [normalizedX, normalizedY, normalizedAngle, ...]
    return [/* normalized values */];
  }
}
```

## Quick Start: Reinforcement Learning (PPO)

### Step 1: Create Training Session

```typescript
import { TrainingSession } from './MimicRL/training/TrainingSession.js';
import { PolicyController } from './MimicRL/controllers/PolicyController.js';
import { HumanController } from './game/controllers/HumanController.js';
import { RandomController } from './game/controllers/RandomController.js';

const gameCore = new MyGameCore();

// Create controllers for each player
const controllers = [
  new HumanController(),  // Player 0 - will be replaced with PolicyController during training
  new RandomController(gameCore.getActionSpaces())  // Player 1 uses random
];

// Create training session
const trainingSession = new TrainingSession(gameCore, controllers, {
  trainablePlayers: [0],  // Train player 0
  maxGames: 1000,  // Maximum number of games to train
  autoSaveInterval: 100,  // Auto-save model every N games
  algorithm: {
    type: 'PPO',
    hyperparameters: {
      learningRate: 0.0003,
      clipRatio: 0.2,
      valueLossCoeff: 0.5,
      entropyCoeff: 0.01
    }
  },
  networkArchitecture: {
    policyHiddenLayers: [64, 32],
    valueHiddenLayers: [64, 32],
    activation: 'relu'
  }
});

// Initialize training session (creates policy agents)
await trainingSession.initialize();

// Start training
await trainingSession.start();
```

### Step 2: Use Trained Agent

```typescript
// After training, get the trained agent
const trainedAgent = trainingSession.policyAgents[0];

// Create a controller with the trained agent
const policyController = new PolicyController(trainedAgent);

// Use in gameplay
const observation = gameCore.reset().observations[0];
const action = policyController.decide(observation);
```

## Quick Start: Behavior Cloning

### Step 1: Record Demonstrations

```typescript
import { DemonstrationCollector } from './MimicRL/bc/DemonstrationCollector.js';

const gameCore = new MyGameCore();
const collector = new DemonstrationCollector({
  autoRecord: false  // User must explicitly enable
});

// Start recording an episode
collector.startEpisode('episode_1', {
  playerIndex: 0  // Record player 0's actions
});

// During gameplay, record each step
let state = gameCore.reset();
while (!state.done) {
  // Get action from human/expert (e.g., from HumanController)
  const actions = [humanController.decide(state.observations[0]), /* opponent action */];
  
  // Record the step
  collector.recordStep(
    state.observations[0],  // Player 0's observation
    actions[0]              // Player 0's action
  );
  
  // Advance game
  state = gameCore.step(actions, 0.05);
}

// End episode
const episode = collector.endEpisode({
  outcome: state.outcome
});
```

### Step 2: Save Demonstrations

```typescript
import { DemonstrationStorage } from './MimicRL/bc/DemonstrationStorage.js';

const storage = new DemonstrationStorage({
  storageDir: './demonstrations',
  format: 'json'
});

// Create dataset from episodes
const dataset = storage.createDataset([episode], {
  observationSize: gameCore.getObservationSize(),
  actionSize: gameCore.getActionSize(),
  actionSpaces: gameCore.getActionSpaces()
});

// Save to disk
const filepath = await storage.saveDataset(dataset);
console.log(`Saved demonstration to: ${filepath}`);
```

### Step 3: Train with Behavior Cloning

```typescript
import { BehaviorCloningTrainer } from './MimicRL/bc/BehaviorCloningTrainer.js';
import { PolicyAgent } from './MimicRL/agents/PolicyAgent.js';

// Create policy agent
const policyAgent = new PolicyAgent({
  observationSize: gameCore.getObservationSize(),
  actionSize: gameCore.getActionSize(),
  actionSpaces: gameCore.getActionSpaces(),
  networkArchitecture: {
    policyHiddenLayers: [64, 32],
    valueHiddenLayers: [64, 32],
    activation: 'relu'
  }
});

// Create BC trainer
const bcTrainer = new BehaviorCloningTrainer({
  learningRate: 0.001,
  batchSize: 32,
  epochs: 10,
  lossType: 'mixed',  // Automatically handles discrete/continuous
  validationSplit: 0.2
});

// Load dataset
const dataset = await storage.loadDataset(filepath);

// Train policy network
const stats = await bcTrainer.train(
  dataset,
  policyAgent,
  (epoch, loss, valLoss) => {
    console.log(`Epoch ${epoch}: loss=${loss.toFixed(4)}, valLoss=${valLoss?.toFixed(4)}`);
  }
);

console.log('Training complete!', stats);
```

### Step 4: Use Trained Agent

```typescript
import { PolicyController } from './MimicRL/controllers/PolicyController.js';

// Create controller with trained agent
const policyController = new PolicyController(policyAgent);

// Use in gameplay
const observation = gameCore.reset().observations[0];
const action = policyController.decide(observation);
```

## Core Components

### PolicyAgent

The core RL agent that contains policy and value networks:

```typescript
import { PolicyAgent } from './MimicRL/agents/PolicyAgent.js';

const agent = new PolicyAgent({
  observationSize: 9,
  actionSize: 4,
  actionSpaces: [
    { type: 'discrete' },
    { type: 'discrete' },
    { type: 'continuous' },
    { type: 'continuous' }
  ],
  networkArchitecture: {
    policyHiddenLayers: [64, 32],
    valueHiddenLayers: [64, 32],
    activation: 'relu'
  }
});

// Act on an observation
const result = agent.act(observation);
// Returns: { action: Action, logProb: number, value: number }
```

**Key Properties:**
- `policyNetwork`: TensorFlow.js model for policy (outputs action logits/means)
- `valueNetwork`: TensorFlow.js model for value function (outputs scalar value)
- `learnableStd`: Trainable standard deviation array for continuous actions
- `actionSpaces`: Action space definitions (one per action index)

### TrainingSession

Manages the training loop, rollout collection, and model updates:

```typescript
import { TrainingSession } from './MimicRL/training/TrainingSession.js';

const session = new TrainingSession(gameCore, controllers, {
  trainablePlayers: [0],
  maxGames: 1000,
  algorithm: {
    type: 'PPO',
    hyperparameters: { /* PPO hyperparameters */ }
  }
});

// Callbacks
session.onGameEnd = (winner, gamesCompleted, metrics) => {
  console.log(`Game ${gamesCompleted} completed. Winner: ${winner}`);
};

session.onTrainingProgress = (metrics) => {
  console.log('Training progress:', metrics);
};

// Start training
await session.start();

// Pause/resume
session.pause();
session.resume();

// Stop training
session.stop();
```

### PPOTrainer

Implements Proximal Policy Optimization algorithm:

```typescript
import { PPOTrainer } from './MimicRL/training/PPOTrainer.js';

const trainer = new PPOTrainer({
  learningRate: 0.0003,
  clipRatio: 0.2,
  valueLossCoeff: 0.5,
  entropyCoeff: 0.01,
  gaeLambda: 0.95,
  epochs: 4,
  miniBatchSize: 64
});

// Train on experiences
const stats = await trainer.train(experiences, policyAgent);
// Returns: { policyLoss, valueLoss, entropy, klDivergence, clipFraction }
```

**Features:**
- Supports mixed discrete/continuous actions
- Generalized Advantage Estimation (GAE)
- Clipped surrogate objective
- Entropy bonus for exploration
- Gradient clipping

## Behavior Cloning

MimicRL includes a complete Behavior Cloning system for imitation learning:

### Recording Demonstrations

```typescript
import { DemonstrationCollector } from './MimicRL/bc/DemonstrationCollector.js';

const collector = new DemonstrationCollector({
  autoRecord: false  // User must explicitly enable
});

// Start recording
collector.startEpisode('episode_1', {
  playerIndex: 0
});

// Record steps during gameplay
const state = gameCore.step(actions, deltaTime);
collector.recordStep(
  state.observations[0],  // Player 0's observation
  actions[0]              // Player 0's action
);

// End episode
const episode = collector.endEpisode({
  outcome: state.outcome
});
```

### Saving Demonstrations

```typescript
import { DemonstrationStorage } from './MimicRL/bc/DemonstrationStorage.js';

const storage = new DemonstrationStorage({
  storageDir: './demonstrations',
  format: 'json'
});

// Save episode
const dataset = storage.createDataset([episode], {
  observationSize: gameCore.getObservationSize(),
  actionSize: gameCore.getActionSize(),
  actionSpaces: gameCore.getActionSpaces()
});

const filepath = await storage.saveDataset(dataset);
```

### Training with Behavior Cloning

```typescript
import { BehaviorCloningTrainer } from './MimicRL/bc/BehaviorCloningTrainer.js';

const bcTrainer = new BehaviorCloningTrainer({
  learningRate: 0.001,
  batchSize: 32,
  epochs: 10,
  lossType: 'mixed',  // Automatically handles discrete/continuous
  validationSplit: 0.2
});

// Load dataset
const dataset = await storage.loadDataset(filepath);

// Train policy network
const stats = await bcTrainer.train(
  dataset,
  policyAgent,
  (epoch, loss, valLoss) => {
    console.log(`Epoch ${epoch}: loss=${loss}, valLoss=${valLoss}`);
  }
);
```

### Integration with TrainingSession

```typescript
const trainingSession = new TrainingSession(gameCore, controllers, {
  trainablePlayers: [0],
  enableBehaviorCloning: true,
  demonstrationStorageKey: 'demonstrations',
  behaviorCloningConfig: {
    learningRate: 0.001,
    batchSize: 32,
    epochs: 10,
    lossType: 'mixed'
  }
});

// Save demonstration episode
await trainingSession.saveDemonstrationEpisode(episode, {
  observationSize: gameCore.getObservationSize(),
  actionSize: gameCore.getActionSize(),
  actionSpaces: gameCore.getActionSpaces()
});

// Train with saved demonstrations
const datasetPaths = await storage.listDatasets();
await trainingSession.trainBehaviorCloning(datasetPaths, (epoch, loss, valLoss) => {
  console.log(`BC Training: Epoch ${epoch}, Loss: ${loss}`);
});
```

## Action Spaces

MimicRL supports **mixed action spaces** - you can have both discrete and continuous actions in the same action array:

```typescript
// Example: Game with discrete movement buttons + continuous mouse position
getActionSpaces(): ActionSpace[] {
  return [
    { type: 'discrete' },   // W key (0 or 1)
    { type: 'discrete' },   // A key (0 or 1)
    { type: 'discrete' },   // S key (0 or 1)
    { type: 'discrete' },   // D key (0 or 1)
    { type: 'continuous' }, // Mouse X position (any real number)
    { type: 'continuous' }  // Mouse Y position (any real number)
  ];
}
```

**Discrete Actions:**
- Policy network outputs logits
- Sigmoid applied to get probability
- Sampled as 0 or 1
- Loss: Binary cross-entropy

**Continuous Actions:**
- Policy network outputs mean directly (in original units)
- Standard deviation is a learnable parameter (one per action index)
- Sampling uses reparameterization trick: `action = mean + std * epsilon`
- Loss: Mean squared error

## Model Management

### Saving and Loading Models

```typescript
import { NetworkUtils } from './MimicRL/utils/NetworkUtils.js';
import { ModelManager } from './MimicRL/utils/ModelManager.js';

// Save policy network
const serializedPolicy = NetworkUtils.serializeNetwork(
  policyAgent.policyNetwork,
  {
    inputSize: policyAgent.observationSize,
    hiddenLayers: [64, 32],
    outputSize: policyAgent.actionSize,
    activation: 'relu'
  }
);

// Save value network
const serializedValue = NetworkUtils.serializeNetwork(
  policyAgent.valueNetwork,
  {
    inputSize: policyAgent.observationSize,
    hiddenLayers: [64, 32],
    outputSize: 1,
    activation: 'relu'
  }
);

// Save to file or localStorage
localStorage.setItem('policy_weights', JSON.stringify(serializedPolicy));
localStorage.setItem('value_weights', JSON.stringify(serializedValue));

// Load later
const loadedPolicy = NetworkUtils.loadNetworkFromSerialized(serializedPolicy);
const loadedValue = NetworkUtils.loadNetworkFromSerialized(serializedValue);

// Create agent with loaded networks
const agent = new PolicyAgent({
  observationSize: 9,
  actionSize: 4,
  actionSpaces: [...],
  policyNetwork: loadedPolicy,
  valueNetwork: loadedValue
});
```

### Using ModelManager

```typescript
import { ModelManager } from './MimicRL/utils/ModelManager.js';

const modelManager = new ModelManager({
  storageKey: 'my_models',
  useIndexedDB: true  // Use IndexedDB for larger models
});

// Save agent
await modelManager.saveAgent('agent_v1', policyAgent);

// Load agent
const loadedAgent = await modelManager.loadAgent('agent_v1', {
  observationSize: 9,
  actionSize: 4,
  actionSpaces: [...]
});

// List saved agents
const agents = await modelManager.listAgents();
```

## Advanced Usage

### Multi-Player Training

Train multiple players simultaneously:

```typescript
const controllers = [
  null,  // Player 0 - will be PolicyController
  null,  // Player 1 - will be PolicyController
  new RandomController(actionSpaces)  // Player 2 - random
];

const session = new TrainingSession(gameCore, controllers, {
  trainablePlayers: [0, 1],  // Train both player 0 and 1
  // ... other config
});
```

### Custom Controllers

Implement your own controller:

```typescript
import { PlayerController } from './MimicRL/controllers/PlayerController.js';
import { Action } from './MimicRL/core/GameCore.js';

class ScriptedController implements PlayerController {
  decide(observation: number[]): Action {
    // Your custom logic
    // Example: if observation[0] > 0.5, move right
    return [
      observation[0] > 0.5 ? 1 : 0,  // Action 0
      observation[1] < 0.3 ? 1 : 0,  // Action 1
      // ... other actions
    ];
  }
}
```

### Parallel Rollout Collection

TrainingSession automatically uses parallel rollout collection for efficiency:

```typescript
const session = new TrainingSession(gameCore, controllers, {
  trainablePlayers: [0],
  numRollouts: 4,  // Collect 4 rollouts in parallel
  // ... other config
});
```

## Training Visualization

MimicRL includes a lightweight, game-agnostic visualization component for debugging training progress:

### TrainingVisualizer

A simple chart-based component that displays core RL metrics without any game-specific dependencies:

```javascript
import { TrainingVisualizer } from './MimicRL/visualization/TrainingVisualizer.js';

// Create visualizer (container must exist in HTML)
const visualizer = new TrainingVisualizer('training-viz-container', {
  maxDataPoints: 100,
  episodeLengthUnit: 'seconds',  // or 'steps'
  actionIntervalSeconds: 0.2
});

// Attach to training session for automatic updates
visualizer.attachToSession(trainingSession);

// Or update manually
visualizer.updateMetrics({
  gamesCompleted: 100,
  rewardStats: { avg: 0.5, min: -1.0, max: 2.0 },
  averageGameLength: 80,
  winRate: 0.75,  // or completionRate
  policyEntropy: 0.9,
  policyLoss: 0.01,
  valueLoss: 0.5,
  trainingTime: 3600000  // milliseconds
});
```

**Features:**
- **Reward Progress Chart**: Average, min, and max rewards over rollouts
- **Episode Length Chart**: Average episode length over time
- **Completion Rate Chart**: Success/completion percentage
- **Policy Metrics Chart**: Entropy, policy loss, and value loss
- **Summary Stats**: Real-time text display of key metrics

**HTML Setup:**
```html
<div id="training-viz-container"></div>
<script src="https://cdn.jsdelivr.net/npm/chart.js@latest"></script>
```

**Options:**
- `maxDataPoints`: Maximum data points to keep in charts (default: 100)
- `showCompletionRate`: Show completion rate chart (default: true)
- `episodeLengthUnit`: 'steps' or 'seconds' (default: 'steps')
- `actionIntervalSeconds`: Seconds per step for conversion (default: 0.2)

**Methods:**
- `attachToSession(trainingSession)`: Automatically update from session callbacks
- `updateMetrics(metrics)`: Manually update with metrics object
- `reset()`: Clear all data
- `show()` / `hide()`: Toggle visibility
- `dispose()`: Clean up resources

**Note:** Requires Chart.js to be loaded globally. The component gracefully handles Chart.js not being available.

## API Reference

### Core Interfaces

#### GameCore

```typescript
interface GameCore {
  reset(): GameState;
  step(actions: Action[], deltaTime: number): GameState;
  getNumPlayers(): number;
  getObservationSize(): number;
  getActionSize(): number;
  getActionSpaces(): ActionSpace[];
  getOutcome?(): ('win' | 'loss' | 'tie')[] | null;
}
```

#### GameState

```typescript
interface GameState {
  observations: number[][];  // One per player
  rewards: number[];          // One per player
  done: boolean;
  outcome: ('win' | 'loss' | 'tie')[] | null;
  info?: {
    stepCount?: number;
    elapsedTime?: number;
    [key: string]: any;
  };
}
```

#### PlayerController

```typescript
interface PlayerController {
  decide(observation: number[]): Action;
}
```

### Training Components

- **TrainingSession**: Main training orchestrator
- **PPOTrainer**: PPO algorithm implementation
- **RolloutCollector**: Collects experiences from game episodes
- **PolicyAgent**: Neural network agent with policy and value networks

### Behavior Cloning Components

- **DemonstrationCollector**: Records expert demonstrations
- **DemonstrationStorage**: Persists demonstrations to disk
- **BehaviorCloningTrainer**: Supervised learning trainer

### Utilities

- **NetworkUtils**: Serialize/deserialize TensorFlow.js models
- **ModelManager**: High-level model save/load management
- **PolicyManager**: Policy versioning and management

## Examples

See the main project (`src/game/SaberGameCore.ts`) for a complete example of implementing the `GameCore` interface for a 2D arena combat game.

## Design Documents

For detailed design information, see:
- `DESIGN-GameAgnosticRL.md` - Core architecture and design principles
- `DESIGN-BehaviorCloning.md` - Behavior Cloning system design

## License

MIT

## Contributing

Contributions are welcome! Please ensure your code follows the game-agnostic design principles:
- RL library code should work exclusively with normalized `number[]` arrays
- Game-specific code should implement the `GameCore` interface
- Controllers should implement the `PlayerController` interface

