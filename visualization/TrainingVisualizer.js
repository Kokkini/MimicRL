/**
 * TrainingVisualizer - Lightweight, game-agnostic RL training visualization
 * Provides charts and metrics for debugging training progress
 * 
 * Usage:
 *   const visualizer = new TrainingVisualizer('container-id');
 *   visualizer.attachToSession(trainingSession);
 * 
 * Or manual updates:
 *   visualizer.updateMetrics(metrics);
 */

export class TrainingVisualizer {
  constructor(containerId, options = {}) {
    // Get container element
    if (typeof containerId === 'string') {
      this.container = document.getElementById(containerId);
    } else {
      this.container = containerId; // Assume it's already an element
    }
    
    if (!this.container) {
      throw new Error(`Container element not found: ${containerId}`);
    }

    // Options
    this.options = {
      maxDataPoints: options.maxDataPoints || 100,
      showCompletionRate: options.showCompletionRate !== false, // Default true
      episodeLengthUnit: options.episodeLengthUnit || 'steps', // 'steps' or 'seconds'
      actionIntervalSeconds: options.actionIntervalSeconds || 0.2, // For step->second conversion
      ...options
    };

    // Chart instances
    this.rewardChart = null;
    this.lengthChart = null;
    this.completionChart = null;
    this.policyChart = null;

    // Chart data tracking
    this.batchNumber = 0;
    this.isInitialized = false;

    // Training session reference
    this.trainingSession = null;

    // Initialize UI
    this.createUI();
  }

  /**
   * Create the UI structure
   */
  createUI() {
    this.container.innerHTML = `
      <style>
        .training-visualizer {
          font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
          background-color: #1a1a1a;
          color: #e0e0e0;
          padding: 16px;
          border-radius: 8px;
        }
        .stats-summary {
          display: grid;
          grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
          gap: 12px;
          margin-bottom: 20px;
          padding: 12px;
          background-color: #2a2a2a;
          border-radius: 4px;
        }
        .stat-item {
          display: flex;
          flex-direction: column;
        }
        .stat-label {
          font-size: 12px;
          color: #aaa;
          margin-bottom: 4px;
        }
        .stat-value {
          font-size: 18px;
          font-weight: bold;
          color: #4a9eff;
        }
        .chart-container {
          margin-bottom: 20px;
          background-color: #2a2a2a;
          padding: 12px;
          border-radius: 4px;
        }
        .chart-container h4 {
          margin: 0 0 12px 0;
          font-size: 14px;
          color: #4a9eff;
        }
        .chart-container canvas {
          max-height: 300px;
        }
      </style>
      <div class="training-visualizer">
        <div class="stats-summary" id="stats-summary">
          <div class="stat-item">
            <span class="stat-label">Games Completed</span>
            <span class="stat-value" id="stat-games">0</span>
          </div>
          <div class="stat-item">
            <span class="stat-label">Avg Reward</span>
            <span class="stat-value" id="stat-avg-reward">0.00</span>
          </div>
          <div class="stat-item">
            <span class="stat-label">Best Reward</span>
            <span class="stat-value" id="stat-best-reward">0.00</span>
          </div>
          <div class="stat-item">
            <span class="stat-label">Completion Rate</span>
            <span class="stat-value" id="stat-completion">0.0%</span>
          </div>
          <div class="stat-item">
            <span class="stat-label">Training Time</span>
            <span class="stat-value" id="stat-time">0s</span>
          </div>
        </div>
        
        <div class="chart-container" id="reward-chart-container" style="display: none;">
          <h4>Reward Progress</h4>
          <canvas id="reward-chart-canvas"></canvas>
        </div>
        
        <div class="chart-container" id="length-chart-container" style="display: none;">
          <h4>Episode Length</h4>
          <canvas id="length-chart-canvas"></canvas>
        </div>
        
        <div class="chart-container" id="completion-chart-container" style="display: none;">
          <h4>Completion Rate</h4>
          <canvas id="completion-chart-canvas"></canvas>
        </div>
        
        <div class="chart-container" id="policy-chart-container" style="display: none;">
          <h4>Policy Metrics</h4>
          <canvas id="policy-chart-canvas"></canvas>
        </div>
      </div>
    `;

    // Get references to elements
    this.statsSummary = document.getElementById('stats-summary');
    this.statGames = document.getElementById('stat-games');
    this.statAvgReward = document.getElementById('stat-avg-reward');
    this.statBestReward = document.getElementById('stat-best-reward');
    this.statCompletion = document.getElementById('stat-completion');
    this.statTime = document.getElementById('stat-time');
    
    this.rewardChartContainer = document.getElementById('reward-chart-container');
    this.rewardChartCanvas = document.getElementById('reward-chart-canvas');
    this.lengthChartContainer = document.getElementById('length-chart-container');
    this.lengthChartCanvas = document.getElementById('length-chart-canvas');
    this.completionChartContainer = document.getElementById('completion-chart-container');
    this.completionChartCanvas = document.getElementById('completion-chart-canvas');
    this.policyChartContainer = document.getElementById('policy-chart-container');
    this.policyChartCanvas = document.getElementById('policy-chart-canvas');
  }

  /**
   * Attach to a TrainingSession for automatic updates
   * @param {TrainingSession} trainingSession - Training session instance
   */
  attachToSession(trainingSession) {
    if (!trainingSession) {
      console.warn('[TrainingVisualizer] No training session provided');
      return;
    }

    this.trainingSession = trainingSession;

    // Set up callback
    trainingSession.onTrainingProgress = (metrics) => {
      // Skip status-only updates
      if (metrics && metrics._statusOnly) {
        return;
      }
      
      this.updateMetrics(metrics);
    };

    console.log('[TrainingVisualizer] Attached to training session');
  }

  /**
   * Update metrics display
   * @param {Object} metrics - Training metrics object
   */
  updateMetrics(metrics) {
    if (!metrics) {
      return;
    }

    // Update summary stats
    this.updateSummaryStats(metrics);

    // Update charts (only if we have valid data)
    if (metrics.gamesCompleted > 0) {
      this.batchNumber++;
      this.updateCharts(metrics);
    }
  }

  /**
   * Update summary statistics display
   * @param {Object} metrics - Training metrics
   */
  updateSummaryStats(metrics) {
    if (this.statGames) {
      this.statGames.textContent = metrics.gamesCompleted || 0;
    }

    if (this.statAvgReward && metrics.rewardStats) {
      this.statAvgReward.textContent = (metrics.rewardStats.avg || 0).toFixed(2);
    }

    if (this.statBestReward && metrics.rewardStats) {
      this.statBestReward.textContent = (metrics.rewardStats.max || 0).toFixed(2);
    }

    if (this.statCompletion) {
      const completionRate = metrics.winRate !== undefined 
        ? metrics.winRate 
        : (metrics.completionRate !== undefined ? metrics.completionRate : 0);
      this.statCompletion.textContent = `${(completionRate * 100).toFixed(1)}%`;
    }

    if (this.statTime && metrics.trainingTime !== undefined) {
      const seconds = Math.floor(metrics.trainingTime / 1000);
      const minutes = Math.floor(seconds / 60);
      const secs = seconds % 60;
      if (minutes > 0) {
        this.statTime.textContent = `${minutes}m ${secs}s`;
      } else {
        this.statTime.textContent = `${secs}s`;
      }
    }
  }

  /**
   * Update all charts with new metrics
   * @param {Object} metrics - Training metrics
   */
  updateCharts(metrics) {
    // Initialize charts if needed
    if (!this.isInitialized) {
      this.initializeCharts();
    }

    // Calculate batch statistics
    const batchStats = this.calculateBatchStats(metrics);

    // Update each chart
    this.updateRewardChart(batchStats);
    this.updateLengthChart(batchStats);
    this.updateCompletionChart(batchStats);
    this.updatePolicyChart(batchStats);
  }

  /**
   * Calculate batch statistics from metrics
   * @param {Object} metrics - Training metrics
   * @returns {Object} Batch statistics
   */
  calculateBatchStats(metrics) {
    const rewardStats = metrics.rewardStats || { avg: 0, min: 0, max: 0 };
    
    // Convert episode length
    let avgLength = metrics.averageGameLength || 0;
    if (this.options.episodeLengthUnit === 'seconds' && avgLength > 0) {
      avgLength = avgLength * this.options.actionIntervalSeconds;
    }

    // Calculate completion rate
    const completionRate = metrics.winRate !== undefined 
      ? metrics.winRate * 100 
      : (metrics.completionRate !== undefined ? metrics.completionRate * 100 : 0);

    return {
      avgReward: rewardStats.avg || 0,
      minReward: rewardStats.min || 0,
      maxReward: rewardStats.max || 0,
      avgLength,
      completionRate,
      policyEntropy: metrics.policyEntropy !== undefined ? metrics.policyEntropy : 0,
      policyLoss: metrics.policyLoss !== undefined ? metrics.policyLoss : 0,
      valueLoss: metrics.valueLoss !== undefined ? metrics.valueLoss : 0
    };
  }

  /**
   * Initialize all charts
   */
  initializeCharts() {
    const ChartConstructor = window.Chart || Chart;
    if (typeof ChartConstructor === 'undefined' || typeof ChartConstructor !== 'function') {
      console.warn('[TrainingVisualizer] Chart.js not available, charts will not be displayed');
      return;
    }

    try {
      this.initializeRewardChart(ChartConstructor);
      this.initializeLengthChart(ChartConstructor);
      this.initializeCompletionChart(ChartConstructor);
      this.initializePolicyChart(ChartConstructor);
      
      this.isInitialized = true;
      console.log('[TrainingVisualizer] Charts initialized');
    } catch (error) {
      console.error('[TrainingVisualizer] Failed to initialize charts:', error);
    }
  }

  /**
   * Initialize reward progress chart
   */
  initializeRewardChart(ChartConstructor) {
    if (!this.rewardChartCanvas) return;

    const ctx = this.rewardChartCanvas.getContext('2d');
    this.rewardChart = new ChartConstructor(ctx, {
      type: 'line',
      data: {
        labels: [],
        datasets: [
          {
            label: 'Avg Reward',
            data: [],
            borderColor: '#4a9eff',
            backgroundColor: 'rgba(74, 158, 255, 0.1)',
            tension: 0.1,
            fill: true
          },
          {
            label: 'Min Reward',
            data: [],
            borderColor: '#ff6b6b',
            backgroundColor: 'rgba(255, 107, 107, 0.1)',
            tension: 0.1,
            fill: false
          },
          {
            label: 'Max Reward',
            data: [],
            borderColor: '#4caf50',
            backgroundColor: 'rgba(76, 175, 80, 0.1)',
            tension: 0.1,
            fill: false
          }
        ]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        layout: {
          padding: { top: 10, bottom: 10, left: 10, right: 10 }
        },
        scales: {
          x: {
            title: {
              display: true,
              text: 'Rollout'
            }
          },
          y: {
            title: {
              display: true,
              text: 'Reward'
            }
          }
        },
        plugins: {
          legend: {
            display: true,
            position: 'top'
          }
        }
      }
    });

    if (this.rewardChartContainer) {
      this.rewardChartContainer.style.display = 'block';
    }
  }

  /**
   * Initialize episode length chart
   */
  initializeLengthChart(ChartConstructor) {
    if (!this.lengthChartCanvas) return;

    const ctx = this.lengthChartCanvas.getContext('2d');
    this.lengthChart = new ChartConstructor(ctx, {
      type: 'line',
      data: {
        labels: [],
        datasets: [
          {
            label: `Avg Episode Length (${this.options.episodeLengthUnit})`,
            data: [],
            borderColor: '#ff9800',
            backgroundColor: 'rgba(255, 152, 0, 0.1)',
            tension: 0.1,
            fill: true
          }
        ]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        layout: {
          padding: { top: 10, bottom: 10, left: 10, right: 10 }
        },
        scales: {
          x: {
            title: {
              display: true,
              text: 'Rollout'
            }
          },
          y: {
            title: {
              display: true,
              text: `Length (${this.options.episodeLengthUnit})`
            }
          }
        },
        plugins: {
          legend: {
            display: true,
            position: 'top'
          }
        }
      }
    });

    if (this.lengthChartContainer) {
      this.lengthChartContainer.style.display = 'block';
    }
  }

  /**
   * Initialize completion rate chart
   */
  initializeCompletionChart(ChartConstructor) {
    if (!this.completionChartCanvas) return;

    const ctx = this.completionChartCanvas.getContext('2d');
    this.completionChart = new ChartConstructor(ctx, {
      type: 'line',
      data: {
        labels: [],
        datasets: [
          {
            label: 'Completion Rate (%)',
            data: [],
            borderColor: '#4caf50',
            backgroundColor: 'rgba(76, 175, 80, 0.1)',
            tension: 0.1,
            fill: true
          }
        ]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        layout: {
          padding: { top: 10, bottom: 10, left: 10, right: 10 }
        },
        scales: {
          x: {
            title: {
              display: true,
              text: 'Rollout'
            }
          },
          y: {
            title: {
              display: true,
              text: 'Rate (%)'
            },
            min: 0,
            max: 100
          }
        },
        plugins: {
          legend: {
            display: true,
            position: 'top'
          }
        }
      }
    });

    if (this.completionChartContainer) {
      this.completionChartContainer.style.display = 'block';
    }
  }

  /**
   * Initialize policy metrics chart
   */
  initializePolicyChart(ChartConstructor) {
    if (!this.policyChartCanvas) return;

    const ctx = this.policyChartCanvas.getContext('2d');
    this.policyChart = new ChartConstructor(ctx, {
      type: 'line',
      data: {
        labels: [],
        datasets: [
          {
            label: 'Policy Entropy',
            data: [],
            borderColor: '#9c27b0',
            backgroundColor: 'rgba(156, 39, 176, 0.1)',
            tension: 0.1,
            fill: false,
            yAxisID: 'y'
          },
          {
            label: 'Policy Loss',
            data: [],
            borderColor: '#2196f3',
            backgroundColor: 'rgba(33, 150, 243, 0.1)',
            tension: 0.1,
            fill: false,
            yAxisID: 'y1'
          },
          {
            label: 'Value Loss',
            data: [],
            borderColor: '#ff9800',
            backgroundColor: 'rgba(255, 152, 0, 0.1)',
            tension: 0.1,
            fill: false,
            yAxisID: 'y2'
          }
        ]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        layout: {
          padding: { top: 10, bottom: 10, left: 10, right: 10 }
        },
        scales: {
          x: {
            title: {
              display: true,
              text: 'Rollout'
            }
          },
          y: {
            type: 'linear',
            position: 'left',
            title: {
              display: true,
              text: 'Entropy'
            }
          },
          y1: {
            type: 'linear',
            position: 'right',
            title: {
              display: true,
              text: 'Policy Loss'
            },
            grid: {
              drawOnChartArea: false
            }
          },
          y2: {
            type: 'linear',
            position: 'right',
            title: {
              display: true,
              text: 'Value Loss'
            },
            grid: {
              drawOnChartArea: false
            }
          }
        },
        plugins: {
          legend: {
            display: true,
            position: 'top'
          }
        }
      }
    });

    if (this.policyChartContainer) {
      this.policyChartContainer.style.display = 'block';
    }
  }

  /**
   * Update reward chart
   */
  updateRewardChart(batchStats) {
    if (!this.rewardChart) return;

    const label = `Rollout ${this.batchNumber}`;
    this.rewardChart.data.labels.push(label);
    this.rewardChart.data.datasets[0].data.push(batchStats.avgReward);
    this.rewardChart.data.datasets[1].data.push(batchStats.minReward);
    this.rewardChart.data.datasets[2].data.push(batchStats.maxReward);

    // Keep only last N points
    if (this.rewardChart.data.labels.length > this.options.maxDataPoints) {
      this.rewardChart.data.labels.shift();
      this.rewardChart.data.datasets.forEach(dataset => {
        dataset.data.shift();
      });
    }

    this.rewardChart.update('none');
  }

  /**
   * Update length chart
   */
  updateLengthChart(batchStats) {
    if (!this.lengthChart) return;

    const label = `Rollout ${this.batchNumber}`;
    this.lengthChart.data.labels.push(label);
    this.lengthChart.data.datasets[0].data.push(batchStats.avgLength);

    if (this.lengthChart.data.labels.length > this.options.maxDataPoints) {
      this.lengthChart.data.labels.shift();
      this.lengthChart.data.datasets[0].data.shift();
    }

    this.lengthChart.update('none');
  }

  /**
   * Update completion chart
   */
  updateCompletionChart(batchStats) {
    if (!this.completionChart) return;

    const label = `Rollout ${this.batchNumber}`;
    this.completionChart.data.labels.push(label);
    this.completionChart.data.datasets[0].data.push(batchStats.completionRate);

    if (this.completionChart.data.labels.length > this.options.maxDataPoints) {
      this.completionChart.data.labels.shift();
      this.completionChart.data.datasets[0].data.shift();
    }

    this.completionChart.update('none');
  }

  /**
   * Update policy chart
   */
  updatePolicyChart(batchStats) {
    if (!this.policyChart) return;

    const label = `Rollout ${this.batchNumber}`;
    this.policyChart.data.labels.push(label);
    this.policyChart.data.datasets[0].data.push(batchStats.policyEntropy);
    this.policyChart.data.datasets[1].data.push(batchStats.policyLoss);
    this.policyChart.data.datasets[2].data.push(batchStats.valueLoss);

    if (this.policyChart.data.labels.length > this.options.maxDataPoints) {
      this.policyChart.data.labels.shift();
      this.policyChart.data.datasets.forEach(dataset => {
        dataset.data.shift();
      });
    }

    this.policyChart.update('none');
  }

  /**
   * Reset all data
   */
  reset() {
    this.batchNumber = 0;

    // Reset summary stats
    if (this.statGames) this.statGames.textContent = '0';
    if (this.statAvgReward) this.statAvgReward.textContent = '0.00';
    if (this.statBestReward) this.statBestReward.textContent = '0.00';
    if (this.statCompletion) this.statCompletion.textContent = '0.0%';
    if (this.statTime) this.statTime.textContent = '0s';

    // Reset charts
    if (this.rewardChart) {
      this.rewardChart.data.labels = [];
      this.rewardChart.data.datasets.forEach(dataset => {
        dataset.data = [];
      });
      this.rewardChart.update();
    }

    if (this.lengthChart) {
      this.lengthChart.data.labels = [];
      this.lengthChart.data.datasets.forEach(dataset => {
        dataset.data = [];
      });
      this.lengthChart.update();
    }

    if (this.completionChart) {
      this.completionChart.data.labels = [];
      this.completionChart.data.datasets.forEach(dataset => {
        dataset.data = [];
      });
      this.completionChart.update();
    }

    if (this.policyChart) {
      this.policyChart.data.labels = [];
      this.policyChart.data.datasets.forEach(dataset => {
        dataset.data = [];
      });
      this.policyChart.update();
    }
  }

  /**
   * Show the visualizer
   */
  show() {
    if (this.container) {
      this.container.style.display = 'block';
    }
  }

  /**
   * Hide the visualizer
   */
  hide() {
    if (this.container) {
      this.container.style.display = 'none';
    }
  }

  /**
   * Dispose of resources
   */
  dispose() {
    // Destroy charts
    if (this.rewardChart) {
      this.rewardChart.destroy();
      this.rewardChart = null;
    }
    if (this.lengthChart) {
      this.lengthChart.destroy();
      this.lengthChart = null;
    }
    if (this.completionChart) {
      this.completionChart.destroy();
      this.completionChart = null;
    }
    if (this.policyChart) {
      this.policyChart.destroy();
      this.policyChart = null;
    }

    // Clear container
    if (this.container) {
      this.container.innerHTML = '';
    }

    // Clear session reference
    if (this.trainingSession) {
      this.trainingSession.onTrainingProgress = null;
      this.trainingSession = null;
    }

    this.isInitialized = false;
  }
}

