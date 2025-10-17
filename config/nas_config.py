"""
Neural Architecture Search Configuration for Spectrum Sensing
Based on successful CNN implementation with 96.6% accuracy
"""

import os
import numpy as np

# ============================================================================
# DATASET CONFIGURATION
# ============================================================================

# Dataset paths
DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
PROCESSED_DIR = os.path.join(DATA_DIR, 'processed')

# Dataset files
TRAIN_FILE = os.path.join(PROCESSED_DIR, 'sdr_wifi_train.h5')
VAL_FILE = os.path.join(PROCESSED_DIR, 'sdr_wifi_val.h5')
TEST_FILE = os.path.join(PROCESSED_DIR, 'sdr_wifi_test.h5')

# Dataset parameters (from successful CNN implementation)
INPUT_SHAPE = (128, 2)  # IQ samples
OUTPUT_SHAPE = 4        # 4 channels
N_CHANNELS = 4
BUFFER_SIZE = 128
STRIDE = 12
SAMPLING_RATE = 20e6    # 20 MS/s

# ============================================================================
# NAS SEARCH SPACE CONFIGURATION
# ============================================================================

# Search space constraints (targeting <100K parameters)
SEARCH_CONSTRAINTS = {
    'max_parameters': 100000,      # < 100K parameters (vs 184K current CNN)
    'max_layers': 15,              # Limited depth
    'min_layers': 3,               # Minimum viable depth
    'target_accuracy': 0.98,       # > 98% exact match (vs 96.6% current)
    'target_latency': 0.5,         # < 0.5ms inference
    'max_model_size_mb': 0.5,      # < 0.5MB (vs 0.7MB current)
}

# Conv1D layer search space
CONV_SEARCH_SPACE = {
    'num_conv_layers': [2, 3, 4, 5],           # 2-5 conv layers
    'filters_per_layer': [32, 48, 64, 96, 128, 144, 160, 192],  # Filter counts
    'kernel_sizes': [3, 5, 7, 9],              # Kernel sizes
    'activation_functions': ['relu', 'leaky_relu', 'swish', 'gelu'],
    'batch_norm': [True, False],               # Batch normalization
    'dropout_rates': [0.1, 0.2, 0.3, 0.4, 0.5],  # Dropout rates
    'pooling_types': ['max', 'avg', 'adaptive_avg'],  # Pooling strategies
    'pooling_sizes': [2, 4],                   # Pooling sizes
}

# Dense layer search space
DENSE_SEARCH_SPACE = {
    'num_dense_layers': [1, 2, 3],             # 1-3 dense layers
    'dense_units': [32, 48, 64, 96, 128],      # Dense layer sizes
    'dense_dropout_rates': [0.3, 0.4, 0.5],    # Dense dropout rates
    'dense_activation': ['relu', 'leaky_relu', 'swish'],
}

# Advanced architecture search space
ADVANCED_SEARCH_SPACE = {
    'skip_connections': [True, False],         # Residual connections
    'global_pooling': ['avg', 'max', 'adaptive'],  # Global pooling
    'attention_mechanisms': [True, False],     # Self-attention
    'separable_convs': [True, False],          # Depthwise separable convolutions
    'dilated_convs': [True, False],            # Dilated convolutions
}

# ============================================================================
# NAS OPTIMIZATION CONFIGURATION
# ============================================================================

# NAS search parameters
NAS_PARAMS = {
    'n_trials': 100,                           # Number of architecture trials
    'n_jobs': 1,                               # Parallel jobs (1 for stability)
    'timeout': 3600,                           # 1 hour timeout per trial
    'pruning': True,                           # Enable pruning
    'pruning_percentile': 10,                  # Prune worst 10%
}

# Multi-objective optimization weights
OPTIMIZATION_WEIGHTS = {
    'accuracy_weight': 0.7,                    # 70% weight on accuracy
    'efficiency_weight': 0.2,                  # 20% weight on efficiency (params)
    'latency_weight': 0.1,                     # 10% weight on latency
}

# Early stopping for architecture evaluation
EARLY_STOPPING = {
    'patience': 5,                             # Stop if no improvement
    'min_delta': 0.001,                        # Minimum improvement
    'monitor': 'val_exact_match_accuracy',     # Metric to monitor
}

# ============================================================================
# TRAINING CONFIGURATION (from successful CNN)
# ============================================================================

# Training parameters (proven to work)
TRAINING_CONFIG = {
    'batch_size': 64,
    'epochs': 50,                              # Reduced for NAS (vs 100 for final)
    'learning_rate': 0.001,
    'optimizer': 'adam',
    'loss': 'binary_crossentropy',
    'metrics': ['binary_accuracy', 'precision', 'recall', 'auc'],
}

# Callbacks (from successful CNN implementation)
CALLBACKS_CONFIG = {
    'early_stopping': {
        'patience': 15,
        'restore_best_weights': True,
        'monitor': 'val_loss'
    },
    'reduce_lr': {
        'factor': 0.5,
        'patience': 7,
        'min_lr': 1e-6
    },
    'model_checkpoint': {
        'save_best_only': True,
        'monitor': 'val_loss'
    }
}

# Data augmentation (from successful CNN)
AUGMENTATION_CONFIG = {
    'noise_std': 0.01,
    'phase_noise_std': 0.03,
    'amplitude_factor_range': [0.8, 1.2],
    'frequency_offset_range': [-1000, 1000],  # Hz
    'iq_imbalance_amplitude': 0.02,
    'iq_imbalance_phase': 0.01,
    'augmentation_factor': 3,                  # 3x data augmentation
}

# ============================================================================
# EVALUATION CONFIGURATION
# ============================================================================

# Evaluation metrics (from successful CNN)
EVALUATION_METRICS = {
    'global_metrics': [
        'exact_match_accuracy',
        'hamming_loss',
        'subset_accuracy'
    ],
    'per_channel_metrics': [
        'accuracy',
        'precision',
        'recall',
        'f1_score',
        'auc'
    ]
}

# Threshold optimization (from successful CNN)
THRESHOLD_CONFIG = {
    'optimization_method': 'precision_recall',
    'metric': 'f1_score',
    'threshold_range': [0.1, 0.9],
    'step_size': 0.01,
}

# ============================================================================
# OUTPUT CONFIGURATION
# ============================================================================

# Results directory
RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')
MODELS_DIR = os.path.join(os.path.dirname(__file__), '..', 'models')
LOGS_DIR = os.path.join(os.path.dirname(__file__), '..', 'logs')

# Ensure directories exist
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)

# File paths
BEST_MODEL_PATH = os.path.join(MODELS_DIR, 'nas_best_model.h5')
FINAL_MODEL_PATH = os.path.join(MODELS_DIR, 'nas_final_model.h5')
NAS_RESULTS_PATH = os.path.join(RESULTS_DIR, 'nas_search_results.json')
COMPARISON_PATH = os.path.join(RESULTS_DIR, 'nas_vs_cnn_comparison.json')

# ============================================================================
# VERIFICATION FUNCTIONS
# ============================================================================

def verify_config():
    """Verify that all configuration parameters are valid"""
    print("🔧 Verifying NAS Configuration...")
    
    # Check dataset files exist
    dataset_files = [TRAIN_FILE, VAL_FILE, TEST_FILE]
    for file_path in dataset_files:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Dataset file not found: {file_path}")
    
    print(f"✅ Dataset files found:")
    for file_path in dataset_files:
        print(f"   - {os.path.basename(file_path)}")
    
    # Check search space constraints
    if SEARCH_CONSTRAINTS['max_parameters'] > 150000:
        print("⚠️  Warning: max_parameters > 150K, may be too large")
    
    if SEARCH_CONSTRAINTS['target_accuracy'] > 0.99:
        print("⚠️  Warning: target_accuracy > 99%, very ambitious")
    
    print("✅ Configuration verified successfully!")
    return True

def print_config_summary():
    """Print a summary of the NAS configuration"""
    print("\n" + "="*60)
    print("🚀 NEURAL ARCHITECTURE SEARCH CONFIGURATION")
    print("="*60)
    print(f"📊 Dataset: {os.path.basename(TRAIN_FILE)}")
    print(f"🎯 Target Parameters: < {SEARCH_CONSTRAINTS['max_parameters']:,}")
    print(f"🎯 Target Accuracy: > {SEARCH_CONSTRAINTS['target_accuracy']*100:.1f}%")
    print(f"🎯 Target Latency: < {SEARCH_CONSTRAINTS['target_latency']}ms")
    print(f"🔍 Search Trials: {NAS_PARAMS['n_trials']}")
    print(f"⚡ Optimization: {OPTIMIZATION_WEIGHTS}")
    print("="*60)

if __name__ == "__main__":
    verify_config()
    print_config_summary()

