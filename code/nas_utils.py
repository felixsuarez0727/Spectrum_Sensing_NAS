"""
Neural Architecture Search Utilities for Spectrum Sensing
Based on successful CNN implementation with 96.6% accuracy
"""

import os
import sys
import numpy as np
import h5py
import tensorflow as tf
from tensorflow import keras
from typing import Dict, List, Tuple, Any, Optional
import json
import time
import pickle
from sklearn.metrics import (
    accuracy_score, precision_recall_curve, roc_auc_score,
    hamming_loss, jaccard_score
)

# Add config to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'config'))
from nas_config import *

# ============================================================================
# DATA LOADING AND PREPROCESSING
# ============================================================================

def load_h5_dataset(file_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load dataset from HDF5 file, supporting both flat and grouped layouts.
    Supported layouts:
      - Root datasets: /iq_data, /labels
      - Grouped by split: /train|/val|/test with datasets inside
      - Alternative names: X/y, data/labels, features/labels
    Args:
        file_path: Path to HDF5 file
    Returns:
        X: IQ data with shape (n_samples, 128, 2)
        y: Labels with shape (n_samples, 4)
    """
    def collect_datasets(h5obj):
        datasets = {}
        def _collector(name, obj):
            if isinstance(obj, h5py.Dataset):
                datasets[name] = obj
        h5obj.visititems(_collector)
        return datasets

    def try_get_pair(dsets: Dict[str, h5py.Dataset]) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        priority_pairs = [
            ("iq_data", "labels"), ("X", "y"), ("data", "labels"), ("features", "labels")
        ]
        # Exact or suffix match
        names = list(dsets.keys())
        for a, b in priority_pairs:
            a_path = next((n for n in names if n.endswith("/" + a) or n == a), None)
            b_path = next((n for n in names if n.endswith("/" + b) or n == b), None)
            if a_path and b_path:
                return dsets[a_path][:], dsets[b_path][:]
        # Fallback: infer by shapes (two datasets sharing same first dim)
        for i_name, i_ds in dsets.items():
            if not hasattr(i_ds, 'shape') or len(i_ds.shape) < 2:
                continue
            for j_name, j_ds in dsets.items():
                if i_name == j_name or not hasattr(j_ds, 'shape'):
                    continue
                if i_ds.shape[0] == j_ds.shape[0]:
                    Xi, yj = i_ds[:], j_ds[:]
                    # Heuristic: X is likely (n, 128, 2) or (n, ..., 2)
                    if Xi.ndim >= 3 and Xi.shape[-1] in (2, ):  # IQ channels
                        return Xi, yj
        return None, None

    with h5py.File(file_path, 'r') as f:
        # Case 1: flat at root
        if 'iq_data' in f and 'labels' in f:
            X = f['iq_data'][:]
            y = f['labels'][:]
        else:
            # Case 2: grouped by split or nested
            # Try common split group names first
            possible_groups = [g for g in ('train', 'val', 'validation', 'test') if g in f]
            X = y = None
            if possible_groups:
                # Use the only group available in file
                grp_name = possible_groups[0]
                grp = f[grp_name]
                dsets = collect_datasets(grp)
                X, y = try_get_pair(dsets)
            if X is None or y is None:
                # Fallback: scan entire file for datasets
                dsets = collect_datasets(f)
                X, y = try_get_pair(dsets)
            if X is None or y is None:
                raise RuntimeError(
                    f"Unable to locate datasets 'iq_data' and 'labels' (or equivalents) in {file_path}. "
                    f"Found top-level keys: {list(f.keys())}"
                )

    print(f"📊 Loaded {X.shape[0]:,} samples from {os.path.basename(file_path)}")
    return X, y

def load_datasets() -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """
    Load all datasets (train, validation, test)
    Returns:
        datasets: Dictionary with 'train', 'val', 'test' keys
    """
    datasets = {}
    
    print("🔄 Loading datasets...")
    datasets['train'] = load_h5_dataset(TRAIN_FILE)
    datasets['val'] = load_h5_dataset(VAL_FILE)
    datasets['test'] = load_h5_dataset(TEST_FILE)
    
    # Verify shapes
    for split, (X, y) in datasets.items():
        assert X.shape[1:] == INPUT_SHAPE, f"Wrong input shape in {split}: {X.shape}"
        assert y.shape[1] == OUTPUT_SHAPE, f"Wrong output shape in {split}: {y.shape}"
    
    print("✅ All datasets loaded successfully!")
    return datasets

def apply_data_augmentation(X: np.ndarray, y: np.ndarray, 
                          factor: int = AUGMENTATION_CONFIG['augmentation_factor']) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply RF-specific data augmentation
    Args:
        X: Input IQ data
        y: Labels
        factor: Augmentation factor
    Returns:
        X_aug: Augmented input data
        y_aug: Augmented labels
    """
    if factor <= 1:
        return X, y
    
    print(f"🔧 Applying data augmentation (factor: {factor}x)...")
    
    X_aug = [X]
    y_aug = [y]
    
    for i in range(factor - 1):
        X_augmented = X.copy()
        
        # Apply random augmentation to each sample
        for j in range(X.shape[0]):
            X_augmented[j] = augment_iq_signal(X[j])
        
        X_aug.append(X_augmented)
        y_aug.append(y)
    
    X_final = np.vstack(X_aug)
    y_final = np.vstack(y_aug)
    
    print(f"✅ Data augmentation completed: {X.shape[0]:,} → {X_final.shape[0]:,} samples")
    return X_final, y_final

def augment_iq_signal(iq_signal: np.ndarray) -> np.ndarray:
    """
    Apply RF-specific augmentation to a single IQ signal
    Args:
        iq_signal: IQ signal with shape (128, 2)
    Returns:
        augmented_signal: Augmented IQ signal
    """
    config = AUGMENTATION_CONFIG
    augmented = iq_signal.copy()
    
    # Add Gaussian noise
    noise = np.random.normal(0, config['noise_std'], augmented.shape)
    augmented += noise
    
    # Phase noise
    phase_noise = np.random.normal(0, config['phase_noise_std'], augmented.shape[0])
    for i in range(augmented.shape[0]):
        phase_shift = np.exp(1j * phase_noise[i])
        augmented[i, 0] *= np.real(phase_shift)  # I component
        augmented[i, 1] *= np.imag(phase_shift)  # Q component
    
    # Amplitude variation
    amplitude_factor = np.random.uniform(*config['amplitude_factor_range'])
    augmented *= amplitude_factor
    
    # Frequency offset (simplified)
    if np.random.random() > 0.5:
        freq_offset = np.random.uniform(*config['frequency_offset_range'])
        # Apply simple frequency shift
        t = np.arange(augmented.shape[0]) / SAMPLING_RATE
        phase_offset = 2 * np.pi * freq_offset * t
        for i in range(augmented.shape[1]):
            augmented[:, i] *= np.cos(phase_offset)
    
    # I/Q imbalance
    iq_amplitude_imbalance = 1 + np.random.uniform(-config['iq_imbalance_amplitude'], 
                                                   config['iq_imbalance_amplitude'])
    iq_phase_imbalance = np.random.uniform(-config['iq_imbalance_phase'], 
                                          config['iq_imbalance_phase'])
    
    augmented[:, 1] *= iq_amplitude_imbalance
    augmented[:, 1] += np.sin(iq_phase_imbalance) * augmented[:, 0]
    
    return augmented

# ============================================================================
# ARCHITECTURE BUILDING
# ============================================================================

def build_nas_architecture(trial, input_shape: Tuple[int, int] = INPUT_SHAPE, 
                          output_shape: int = OUTPUT_SHAPE) -> keras.Model:
    """
    Build CNN architecture based on Optuna trial parameters
    Args:
        trial: Optuna trial object
        input_shape: Input shape (height, width)
        output_shape: Output shape
    Returns:
        model: Compiled Keras model
    """
    # Sample architecture parameters
    num_conv_layers = trial.suggest_categorical('num_conv_layers', 
                                               CONV_SEARCH_SPACE['num_conv_layers'])
    num_dense_layers = trial.suggest_categorical('num_dense_layers',
                                                DENSE_SEARCH_SPACE['num_dense_layers'])
    
    # Build model
    inputs = keras.Input(shape=input_shape, name='iq_input')
    x = inputs
    
    # Conv1D layers
    for i in range(num_conv_layers):
        # Sample layer parameters
        filters = trial.suggest_categorical(f'conv_filters_{i}',
                                          CONV_SEARCH_SPACE['filters_per_layer'])
        kernel_size = trial.suggest_categorical(f'conv_kernel_{i}',
                                              CONV_SEARCH_SPACE['kernel_sizes'])
        activation = trial.suggest_categorical(f'conv_activation_{i}',
                                             CONV_SEARCH_SPACE['activation_functions'])
        use_batch_norm = trial.suggest_categorical(f'use_bn_{i}',
                                                 CONV_SEARCH_SPACE['batch_norm'])
        dropout_rate = trial.suggest_categorical(f'conv_dropout_{i}',
                                               CONV_SEARCH_SPACE['dropout_rates'])
        
        # Conv1D layer
        x = keras.layers.Conv1D(
            filters=filters,
            kernel_size=kernel_size,
            activation=activation,
            padding='same',
            name=f'conv1d_{i+1}'
        )(x)
        
        # Batch normalization
        if use_batch_norm:
            x = keras.layers.BatchNormalization(name=f'bn_{i+1}')(x)
        
        # Pooling (every other layer)
        if i % 2 == 1:
            pool_type = trial.suggest_categorical(f'pool_type_{i}',
                                                CONV_SEARCH_SPACE['pooling_types'])
            if pool_type == 'max':
                x = keras.layers.MaxPooling1D(2, name=f'pool_{i+1}')(x)
            elif pool_type == 'avg':
                x = keras.layers.AveragePooling1D(2, name=f'pool_{i+1}')(x)
        
        # Dropout
        x = keras.layers.Dropout(dropout_rate, name=f'dropout_{i+1}')(x)
    
    # Global pooling
    global_pool = trial.suggest_categorical('global_pool',
                                          ADVANCED_SEARCH_SPACE['global_pooling'])
    if global_pool == 'avg':
        x = keras.layers.GlobalAveragePooling1D(name='global_avg_pool')(x)
    elif global_pool == 'max':
        x = keras.layers.GlobalMaxPooling1D(name='global_max_pool')(x)
    
    # Dense layers
    for i in range(num_dense_layers):
        units = trial.suggest_categorical(f'dense_units_{i}',
                                        DENSE_SEARCH_SPACE['dense_units'])
        activation = trial.suggest_categorical(f'dense_activation_{i}',
                                             DENSE_SEARCH_SPACE['dense_activation'])
        dropout_rate = trial.suggest_categorical(f'dense_dropout_{i}',
                                               DENSE_SEARCH_SPACE['dense_dropout_rates'])
        
        x = keras.layers.Dense(units, activation=activation, 
                              name=f'dense_{i+1}')(x)
        x = keras.layers.Dropout(dropout_rate, name=f'dense_dropout_{i+1}')(x)
    
    # Output layer
    outputs = keras.layers.Dense(output_shape, activation='sigmoid', 
                                name='output')(x)
    
    model = keras.Model(inputs=inputs, outputs=outputs, name='nas_cnn_1d')
    return model

def count_model_parameters(model: keras.Model) -> int:
    """Count trainable parameters in model"""
    return model.count_params()

def estimate_model_size_mb(model: keras.Model) -> float:
    """Estimate model size in MB"""
    # Rough estimation: 4 bytes per parameter (float32)
    params = count_model_parameters(model)
    size_mb = (params * 4) / (1024 * 1024)
    return size_mb

# ============================================================================
# TRAINING AND EVALUATION
# ============================================================================

def train_nas_model(model: keras.Model, X_train: np.ndarray, y_train: np.ndarray,
                   X_val: np.ndarray, y_val: np.ndarray, 
                   epochs: int = TRAINING_CONFIG['epochs']) -> keras.Model:
    """
    Train NAS model with early stopping
    Args:
        model: Keras model to train
        X_train, y_train: Training data
        X_val, y_val: Validation data
        epochs: Maximum epochs
    Returns:
        trained_model: Best model during training
    """
    # Compile model
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=TRAINING_CONFIG['learning_rate']),
        loss=TRAINING_CONFIG['loss'],
        metrics=TRAINING_CONFIG['metrics']
    )
    
    # Setup callbacks
    callbacks = []
    
    # Early stopping
    early_stopping = keras.callbacks.EarlyStopping(
        monitor=CALLBACKS_CONFIG['early_stopping']['monitor'],
        patience=CALLBACKS_CONFIG['early_stopping']['patience'],
        restore_best_weights=CALLBACKS_CONFIG['early_stopping']['restore_best_weights'],
        verbose=1
    )
    callbacks.append(early_stopping)
    
    # Reduce learning rate
    reduce_lr = keras.callbacks.ReduceLROnPlateau(
        factor=CALLBACKS_CONFIG['reduce_lr']['factor'],
        patience=CALLBACKS_CONFIG['reduce_lr']['patience'],
        min_lr=CALLBACKS_CONFIG['reduce_lr']['min_lr'],
        verbose=1
    )
    callbacks.append(reduce_lr)
    
    # Train model
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=TRAINING_CONFIG['batch_size'],
        callbacks=callbacks,
        verbose=0
    )
    
    return model

def evaluate_nas_model(model: keras.Model, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
    """
    Evaluate NAS model and return comprehensive metrics
    Args:
        model: Trained Keras model
        X_test, y_test: Test data
    Returns:
        metrics: Dictionary of evaluation metrics
    """
    # Get predictions
    y_proba = model.predict(X_test, verbose=0)
    y_pred = (y_proba >= 0.5).astype(int)
    
    # Global metrics
    exact_match_accuracy = accuracy_score(y_test, y_pred)
    hamming_loss_score = hamming_loss(y_test, y_pred)
    subset_accuracy = jaccard_score(y_test, y_pred, average='samples')
    
    # Per-channel metrics
    channel_metrics = []
    for i in range(OUTPUT_SHAPE):
        if len(np.unique(y_test[:, i])) > 1:  # Skip channels with only one class
            accuracy = accuracy_score(y_test[:, i], y_pred[:, i])
            precision, recall, thresholds = precision_recall_curve(y_test[:, i], y_proba[:, i])
            f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
            best_f1_idx = np.argmax(f1_scores)
            
            channel_metrics.append({
                'accuracy': accuracy,
                'precision': precision[best_f1_idx],
                'recall': recall[best_f1_idx],
                'f1_score': f1_scores[best_f1_idx],
                'auc': roc_auc_score(y_test[:, i], y_proba[:, i])
            })
    
    # Average metrics (excluding channels with single class)
    if channel_metrics:
        avg_accuracy = np.mean([m['accuracy'] for m in channel_metrics])
        avg_precision = np.mean([m['precision'] for m in channel_metrics])
        avg_recall = np.mean([m['recall'] for m in channel_metrics])
        avg_f1 = np.mean([m['f1_score'] for m in channel_metrics])
        avg_auc = np.mean([m['auc'] for m in channel_metrics])
    else:
        avg_accuracy = avg_precision = avg_recall = avg_f1 = avg_auc = 0.0
    
    metrics = {
        # Global metrics
        'exact_match_accuracy': exact_match_accuracy,
        'hamming_loss': hamming_loss_score,
        'subset_accuracy': subset_accuracy,
        
        # Average metrics
        'average_accuracy': avg_accuracy,
        'average_precision': avg_precision,
        'average_recall': avg_recall,
        'average_f1_score': avg_f1,
        'average_auc': avg_auc,
        
        # Model metrics
        'num_parameters': count_model_parameters(model),
        'model_size_mb': estimate_model_size_mb(model),
        
        # Per-channel metrics
        'channel_metrics': channel_metrics
    }
    
    return metrics

def measure_inference_time(model: keras.Model, X_sample: np.ndarray, 
                          num_runs: int = 100) -> float:
    """
    Measure inference time per sample
    Args:
        model: Trained Keras model
        X_sample: Sample input data
        num_runs: Number of runs for averaging
    Returns:
        avg_time_ms: Average inference time in milliseconds
    """
    # Warmup
    _ = model.predict(X_sample[:1], verbose=0)
    
    # Measure inference time
    start_time = time.time()
    for _ in range(num_runs):
        _ = model.predict(X_sample[:1], verbose=0)
    end_time = time.time()
    
    avg_time_ms = (end_time - start_time) / num_runs * 1000
    return avg_time_ms

# ============================================================================
# NAS OPTIMIZATION
# ============================================================================

def nas_objective(accuracy: float, num_params: int, latency_ms: float) -> float:
    """
    Multi-objective function for NAS optimization
    Args:
        accuracy: Model accuracy
        num_params: Number of parameters
        latency_ms: Inference latency in milliseconds
    Returns:
        score: Combined optimization score
    """
    weights = OPTIMIZATION_WEIGHTS
    
    # Normalize parameters (target: <100K)
    param_score = 1 - min(num_params / SEARCH_CONSTRAINTS['max_parameters'], 1)
    
    # Normalize latency (target: <0.5ms)
    latency_score = 1 - min(latency_ms / SEARCH_CONSTRAINTS['target_latency'], 1)
    
    # Combined score
    score = (weights['accuracy_weight'] * accuracy +
             weights['efficiency_weight'] * param_score +
             weights['latency_weight'] * latency_score)
    
    return score

def save_nas_results(results: Dict[str, Any], file_path: str):
    """Save NAS search results to JSON file"""
    # Convert numpy types to Python types for JSON serialization
    def convert_numpy(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, dict):
            return {k: convert_numpy(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy(item) for item in obj]
        return obj
    
    results_serializable = convert_numpy(results)
    
    with open(file_path, 'w') as f:
        json.dump(results_serializable, f, indent=2)
    
    print(f"💾 NAS results saved to {file_path}")

def load_nas_results(file_path: str) -> Dict[str, Any]:
    """Load NAS search results from JSON file"""
    with open(file_path, 'r') as f:
        results = json.load(f)
    return results

# ============================================================================
# COMPARISON WITH BASELINE CNN
# ============================================================================

def compare_with_baseline_cnn(nas_metrics: Dict[str, float]) -> Dict[str, Any]:
    """
    Compare NAS results with baseline CNN (96.6% accuracy, 184K params)
    Args:
        nas_metrics: NAS model metrics
    Returns:
        comparison: Comparison results
    """
    baseline_cnn = {
        'exact_match_accuracy': 0.966,
        'average_f1_score': 0.985,
        'average_auc': 0.9997,
        'num_parameters': 184000,
        'model_size_mb': 0.7
    }
    
    comparison = {
        'nas_results': nas_metrics,
        'baseline_cnn': baseline_cnn,
        'improvements': {
            'accuracy_improvement': nas_metrics['exact_match_accuracy'] - baseline_cnn['exact_match_accuracy'],
            'parameter_reduction': baseline_cnn['num_parameters'] - nas_metrics['num_parameters'],
            'size_reduction_mb': baseline_cnn['model_size_mb'] - nas_metrics['model_size_mb'],
            'parameter_reduction_percent': (1 - nas_metrics['num_parameters'] / baseline_cnn['num_parameters']) * 100,
            'size_reduction_percent': (1 - nas_metrics['model_size_mb'] / baseline_cnn['model_size_mb']) * 100
        }
    }
    
    return comparison

# ============================================================================
# VERIFICATION FUNCTIONS
# ============================================================================

def verify_nas_implementation():
    """Verify that NAS implementation is working correctly"""
    print("🔍 Verifying NAS implementation...")
    
    try:
        # Test data loading
        datasets = load_datasets()
        print("✅ Data loading verified")
        
        # Test model building
        import optuna
        study = optuna.create_study(direction='maximize')
        trial = study.ask()
        model = build_nas_architecture(trial)
        print("✅ Model building verified")
        
        # Test parameter counting
        params = count_model_parameters(model)
        print(f"✅ Parameter counting verified: {params:,} parameters")
        
        # Test model size estimation
        size_mb = estimate_model_size_mb(model)
        print(f"✅ Model size estimation verified: {size_mb:.2f} MB")
        
        print("🎉 NAS implementation verification completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ NAS implementation verification failed: {e}")
        return False

if __name__ == "__main__":
    verify_nas_implementation()
