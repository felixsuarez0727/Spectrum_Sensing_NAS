# 🚀 Neural Architecture Search (NAS) for Spectrum Sensing

**Automated CNN Architecture Optimization for Wireless Spectrum Detection**

A complete Neural Architecture Search implementation for Spectrum Sensing using the DeepSense dataset, targeting optimal CNN 1D architectures with **<100K parameters**, **>98% accuracy**, and **<0.5ms latency**.

---

## 📊 Project Overview

### 🎯 Objectives
Based on the successful CNN implementation achieving **96.6% accuracy** with **184K parameters**, this NAS project aims to:

- **Reduce parameters**: Target <100K (vs 184K current)
- **Improve accuracy**: Target >98% (vs 96.6% current)  
- **Reduce latency**: Target <0.5ms inference time
- **Optimize efficiency**: Multi-objective optimization

### 📈 Baseline Performance
- **Exact Match Accuracy**: 96.6%
- **Average F1-Score**: 98.5%
- **Parameters**: 184,000
- **Model Size**: 0.7MB
- **Inference Time**: ~0.8ms

### 🎯 NAS Targets
- **Parameters**: <100,000 (-45% reduction)
- **Accuracy**: >98% (+1.4% improvement)
- **Latency**: <0.5ms (-37% reduction)
- **Model Size**: <0.5MB (-29% reduction)

---

## 🏗️ Architecture Search Space

### 🔍 Search Parameters

#### Conv1D Layers
- **Number of layers**: 2-5 conv layers
- **Filters**: [32, 48, 64, 96, 128, 144, 160, 192]
- **Kernel sizes**: [3, 5, 7, 9]
- **Activations**: ['relu', 'leaky_relu', 'swish', 'gelu']
- **Batch normalization**: [True, False]
- **Dropout rates**: [0.1, 0.2, 0.3, 0.4, 0.5]

#### Dense Layers
- **Number of layers**: 1-3 dense layers
- **Units**: [32, 48, 64, 96, 128]
- **Dropout rates**: [0.3, 0.4, 0.5]
- **Activations**: ['relu', 'leaky_relu', 'swish']

#### Advanced Features
- **Global pooling**: ['avg', 'max', 'adaptive']
- **Skip connections**: [True, False]
- **Attention mechanisms**: [True, False]
- **Separable convolutions**: [True, False]

### ⚡ Optimization Strategy
- **Search Algorithm**: Tree-structured Parzen Estimator (TPE)
- **Pruning**: Median pruner for early stopping
- **Trials**: 100 architecture evaluations
- **Multi-objective**: Accuracy (70%) + Efficiency (20%) + Latency (10%)

---

## 🚀 Quick Start

### 1. Installation

```bash
# Clone or navigate to project directory
cd /Users/jaime07ag/Library/Mobile\ Documents/com~apple~CloudDocs/Downloads/Elva/Spectrum_Sensing_NAS

# Install dependencies
pip install -r requirements.txt

# Verify installation
python -c "import tensorflow as tf; print(f'TensorFlow: {tf.__version__}')"
python -c "import optuna; print(f'Optuna: {optuna.__version__}')"
```

### 2. Run Neural Architecture Search

```bash
# Start NAS optimization
python code/nas_search.py

# Monitor progress (optional)
optuna-dashboard sqlite:///results/nas_study.db
```

### 3. Evaluate Results

```bash
# Comprehensive evaluation of best architecture
python code/nas_evaluate.py

# View results
open results/nas_evaluation_report.md
open results/nas_performance_summary.png
```

---

## 📁 Project Structure

```
Spectrum_Sensing_NAS/
├── code/                           # NAS implementation
│   ├── nas_search.py              # Main NAS optimization
│   ├── nas_evaluate.py            # Comprehensive evaluation
│   └── nas_utils.py               # Utilities and helpers
├── config/                         # Configuration
│   └── nas_config.py              # NAS search space and parameters
├── data/                           # Datasets
│   └── processed/                 # Preprocessed HDF5 files
│       ├── sdr_wifi_train.h5      # Training data
│       ├── sdr_wifi_val.h5        # Validation data
│       └── sdr_wifi_test.h5       # Test data
├── results/                        # NAS results and outputs
│   ├── nas_search_results.json    # Search results
│   ├── nas_evaluation_report.md   # Comprehensive report
│   └── *.png                      # Visualization plots
├── models/                         # Trained models
│   ├── nas_best_model.h5          # Best architecture found
│   └── nas_final_model.h5         # Final trained model
├── logs/                           # Training logs
├── requirements.txt                # Dependencies
└── README.md                       # This file
```

---

## 🔧 Configuration

### Key Parameters (`config/nas_config.py`)

```python
# Search constraints
SEARCH_CONSTRAINTS = {
    'max_parameters': 100000,      # < 100K parameters
    'target_accuracy': 0.98,       # > 98% accuracy
    'target_latency': 0.5,         # < 0.5ms latency
    'max_model_size_mb': 0.5,      # < 0.5MB
}

# Optimization weights
OPTIMIZATION_WEIGHTS = {
    'accuracy_weight': 0.7,        # 70% weight on accuracy
    'efficiency_weight': 0.2,      # 20% weight on efficiency
    'latency_weight': 0.1,         # 10% weight on latency
}

# NAS parameters
NAS_PARAMS = {
    'n_trials': 100,               # Number of trials
    'pruning': True,               # Enable pruning
    'timeout': 3600,               # 1 hour per trial
}
```

---

## 📊 Expected Results

### 🎯 Target Achievements

| Metric | Baseline CNN | NAS Target | Expected Improvement |
|--------|-------------|------------|---------------------|
| **Exact Match Accuracy** | 96.6% | >98.0% | +1.4% |
| **Parameters** | 184K | <100K | -45% |
| **Model Size** | 0.7MB | <0.5MB | -29% |
| **Inference Time** | 0.8ms | <0.5ms | -37% |
| **Average F1-Score** | 98.5% | >99.0% | +0.5% |

### 📈 Search Process

1. **Architecture Generation**: Optuna generates candidate architectures
2. **Constraint Filtering**: Filter architectures exceeding limits
3. **Training**: Train each architecture with early stopping
4. **Evaluation**: Comprehensive metrics calculation
5. **Optimization**: Multi-objective score calculation
6. **Selection**: Best architecture selection

---

## 🔍 Search Space Details

### Architecture Building Blocks

```python
# Example NAS architecture generation
def build_nas_architecture(trial):
    # Sample parameters
    num_conv_layers = trial.suggest_categorical('num_conv_layers', [2, 3, 4, 5])
    filters = trial.suggest_categorical('conv_filters', [32, 64, 128, 192])
    kernel_size = trial.suggest_categorical('kernel_size', [3, 5, 7])
    
    # Build architecture dynamically
    model = build_cnn_1d_model(
        conv_layers=num_conv_layers,
        filters=filters,
        kernel_size=kernel_size,
        # ... other parameters
    )
    return model
```

### Multi-Objective Optimization

```python
def nas_objective(accuracy, num_params, latency_ms):
    # Combined optimization score
    score = (0.7 * accuracy +                    # 70% accuracy weight
             0.2 * (1 - min(num_params/100000, 1)) +  # 20% efficiency weight
             0.1 * (1 - min(latency_ms/0.5, 1)))     # 10% latency weight
    return score
```

---

## 📊 Evaluation Metrics

### Global Metrics
- **Exact Match Accuracy**: All 4 channels correct simultaneously
- **Hamming Loss**: Fraction of incorrect labels
- **Subset Accuracy**: Jaccard similarity between predicted and true sets

### Per-Channel Metrics
- **Accuracy**: Correct predictions per channel
- **Precision**: True positives / (True positives + False positives)
- **Recall**: True positives / (True positives + False negatives)
- **F1-Score**: Harmonic mean of precision and recall
- **AUC**: Area under ROC curve

### Efficiency Metrics
- **Parameter Count**: Total trainable parameters
- **Model Size**: File size in MB
- **Inference Time**: Average inference time per sample

---

## 🚀 Usage Examples

### Basic NAS Search

```python
from code.nas_search import run_nas_search

# Run NAS with default parameters
study = run_nas_search(n_trials=100)

# Analyze results
best_trial = study.best_trial
print(f"Best accuracy: {best_trial.user_attrs['exact_match_accuracy']:.3f}")
print(f"Best parameters: {best_trial.user_attrs['num_parameters']:,}")
```

### Custom Configuration

```python
from config.nas_config import *

# Modify search constraints
SEARCH_CONSTRAINTS['max_parameters'] = 80000  # Stricter constraint
SEARCH_CONSTRAINTS['target_accuracy'] = 0.985  # Higher target

# Run with custom constraints
study = run_nas_search(n_trials=50)
```

### Evaluation and Visualization

```python
from code.nas_evaluate import main as evaluate_nas

# Run comprehensive evaluation
evaluate_nas()

# Results will be saved to results/ directory
```

---

## 🔧 Advanced Configuration

### Custom Search Space

```python
# Modify search space in config/nas_config.py
CONV_SEARCH_SPACE = {
    'num_conv_layers': [3, 4, 5],              # Fewer options
    'filters_per_layer': [64, 128, 192],       # Larger filters only
    'kernel_sizes': [5, 7],                    # Larger kernels only
    'activation_functions': ['relu', 'swish'], # Fewer activations
}
```

### Multi-GPU Support

```python
# For systems with multiple GPUs
import tensorflow as tf

# Configure GPU memory growth
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
```

---

## 📈 Monitoring and Debugging

### Optuna Dashboard

```bash
# Start web dashboard for monitoring
optuna-dashboard sqlite:///results/nas_study.db

# Access at http://localhost:8080
```

### TensorBoard Integration

```python
# TensorBoard logging during training
tensorboard_callback = keras.callbacks.TensorBoard(
    log_dir='logs/tensorboard',
    histogram_freq=1
)
```

### Progress Monitoring

```python
# Real-time progress monitoring
def progress_callback(study, trial):
    print(f"Trial {trial.number}: Score = {trial.value:.3f}")

study.optimize(objective, n_trials=100, callbacks=[progress_callback])
```

---

## 🐛 Troubleshooting

### Common Issues

#### 1. TensorFlow Installation Issues
```bash
# For macOS ARM
pip uninstall tensorflow
pip install tensorflow-macos==2.16.2

# Verify installation
python -c "import tensorflow as tf; print(tf.config.list_physical_devices())"
```

#### 2. Memory Issues
```python
# Reduce batch size in config
TRAINING_CONFIG['batch_size'] = 32  # Instead of 64

# Reduce augmentation factor
AUGMENTATION_CONFIG['augmentation_factor'] = 2  # Instead of 3
```

#### 3. Slow Search
```python
# Reduce trials for faster testing
NAS_PARAMS['n_trials'] = 20

# Enable more aggressive pruning
study = optuna.create_study(
    pruner=optuna.pruners.MedianPruner(
        n_startup_trials=3,    # Reduced from 5
        n_warmup_steps=5,      # Reduced from 10
    )
)
```

### Performance Optimization

#### 1. Faster Training
```python
# Use mixed precision training
from tensorflow.keras.mixed_precision import set_global_policy
set_global_policy('mixed_float16')

# Enable XLA compilation
tf.config.optimizer.set_jit(True)
```

#### 2. Parallel Search
```python
# Multi-process search (if system supports)
NAS_PARAMS['n_jobs'] = 4  # Use 4 parallel processes
```

---

## 📚 References and Resources

### Papers and Datasets
- **DeepSense Paper**: "Fast Wideband Spectrum Sensing Through Real-Time In-the-Loop Deep Learning"
- **IEEE INFOCOM 2021**: Original DeepSense implementation
- **DeepSense Dataset**: [deeplearningsensing.com](https://deeplearningsensing.com/)

### Frameworks and Tools
- **Optuna**: [optuna.org](https://optuna.org/) - Hyperparameter optimization
- **TensorFlow**: [tensorflow.org](https://tensorflow.org/) - Deep learning framework
- **Keras**: [keras.io](https://keras.io/) - High-level neural networks API

### Related Research
- **Neural Architecture Search**: Survey of NAS methods and applications
- **Spectrum Sensing**: Cognitive radio and dynamic spectrum access
- **CNN 1D**: One-dimensional convolutional neural networks for time series

---

## 🤝 Contributing

### Development Setup
```bash
# Install development dependencies
pip install -r requirements.txt
pip install pytest black flake8

# Run tests
pytest tests/

# Format code
black code/
flake8 code/
```

### Adding New Search Spaces
```python
# Extend search space in config/nas_config.py
ADVANCED_SEARCH_SPACE.update({
    'attention_heads': [1, 2, 4, 8],
    'transformer_layers': [1, 2, 3],
})
```

---

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

## 🙏 Acknowledgments

- **DeepSense Team**: For providing the dataset and baseline implementation
- **Optuna Community**: For the excellent hyperparameter optimization framework
- **TensorFlow Team**: For the robust deep learning framework
- **IEEE INFOCOM 2021**: For publishing the original research

---

**🎯 Ready to find the optimal CNN architecture for Spectrum Sensing!**

*Start your NAS journey with: `python code/nas_search.py`*

