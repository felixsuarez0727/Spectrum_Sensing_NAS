"""
Neural Architecture Search (NAS) for Spectrum Sensing
Automated search for optimal CNN 1D architectures
Target: <100K parameters, >98% accuracy, <0.5ms latency
"""

import os
import sys
import time
import json
import optuna
import numpy as np
import tensorflow as tf
from tensorflow import keras
from typing import Dict, Any, Tuple
import warnings
warnings.filterwarnings('ignore')

# Add config and utils to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'config'))
sys.path.append(os.path.dirname(__file__))

from nas_config import *
from nas_utils import *

# ============================================================================
# OPTUNA OBJECTIVE FUNCTION
# ============================================================================

def objective(trial) -> float:
    """
    Optuna objective function for NAS optimization
    Args:
        trial: Optuna trial object
    Returns:
        score: Combined optimization score
    """
    print(f"\n🔍 Trial {trial.number}: Starting architecture search...")
    
    try:
        # Load datasets (cached for efficiency)
        if not hasattr(objective, 'datasets'):
            objective.datasets = load_datasets()
        
        datasets = objective.datasets
        X_train, y_train = datasets['train']
        X_val, y_val = datasets['val']
        X_test, y_test = datasets['test']
        
        # Apply data augmentation
        X_train_aug, y_train_aug = apply_data_augmentation(X_train, y_train)
        
        # Build architecture
        model = build_nas_architecture(trial)
        
        # Check parameter constraints
        num_params = count_model_parameters(model)
        if num_params > SEARCH_CONSTRAINTS['max_parameters']:
            print(f"❌ Too many parameters: {num_params:,} > {SEARCH_CONSTRAINTS['max_parameters']:,}")
            raise optuna.TrialPruned()
        
        model_size_mb = estimate_model_size_mb(model)
        if model_size_mb > SEARCH_CONSTRAINTS['max_model_size_mb']:
            print(f"❌ Model too large: {model_size_mb:.2f}MB > {SEARCH_CONSTRAINTS['max_model_size_mb']}MB")
            raise optuna.TrialPruned()
        
        print(f"✅ Architecture valid: {num_params:,} params, {model_size_mb:.2f}MB")
        
        # Train model
        print("🚀 Training model...")
        trained_model = train_nas_model(model, X_train_aug, y_train_aug, X_val, y_val)
        
        # Evaluate model
        print("📊 Evaluating model...")
        metrics = evaluate_nas_model(trained_model, X_test, y_test)
        
        # Measure inference time
        print("⏱️  Measuring inference time...")
        latency_ms = measure_inference_time(trained_model, X_test[:1])
        
        # Calculate combined score
        score = nas_objective(
            accuracy=metrics['exact_match_accuracy'],
            num_params=metrics['num_parameters'],
            latency_ms=latency_ms
        )
        
        # Report metrics
        print(f"📈 Results for Trial {trial.number}:")
        print(f"   Exact Match Accuracy: {metrics['exact_match_accuracy']:.3f}")
        print(f"   Average F1-Score: {metrics['average_f1_score']:.3f}")
        print(f"   Parameters: {metrics['num_parameters']:,}")
        print(f"   Model Size: {metrics['model_size_mb']:.2f}MB")
        print(f"   Inference Time: {latency_ms:.2f}ms")
        print(f"   Combined Score: {score:.3f}")
        
        # Store metrics in trial
        trial.set_user_attr('exact_match_accuracy', metrics['exact_match_accuracy'])
        trial.set_user_attr('average_f1_score', metrics['average_f1_score'])
        trial.set_user_attr('average_auc', metrics['average_auc'])
        trial.set_user_attr('num_parameters', metrics['num_parameters'])
        trial.set_user_attr('model_size_mb', metrics['model_size_mb'])
        trial.set_user_attr('inference_time_ms', latency_ms)
        trial.set_user_attr('channel_metrics', metrics['channel_metrics'])
        
        return score
        
    except optuna.TrialPruned:
        raise
    except Exception as e:
        print(f"❌ Trial {trial.number} failed: {e}")
        return 0.0

# ============================================================================
# NAS SEARCH EXECUTION
# ============================================================================

def run_nas_search(n_trials: int = None) -> optuna.Study:
    """
    Run Neural Architecture Search
    Args:
        n_trials: Number of trials (default from config)
    Returns:
        study: Completed Optuna study
    """
    if n_trials is None:
        n_trials = NAS_PARAMS['n_trials']
    
    print("🚀 Starting Neural Architecture Search for Spectrum Sensing")
    print("="*70)
    print(f"🎯 Target Parameters: < {SEARCH_CONSTRAINTS['max_parameters']:,}")
    print(f"🎯 Target Accuracy: > {SEARCH_CONSTRAINTS['target_accuracy']*100:.1f}%")
    print(f"🎯 Target Latency: < {SEARCH_CONSTRAINTS['target_latency']}ms")
    print(f"🔍 Search Trials: {n_trials}")
    print(f"⚡ Optimization Weights: {OPTIMIZATION_WEIGHTS}")
    print("="*70)
    
    # Create Optuna study
    study = optuna.create_study(
        direction='maximize',
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=5,
            n_warmup_steps=10,
            interval_steps=1
        )
    )
    
    # Run optimization
    start_time = time.time()
    
    try:
        study.optimize(
            objective,
            n_trials=n_trials,
            timeout=NAS_PARAMS.get('timeout', None),
            n_jobs=NAS_PARAMS['n_jobs'],
            show_progress_bar=True
        )
    except KeyboardInterrupt:
        print("\n⚠️  Search interrupted by user")
    except Exception as e:
        print(f"❌ Search failed: {e}")
        raise
    
    end_time = time.time()
    search_duration = end_time - start_time
    
    print(f"\n🎉 NAS Search completed!")
    print(f"⏱️  Total time: {search_duration/3600:.2f} hours")
    print(f"🔍 Completed trials: {len(study.trials)}")
    
    return study

# ============================================================================
# RESULTS ANALYSIS
# ============================================================================

def analyze_nas_results(study: optuna.Study) -> Dict[str, Any]:
    """
    Analyze NAS search results
    Args:
        study: Completed Optuna study
    Returns:
        analysis: Comprehensive results analysis
    """
    print("\n📊 Analyzing NAS search results...")
    
    if len(study.trials) == 0:
        print("❌ No completed trials found")
        return {}
    
    # Best trial
    best_trial = study.best_trial
    best_params = best_trial.params
    best_value = best_trial.value
    
    print(f"\n🏆 Best Architecture Found:")
    print(f"   Combined Score: {best_value:.3f}")
    print(f"   Exact Match Accuracy: {best_trial.user_attrs.get('exact_match_accuracy', 'N/A'):.3f}")
    print(f"   Parameters: {best_trial.user_attrs.get('num_parameters', 'N/A'):,}")
    print(f"   Model Size: {best_trial.user_attrs.get('model_size_mb', 'N/A'):.2f}MB")
    print(f"   Inference Time: {best_trial.user_attrs.get('inference_time_ms', 'N/A'):.2f}ms")
    
    # Architecture summary
    print(f"\n🏗️  Best Architecture:")
    print(f"   Conv Layers: {best_params.get('num_conv_layers', 'N/A')}")
    print(f"   Dense Layers: {best_params.get('num_dense_layers', 'N/A')}")
    print(f"   Global Pooling: {best_params.get('global_pool', 'N/A')}")
    
    # Performance distribution
    completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if completed_trials:
        accuracies = [t.user_attrs.get('exact_match_accuracy', 0) for t in completed_trials]
        parameters = [t.user_attrs.get('num_parameters', 0) for t in completed_trials]
        
        print(f"\n📈 Performance Distribution:")
        print(f"   Accuracy - Min: {min(accuracies):.3f}, Max: {max(accuracies):.3f}, Mean: {np.mean(accuracies):.3f}")
        print(f"   Parameters - Min: {min(parameters):,}, Max: {max(parameters):,}, Mean: {np.mean(parameters):,.0f}")
    
    # Compare with baseline CNN
    best_metrics = {
        'exact_match_accuracy': best_trial.user_attrs.get('exact_match_accuracy', 0),
        'average_f1_score': best_trial.user_attrs.get('average_f1_score', 0),
        'average_auc': best_trial.user_attrs.get('average_auc', 0),
        'num_parameters': best_trial.user_attrs.get('num_parameters', 0),
        'model_size_mb': best_trial.user_attrs.get('model_size_mb', 0),
        'inference_time_ms': best_trial.user_attrs.get('inference_time_ms', 0)
    }
    
    comparison = compare_with_baseline_cnn(best_metrics)
    
    # Compile analysis
    analysis = {
        'search_summary': {
            'total_trials': len(study.trials),
            'completed_trials': len(completed_trials),
            'best_score': best_value,
            'best_params': best_params,
            'best_metrics': best_metrics
        },
        'performance_distribution': {
            'accuracies': accuracies if completed_trials else [],
            'parameters': parameters if completed_trials else [],
            'avg_accuracy': np.mean(accuracies) if completed_trials else 0,
            'avg_parameters': np.mean(parameters) if completed_trials else 0
        },
        'baseline_comparison': comparison,
        'search_config': {
            'constraints': SEARCH_CONSTRAINTS,
            'optimization_weights': OPTIMIZATION_WEIGHTS,
            'nas_params': NAS_PARAMS
        }
    }
    
    return analysis

def save_nas_study(study: optuna.Study, file_path: str):
    """Save Optuna study to file"""
    # Save study
    study_file = file_path.replace('.json', '_study.pkl')
    with open(study_file, 'wb') as f:
        pickle.dump(study, f)
    
    # Save results as JSON
    analysis = analyze_nas_results(study)
    save_nas_results(analysis, file_path)
    
    print(f"💾 NAS study saved to {study_file}")
    print(f"💾 NAS results saved to {file_path}")

# ============================================================================
# BEST MODEL TRAINING
# ============================================================================

def train_best_nas_model(study: optuna.Study, final_epochs: int = 100) -> keras.Model:
    """
    Train the best NAS architecture for final evaluation
    Args:
        study: Completed Optuna study
        final_epochs: Number of epochs for final training
    Returns:
        final_model: Best trained model
    """
    print("\n🏆 Training best NAS architecture for final evaluation...")
    
    best_trial = study.best_trial
    
    # Load datasets
    datasets = load_datasets()
    X_train, y_train = datasets['train']
    X_val, y_val = datasets['val']
    X_test, y_test = datasets['test']
    
    # Apply data augmentation
    X_train_aug, y_train_aug = apply_data_augmentation(X_train, y_train)
    
    # Build best architecture
    model = build_nas_architecture(best_trial)
    
    print(f"✅ Built best architecture: {count_model_parameters(model):,} parameters")
    
    # Train with more epochs
    print(f"🚀 Training for {final_epochs} epochs...")
    final_model = train_nas_model(model, X_train_aug, y_train_aug, X_val, y_val, epochs=final_epochs)
    
    # Final evaluation
    print("📊 Final evaluation...")
    final_metrics = evaluate_nas_model(final_model, X_test, y_test)
    
    print(f"\n🎉 Final Model Results:")
    print(f"   Exact Match Accuracy: {final_metrics['exact_match_accuracy']:.3f}")
    print(f"   Average F1-Score: {final_metrics['average_f1_score']:.3f}")
    print(f"   Average AUC: {final_metrics['average_auc']:.3f}")
    print(f"   Parameters: {final_metrics['num_parameters']:,}")
    print(f"   Model Size: {final_metrics['model_size_mb']:.2f}MB")
    
    # Save final model
    final_model.save(FINAL_MODEL_PATH)
    print(f"💾 Final model saved to {FINAL_MODEL_PATH}")
    
    return final_model

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main NAS execution function"""
    print("🚀 Neural Architecture Search for Spectrum Sensing")
    print("Target: <100K parameters, >98% accuracy, <0.5ms latency")
    print("Based on successful CNN implementation (96.6% accuracy, 184K params)")
    print("="*80)
    
    # Verify configuration
    verify_config()
    
    # Verify NAS implementation
    if not verify_nas_implementation():
        print("❌ NAS implementation verification failed")
        return
    
    # Run NAS search
    study = run_nas_search()
    
    # Analyze results
    analysis = analyze_nas_results(study)
    
    # Save results
    save_nas_study(study, NAS_RESULTS_PATH)
    save_nas_results(analysis, COMPARISON_PATH)
    
    # Train best model
    final_model = train_best_nas_model(study)
    
    print("\n🎉 Neural Architecture Search completed successfully!")
    print("📁 Results saved in:")
    print(f"   - {NAS_RESULTS_PATH}")
    print(f"   - {COMPARISON_PATH}")
    print(f"   - {FINAL_MODEL_PATH}")

if __name__ == "__main__":
    main()

