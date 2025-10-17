#!/usr/bin/env python3
"""
Neural Architecture Search (NAS) Runner for Spectrum Sensing
Simple script to execute NAS optimization with different configurations
"""

import os
import sys
import argparse
import time
from typing import Optional

# Add project paths
sys.path.append(os.path.join(os.path.dirname(__file__), 'code'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'config'))

def run_quick_test():
    """Run a quick NAS test with minimal trials"""
    print("🧪 Running quick NAS test (5 trials)...")
    
    from nas_search import run_nas_search
    
    # Override config for quick test
    import nas_config
    nas_config.NAS_PARAMS['n_trials'] = 5
    nas_config.TRAINING_CONFIG['epochs'] = 10
    
    study = run_nas_search(n_trials=5)
    
    print(f"\n🎉 Quick test completed!")
    print(f"Best score: {study.best_value:.3f}")
    print(f"Best accuracy: {study.best_trial.user_attrs.get('exact_match_accuracy', 'N/A')}")
    
    return study

def run_full_search():
    """Run full NAS search with all trials"""
    print("🚀 Running full NAS search (100 trials)...")
    
    from nas_search import run_nas_search, analyze_nas_results, save_nas_study
    
    start_time = time.time()
    
    # Run full search
    study = run_nas_search(n_trials=100)
    
    # Analyze results
    analysis = analyze_nas_results(study)
    
    # Save results
    save_nas_study(study, 'results/nas_search_results.json')
    
    end_time = time.time()
    duration_hours = (end_time - start_time) / 3600
    
    print(f"\n🎉 Full NAS search completed!")
    print(f"Duration: {duration_hours:.2f} hours")
    print(f"Best score: {study.best_value:.3f}")
    print(f"Best accuracy: {study.best_trial.user_attrs.get('exact_match_accuracy', 'N/A')}")
    print(f"Best parameters: {study.best_trial.user_attrs.get('num_parameters', 'N/A'):,}")
    
    return study

def run_custom_search(n_trials: int, epochs: int):
    """Run custom NAS search with specified parameters"""
    print(f"🔧 Running custom NAS search ({n_trials} trials, {epochs} epochs)...")
    
    # Override configuration
    import nas_config
    nas_config.NAS_PARAMS['n_trials'] = n_trials
    nas_config.TRAINING_CONFIG['epochs'] = epochs
    
    from nas_search import run_nas_search
    study = run_nas_search(n_trials=n_trials)
    
    print(f"\n🎉 Custom search completed!")
    print(f"Best score: {study.best_value:.3f}")
    
    return study

def run_evaluation():
    """Run comprehensive evaluation of the best NAS model"""
    print("📊 Running NAS evaluation...")
    
    from nas_evaluate import main as evaluate_nas
    evaluate_nas()
    
    print("\n🎉 Evaluation completed!")
    print("Check results/ directory for detailed reports and visualizations")

def run_complete_pipeline():
    """Run complete NAS pipeline: search + evaluation"""
    print("🔄 Running complete NAS pipeline...")
    
    # Step 1: Run NAS search
    study = run_full_search()
    
    # Step 2: Train best model
    from nas_search import train_best_nas_model
    final_model = train_best_nas_model(study)
    
    # Step 3: Run evaluation
    run_evaluation()
    
    print("\n🎉 Complete pipeline finished!")
    print("All results saved in results/ directory")

def main():
    """Main function with command line interface"""
    parser = argparse.ArgumentParser(
        description="Neural Architecture Search for Spectrum Sensing",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_nas.py test                    # Quick test (5 trials)
  python run_nas.py search                  # Full search (100 trials)
  python run_nas.py search --trials 50      # Custom search (50 trials)
  python run_nas.py evaluate                # Evaluate best model
  python run_nas.py pipeline                # Complete pipeline
        """
    )
    
    parser.add_argument(
        'command',
        choices=['test', 'search', 'evaluate', 'pipeline'],
        help='Command to execute'
    )
    
    parser.add_argument(
        '--trials',
        type=int,
        default=100,
        help='Number of NAS trials (default: 100)'
    )
    
    parser.add_argument(
        '--epochs',
        type=int,
        default=50,
        help='Number of training epochs per trial (default: 50)'
    )
    
    args = parser.parse_args()
    
    print("🚀 Neural Architecture Search for Spectrum Sensing")
    print("Target: <100K parameters, >98% accuracy, <0.5ms latency")
    print("="*60)
    
    try:
        if args.command == 'test':
            run_quick_test()
            
        elif args.command == 'search':
            if args.trials == 100:
                run_full_search()
            else:
                run_custom_search(args.trials, args.epochs)
                
        elif args.command == 'evaluate':
            run_evaluation()
            
        elif args.command == 'pipeline':
            run_complete_pipeline()
            
    except KeyboardInterrupt:
        print("\n⚠️  Process interrupted by user")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        return 1
    
    print("\n✅ Command completed successfully!")
    return 0

if __name__ == "__main__":
    sys.exit(main())

