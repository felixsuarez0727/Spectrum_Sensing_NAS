"""
Setup script for Neural Architecture Search (NAS) Spectrum Sensing project
Automated installation and verification of all dependencies
"""

import os
import sys
import subprocess
import importlib
from typing import List, Tuple

def run_command(command: str, description: str) -> Tuple[bool, str]:
    """Run a command and return success status and output"""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(
            command, 
            shell=True, 
            capture_output=True, 
            text=True, 
            check=True
        )
        print(f"✅ {description} completed successfully")
        return True, result.stdout
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed: {e.stderr}")
        return False, e.stderr

def check_python_version():
    """Check if Python version is compatible"""
    print("🐍 Checking Python version...")
    
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print(f"❌ Python 3.8+ required, found {version.major}.{version.minor}")
        return False
    
    print(f"✅ Python {version.major}.{version.minor}.{version.micro} is compatible")
    return True

def install_package(package: str, description: str = None) -> bool:
    """Install a Python package using pip"""
    if description is None:
        description = f"Installing {package}"
    
    success, output = run_command(f"pip install {package}", description)
    return success

def check_package(package: str, import_name: str = None) -> bool:
    """Check if a package is installed and importable"""
    if import_name is None:
        import_name = package
    
    try:
        importlib.import_module(import_name)
        return True
    except ImportError:
        return False

def verify_tensorflow():
    """Verify TensorFlow installation and configuration"""
    print("🔍 Verifying TensorFlow installation...")
    
    try:
        import tensorflow as tf
        print(f"✅ TensorFlow {tf.__version__} imported successfully")
        
        # Check for GPU availability
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            print(f"🚀 GPU acceleration available: {len(gpus)} GPU(s)")
            for i, gpu in enumerate(gpus):
                print(f"   GPU {i}: {gpu.name}")
        else:
            print("💻 Using CPU (GPU not available)")
        
        # Test basic functionality
        print("🧪 Testing TensorFlow functionality...")
        test_tensor = tf.constant([1, 2, 3, 4, 5])
        result = tf.reduce_sum(test_tensor)
        print(f"✅ TensorFlow test passed: sum([1,2,3,4,5]) = {result.numpy()}")
        
        return True
        
    except Exception as e:
        print(f"❌ TensorFlow verification failed: {e}")
        return False

def verify_optuna():
    """Verify Optuna installation and functionality"""
    print("🔍 Verifying Optuna installation...")
    
    try:
        import optuna
        print(f"✅ Optuna {optuna.__version__} imported successfully")
        
        # Test basic functionality
        print("🧪 Testing Optuna functionality...")
        
        def objective(trial):
            x = trial.suggest_float('x', -10, 10)
            return (x - 2) ** 2
        
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=3)
        
        print(f"✅ Optuna test passed: best value = {study.best_value:.3f}")
        return True
        
    except Exception as e:
        print(f"❌ Optuna verification failed: {e}")
        return False

def verify_dataset_files():
    """Verify that dataset files are present"""
    print("🔍 Verifying dataset files...")
    
    dataset_files = [
        'data/processed/sdr_wifi_train.h5',
        'data/processed/sdr_wifi_val.h5',
        'data/processed/sdr_wifi_test.h5'
    ]
    
    all_present = True
    for file_path in dataset_files:
        if os.path.exists(file_path):
            file_size = os.path.getsize(file_path) / (1024 * 1024)  # MB
            print(f"✅ {file_path} found ({file_size:.1f} MB)")
        else:
            print(f"❌ {file_path} not found")
            all_present = False
    
    return all_present

def create_directories():
    """Create necessary directories if they don't exist"""
    print("📁 Creating project directories...")
    
    directories = [
        'results',
        'models', 
        'logs',
        'logs/tensorboard'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"✅ Directory created: {directory}")

def run_configuration_test():
    """Run configuration and utility tests"""
    print("🧪 Running configuration tests...")
    
    try:
        # Test configuration loading
        sys.path.append('config')
        from nas_config import *
        print("✅ Configuration loaded successfully")
        
        # Test utility functions
        sys.path.append('code')
        from nas_utils import verify_nas_implementation
        if verify_nas_implementation():
            print("✅ NAS utilities verified successfully")
            return True
        else:
            print("❌ NAS utilities verification failed")
            return False
            
    except Exception as e:
        print(f"❌ Configuration test failed: {e}")
        return False

def main():
    """Main setup function"""
    print("🚀 Neural Architecture Search (NAS) Setup for Spectrum Sensing")
    print("="*70)
    
    # Check Python version
    if not check_python_version():
        print("❌ Setup failed: Incompatible Python version")
        return False
    
    # Install core packages
    print("\n📦 Installing core dependencies...")
    
    packages_to_install = [
        ("tensorflow-macos==2.16.2", "TensorFlow for macOS ARM"),
        ("optuna>=3.0.0", "Neural Architecture Search framework"),
        ("numpy>=1.21.0", "Numerical computing"),
        ("scipy>=1.7.0", "Scientific computing"),
        ("scikit-learn>=1.0.0", "Machine learning utilities"),
        ("pandas>=1.3.0", "Data manipulation"),
        ("h5py>=3.1.0", "HDF5 file support"),
        ("matplotlib>=3.5.0", "Plotting and visualization"),
        ("seaborn>=0.11.0", "Statistical visualization"),
        ("tqdm>=4.62.0", "Progress bars"),
        ("jupyter>=1.0.0", "Jupyter notebook support")
    ]
    
    failed_installations = []
    for package, description in packages_to_install:
        if not install_package(package, description):
            failed_installations.append(package)
    
    if failed_installations:
        print(f"\n❌ Failed to install: {', '.join(failed_installations)}")
        print("Please install these packages manually:")
        for package in failed_installations:
            print(f"   pip install {package}")
        return False
    
    # Verify installations
    print("\n🔍 Verifying installations...")
    
    if not verify_tensorflow():
        print("❌ TensorFlow verification failed")
        return False
    
    if not verify_optuna():
        print("❌ Optuna verification failed")
        return False
    
    # Verify dataset files
    if not verify_dataset_files():
        print("❌ Dataset files not found")
        print("Please ensure dataset files are present in data/processed/")
        return False
    
    # Create directories
    create_directories()
    
    # Run configuration tests
    if not run_configuration_test():
        print("❌ Configuration test failed")
        return False
    
    # Final verification
    print("\n🎉 Setup completed successfully!")
    print("\n📋 Installation Summary:")
    print("✅ Python environment verified")
    print("✅ All dependencies installed")
    print("✅ TensorFlow configured")
    print("✅ Optuna functional")
    print("✅ Dataset files present")
    print("✅ Project structure created")
    
    print("\n🚀 Ready to run Neural Architecture Search!")
    print("\nNext steps:")
    print("1. Run NAS search: python code/nas_search.py")
    print("2. Evaluate results: python code/nas_evaluate.py")
    print("3. View results in results/ directory")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

