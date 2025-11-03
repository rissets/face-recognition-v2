#!/usr/bin/env python3
"""
Script untuk mengecek dan menginstall dependencies untuk DeepFace anti-spoofing
"""

import subprocess
import sys
import os

def run_command(command):
    """Run shell command and return output"""
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        return result.returncode == 0, result.stdout, result.stderr
    except Exception as e:
        return False, "", str(e)

def check_python_version():
    """Check Python version"""
    print("=== PYTHON VERSION CHECK ===")
    print(f"Python version: {sys.version}")
    
    version_info = sys.version_info
    if version_info.major < 3 or (version_info.major == 3 and version_info.minor < 7):
        print("⚠️  Warning: Python 3.7+ recommended for DeepFace")
    else:
        print("✓ Python version is compatible")

def check_and_install_packages():
    """Check and install required packages"""
    print("\n=== CHECKING REQUIRED PACKAGES ===")
    
    required_packages = [
        'opencv-python',
        'deepface',
        'tensorflow',
        'numpy',
        'pillow',
        'pandas',
        'gdown',
        'tqdm',
        'requests'
    ]
    
    optional_packages = [
        'mtcnn',
        'retina-face',
        'mediapipe',
        'dlib'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
            print(f"✓ {package} - installed")
        except ImportError:
            print(f"✗ {package} - missing")
            missing_packages.append(package)
    
    print("\n=== OPTIONAL PACKAGES ===")
    for package in optional_packages:
        try:
            if package == 'retina-face':
                __import__('retinaface')
            else:
                __import__(package.replace('-', '_'))
            print(f"✓ {package} - installed")
        except ImportError:
            print(f"- {package} - not installed (optional)")
    
    if missing_packages:
        print(f"\n=== INSTALLING MISSING PACKAGES ===")
        print("Installing required packages...")
        
        for package in missing_packages:
            print(f"Installing {package}...")
            success, stdout, stderr = run_command(f"{sys.executable} -m pip install {package}")
            if success:
                print(f"✓ {package} installed successfully")
            else:
                print(f"✗ Failed to install {package}")
                print(f"Error: {stderr}")
    else:
        print("\n✓ All required packages are installed")

def check_deepface_models():
    """Check if DeepFace models are downloaded"""
    print("\n=== CHECKING DEEPFACE MODELS ===")
    
    home_dir = os.path.expanduser("~")
    deepface_dir = os.path.join(home_dir, ".deepface")
    weights_dir = os.path.join(deepface_dir, "weights")
    
    print(f"DeepFace directory: {deepface_dir}")
    
    if os.path.exists(deepface_dir):
        print("✓ DeepFace directory exists")
        
        if os.path.exists(weights_dir):
            print("✓ Weights directory exists")
            
            # List downloaded models
            try:
                weights_files = os.listdir(weights_dir)
                if weights_files:
                    print("Downloaded models:")
                    for file in weights_files:
                        file_path = os.path.join(weights_dir, file)
                        size = os.path.getsize(file_path) / (1024*1024)  # MB
                        print(f"  - {file} ({size:.1f} MB)")
                else:
                    print("- No model weights found")
            except Exception as e:
                print(f"Error listing weights: {e}")
        else:
            print("- Weights directory does not exist")
    else:
        print("- DeepFace directory does not exist")

def test_basic_import():
    """Test basic imports"""
    print("\n=== TESTING BASIC IMPORTS ===")
    
    test_imports = [
        ('cv2', 'OpenCV'),
        ('numpy', 'NumPy'),
        ('deepface', 'DeepFace'),
        ('tensorflow', 'TensorFlow'),
    ]
    
    for module, name in test_imports:
        try:
            imported = __import__(module)
            version = getattr(imported, '__version__', 'Unknown')
            print(f"✓ {name} - version {version}")
        except ImportError as e:
            print(f"✗ {name} - import failed: {e}")
        except Exception as e:
            print(f"⚠️  {name} - import warning: {e}")

def create_requirements_file():
    """Create requirements.txt file"""
    print("\n=== CREATING REQUIREMENTS FILE ===")
    
    requirements = """# DeepFace Anti-Spoofing Requirements
opencv-python>=4.5.0
deepface>=0.0.75
tensorflow>=2.4.0
numpy>=1.19.0
pillow>=8.0.0
pandas>=1.1.0
gdown>=3.12.0
tqdm>=4.60.0
requests>=2.25.0

# Optional packages for better detection
mtcnn>=0.1.1
retina-face>=0.0.12
mediapipe>=0.8.0
# dlib>=19.22.0  # Uncomment if needed, but can be hard to install
"""
    
    with open("requirements_antispoof.txt", "w") as f:
        f.write(requirements)
    
    print("✓ Created requirements_antispoof.txt")
    print("\nTo install all requirements, run:")
    print("pip install -r requirements_antispoof.txt")

def main():
    print("=== DEEPFACE ANTI-SPOOFING SETUP CHECKER ===\n")
    
    check_python_version()
    check_and_install_packages()
    test_basic_import()
    check_deepface_models()
    create_requirements_file()
    
    print("\n=== SETUP COMPLETE ===")
    print("\nNext steps:")
    print("1. If any packages failed to install, try installing them manually")
    print("2. Run 'python debug_deepface.py' to test DeepFace functionality")
    print("3. Run 'python antispoof_enhanced.py' for the enhanced anti-spoofing detector")
    print("\nNote: The first time you run DeepFace, it may take time to download models.")

if __name__ == "__main__":
    main()