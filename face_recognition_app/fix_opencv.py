#!/usr/bin/env python3
"""
Script to fix OpenCV installation issues in Docker containers.
This script uninstalls conflicting opencv packages and installs opencv-python-headless.
"""
import subprocess
import sys
import pkg_resources

def run_command(command):
    """Run a shell command and return the result."""
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        return result.returncode == 0, result.stdout, result.stderr
    except Exception as e:
        return False, "", str(e)

def check_opencv_packages():
    """Check which OpenCV packages are installed."""
    opencv_packages = []
    for pkg in pkg_resources.working_set:
        if 'opencv' in pkg.project_name.lower():
            opencv_packages.append(pkg.project_name)
    return opencv_packages

def main():
    print("Checking current OpenCV installation...")
    
    # Check current packages
    opencv_packages = check_opencv_packages()
    print(f"Found OpenCV packages: {opencv_packages}")
    
    # Uninstall all opencv packages
    for package in opencv_packages:
        print(f"Uninstalling {package}...")
        success, stdout, stderr = run_command(f"pip uninstall -y {package}")
        if success:
            print(f"Successfully uninstalled {package}")
        else:
            print(f"Failed to uninstall {package}: {stderr}")
    
    # Install opencv-python-headless
    print("Installing opencv-python-headless...")
    success, stdout, stderr = run_command("pip install opencv-python-headless==4.12.0.88")
    if success:
        print("Successfully installed opencv-python-headless")
    else:
        print(f"Failed to install opencv-python-headless: {stderr}")
        sys.exit(1)
    
    # Test import
    print("Testing OpenCV import...")
    try:
        import cv2
        print(f"OpenCV version: {cv2.__version__}")
        print("OpenCV import successful!")
    except ImportError as e:
        print(f"OpenCV import failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()