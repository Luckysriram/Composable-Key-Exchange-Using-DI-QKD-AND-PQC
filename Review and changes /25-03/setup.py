"""
Quick setup and initialization script for DI-QKD Simulator
Run this script to automatically set up the environment and start the application
"""

import os
import sys
import subprocess
import webbrowser
from pathlib import Path
import time


def check_python_version():
    """Check if Python version is compatible"""
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher required")
        sys.exit(1)
    print(f"✅ Python {sys.version.split()[0]} detected")


def install_dependencies():
    """Install required packages"""
    print("\n📦 Installing dependencies...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt", "-q"])
        print("✅ Dependencies installed")
    except subprocess.CalledProcessError:
        print("❌ Failed to install dependencies")
        sys.exit(1)


def create_directories():
    """Ensure all necessary directories exist"""
    dirs = ['backend', 'frontend', 'results']
    for dir_name in dirs:
        Path(dir_name).mkdir(exist_ok=True)
    print(f"✅ Directories verified: {', '.join(dirs)}")


def run_demo():
    """Ask user if they want to run demo"""
    print("\n" + "="*60)
    print("Would you like to run a demonstration? (y/n)")
    response = input(">>> ").strip().lower()
    
    if response == 'y':
        print("\n🚀 Running demonstration suite...")
        subprocess.run([sys.executable, "demo.py"])


def start_backend():
    """Start Flask backend server"""
    print("\n" + "="*60)
    print("🚀 Starting DI-QKD Simulator Backend...")
    print("="*60)
    print("\nFlask server starting on http://localhost:5000")
    print("Press Ctrl+C to stop the server\n")
    
    try:
        subprocess.run([sys.executable, "-m", "backend.api"])
    except KeyboardInterrupt:
        print("\n\n📛 Server stopped")
        sys.exit(0)


def main():
    """Main setup and launch function"""
    print("\n")
    print("╔" + "="*58 + "╗")
    print("║" + " "*12 + "DI-QKD Simulator - Setup & Launch" + " "*14 + "║")
    print("╚" + "="*58 + "╝\n")
    
    # Check Python version
    check_python_version()
    
    # Install dependencies
    install_dependencies()
    
    # Create directories
    create_directories()
    
    # Ask about running demo
    run_demo()
    
    # Ask about starting backend
    print("\n" + "="*60)
    print("Ready to launch the application!")
    print("="*60)
    print("\nOptions:")
    print("1. Start Backend (Flask API) - required for web UI")
    print("2. Open Web UI (requires backend running)")
    print("3. Exit")
    
    choice = input("\nEnter choice (1-3): ").strip()
    
    if choice == '1':
        start_backend()
    elif choice == '2':
        print("\n⚠️  Make sure backend is running!")
        print("Opening web interface...")
        
        # Determine the correct path to index.html
        frontend_path = Path('frontend/index.html').resolve()
        
        if frontend_path.exists():
            # Open in default browser
            webbrowser.open(f'file://{frontend_path}')
            print(f"✅ Opening {frontend_path}")
            print("\nTo start the backend server, run:")
            print("  python -m backend.api")
        else:
            print("❌ frontend/index.html not found")
    else:
        print("\nℹ️  To start the application later:")
        print("  1. Start backend: python -m backend.api")
        print("  2. Open: frontend/index.html in your browser")
        print("\nOr run the demo: python demo.py")
        sys.exit(0)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 Setup cancelled")
        sys.exit(0)
