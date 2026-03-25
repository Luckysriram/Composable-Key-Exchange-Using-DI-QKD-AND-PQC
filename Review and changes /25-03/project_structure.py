"""
DI-QKD Simulator Project Structure & File Guide
Run this to see a nice visualization of all project files
"""

import os
from pathlib import Path


def print_tree(directory, prefix="", is_last=True, max_depth=None, current_depth=0):
    """Print directory tree structure"""
    
    if max_depth is not None and current_depth >= max_depth:
        return
    
    if not os.path.exists(directory):
        return
    
    try:
        items = sorted(os.listdir(directory))
    except PermissionError:
        return
    
    # Filter out unwanted directories
    skip_dirs = {'.git', '__pycache__', '.pytest_cache', 'venv', '.venv', 'node_modules', '.idea'}
    items = [item for item in items if item not in skip_dirs]
    
    # Skip hidden files (except special ones)
    items = [item for item in items if not item.startswith('.') or item in ['.env']]
    
    for i, item in enumerate(items):
        path = os.path.join(directory, item)
        is_last_item = i == len(items) - 1
        
        current_prefix = "└── " if is_last_item else "├── "
        print(f"{prefix}{current_prefix}{item}")
        
        if os.path.isdir(path) and item not in skip_dirs:
            next_prefix = prefix + ("    " if is_last_item else "│   ")
            print_tree(path, next_prefix, is_last_item, max_depth, current_depth + 1)


def print_file_info():
    """Print information about all important files"""
    
    files_info = {
        "Backend Core": {
            "backend/quantum_simulator.py": "Quantum state simulation, Bell states, measurements",
            "backend/bb84.py": "BB84 quantum key distribution protocol",
            "backend/chsh.py": "CHSH Bell test for device-independent verification",
            "backend/diqkd_simulator.py": "Main DI-QKD orchestrator combining BB84 + CHSH",
            "backend/api.py": "Flask REST API server with 13+ endpoints",
            "backend/__init__.py": "Backend package initialization",
        },
        "Frontend": {
            "frontend/index.html": "Web UI with controls, results display, and visualization",
            "frontend/app.js": "JavaScript frontend application with API integration",
        },
        "Testing & Demo": {
            "test_simulator.py": "Comprehensive test suite with 20+ test cases",
            "demo.py": "5 demonstration scenarios showing all features",
            "setup.py": "Automated setup and initialization script",
        },
        "Configuration": {
            "requirements.txt": "Python package dependencies (Flask, NumPy, pytest)",
        },
        "Documentation": {
            "README.md": "Complete project documentation and user guide",
            "QUICKSTART.md": "5-minute quick start guide and examples",
            "ARCHITECTURE.md": "System architecture and design documentation",
            "IMPLEMENTATION_SUMMARY.md": "Project completion status and statistics",
        }
    }
    
    for category, files in files_info.items():
        print(f"\n📁 {category}")
        print("=" * 70)
        for filepath, description in files.items():
            print(f"  📄 {filepath:<40} → {description}")


def print_quick_commands():
    """Print quick command reference"""
    
    commands = {
        "Installation": [
            "pip install -r requirements.txt",
        ],
        "Running the Application": [
            "# Method 1: Automated setup (Recommended)",
            "python setup.py",
            "",
            "# Method 2: Backend + Frontend separately",
            "python -m backend.api          # Terminal 1 - Start backend",
            "# Terminal 2 - Open frontend/index.html in browser",
            "",
            "# Method 3: Command-line demo",
            "python demo.py",
        ],
        "Testing": [
            "# Run all tests",
            "python -m pytest test_simulator.py -v",
            "",
            "# Run specific test class",
            "python -m pytest test_simulator.py::TestBB84 -v",
        ],
        "Development": [
            "# Check code",
            "python -c 'import backend; print(\"✓ Backend OK\")'",
            "",
            "# Run quick simulation",
            "python -c 'from backend.diqkd_simulator import DIQKDSimulator; sim = DIQKDSimulator(); sim.run_full_simulation()'",
        ]
    }
    
    print("\n\n🚀 QUICK COMMANDS")
    print("=" * 70)
    
    for category, command_list in commands.items():
        print(f"\n{category}:")
        print("-" * 70)
        for cmd in command_list:
            if cmd.startswith("#"):
                print(f"  {cmd}")
            elif cmd == "":
                print()
            else:
                print(f"  $ {cmd}")


def main():
    """Main function"""
    
    print("\n")
    print("╔" + "=" * 68 + "╗")
    print("║" + " " * 12 + "DI-QKD Simulator - Project File Structure" + " " * 15 + "║")
    print("╚" + "=" * 68 + "╝")
    
    # Print file tree
    print("\n📂 PROJECT STRUCTURE")
    print("=" * 70)
    print("code-2/")
    print_tree(".", "", True, max_depth=2)
    
    # Print file descriptions
    print("\n\n📚 FILE DESCRIPTIONS")
    print_file_info()
    
    # Print quick commands
    print_quick_commands()
    
    # Print statistics
    print("\n\n📊 PROJECT STATISTICS")
    print("=" * 70)
    
    # Count lines of code
    total_lines = 0
    total_files = 0
    
    for root, dirs, files in os.walk("."):
        # Skip certain directories
        dirs[:] = [d for d in dirs if d not in {
            '__pycache__', '.git', 'venv', '.venv', 'node_modules'
        }]
        
        for file in files:
            if file.endswith(('.py', '.html', '.js', '.md', '.txt')):
                filepath = os.path.join(root, file)
                try:
                    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                        lines = len(f.readlines())
                        total_lines += lines
                        total_files += 1
                        if file.endswith('.py'):
                            print(f"  {filepath:<50} {lines:>4} lines")
                except:
                    pass
    
    print(f"\nTotal Source Files: {total_files}")
    print(f"Total Lines of Code: {total_lines:,}")
    
    # Print next steps
    print("\n\n✅ QUICK START CHECKLIST")
    print("=" * 70)
    steps = [
        "Install dependencies: pip install -r requirements.txt",
        "Run demo to verify: python demo.py",
        "Start backend: python -m backend.api",
        "Open frontend: frontend/index.html in browser",
        "Run tests: python -m pytest test_simulator.py -v",
    ]
    
    for i, step in enumerate(steps, 1):
        print(f"  {i}. {step}")
    
    print("\n\n📖 DOCUMENTATION")
    print("=" * 70)
    docs = {
        "README.md": "Complete overview and user guide",
        "QUICKSTART.md": "5-minute quick start guide",
        "ARCHITECTURE.md": "System design and technical details",
        "IMPLEMENTATION_SUMMARY.md": "Project completion status",
    }
    
    for doc_file, description in docs.items():
        status = "✅" if os.path.exists(doc_file) else "❌"
        print(f"  {status} {doc_file:<30} - {description}")
    
    print("\n\n🎉 READY TO USE!")
    print("=" * 70)
    print("""
  The DI-QKD Simulator is fully implemented and ready:
  
  ✓ Complete backend with BB84 + CHSH protocols
  ✓ Interactive web-based frontend
  ✓ Comprehensive test suite
  ✓ Working demonstrations
  ✓ Full documentation
  
  Choose your path:
  1. Quick demo:  python demo.py
  2. Web UI:      python setup.py
  3. Tests:       python -m pytest test_simulator.py -v
  4. Development: See documentation files
""")
    print("=" * 70 + "\n")


if __name__ == '__main__':
    main()
