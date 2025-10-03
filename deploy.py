#!/usr/bin/env python3
"""
Deployment helper script for Prompt Optimizer
Helps with setup, testing, and deployment to various platforms
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path

def check_python_version():
    """Ensure Python 3.8+ is installed"""
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher is required")
        sys.exit(1)
    print(f"✅ Python {sys.version.split()[0]} detected")

def create_project_structure():
    """Create necessary directories and files"""
    directories = ['.streamlit']
    
    for dir_name in directories:
        Path(dir_name).mkdir(exist_ok=True)
        print(f"✅ Created directory: {dir_name}")

def install_dependencies():
    """Install required packages"""
    print("\n📦 Installing dependencies...")
    try:
        subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"], check=True)
        print("✅ Dependencies installed successfully")
    except subprocess.CalledProcessError:
        print("❌ Failed to install dependencies")
        print("Try: pip install -r requirements.txt")
        sys.exit(1)

def test_local():
    """Run the app locally for testing"""
    print("\n🚀 Starting local server...")
    print("Press Ctrl+C to stop")
    try:
        subprocess.run(["streamlit", "run", "app.py"])
    except KeyboardInterrupt:
        print("\n✅ Server stopped")

def prepare_huggingface():
    """Prepare files for Hugging Face deployment"""
    print("\n🤗 Preparing for Hugging Face Spaces...")
    
    # Create README for Hugging Face
    hf_readme = """---
title: Prompt Optimizer
emoji: 🎯
colorFrom: blue
colorTo: green
sdk: streamlit
sdk_version: 1.32.0
app_file: app.py
pinned: false
---

# Prompt Optimizer

Transform your prompts into precision-engineered instructions for AI assistants.

## Features
- Two-step optimization process
- Intelligent clarifying questions
- Session history tracking
- Free and open-source

Check out the [GitHub repository](https://github.com/yourusername/prompt-optimizer) for more details.
"""
    
    with open("README_HF.md", "w") as f:
        f.write(hf_readme)
    
    print("✅ Created README_HF.md for Hugging Face")
    print("\nNext steps:")
    print("1. Create a new Space at https://huggingface.co/spaces")
    print("2. Upload these files: app.py, requirements.txt, README_HF.md")
    print("3. Upload .streamlit/config.toml")
    print("4. Select 'Streamlit' as SDK")
    print("5. Deploy!")

def prepare_streamlit_cloud():
    """Prepare for Streamlit Cloud deployment"""
    print("\n☁️ Preparing for Streamlit Cloud...")
    
    # Check for git repository
    if not Path(".git").exists():
        print("⚠️  No git repository found. Initializing...")
        subprocess.run(["git", "init"])
        print("✅ Git repository initialized")
    
    # Create .gitignore if it doesn't exist
    gitignore_content = """# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
env/
venv/
ENV/
.venv

# Streamlit
.streamlit/secrets.toml
*.log

# IDE
.vscode/
.idea/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db

# Project specific
*.csv
*.json
!requirements.txt
!config.toml
"""
    
    with open(".gitignore", "w") as f:
        f.write(gitignore_content)
    
    print("✅ Created .gitignore")
    print("\nNext steps:")
    print("1. Commit all files: git add . && git commit -m 'Initial commit'")
    print("2. Push to GitHub: git push origin main")
    print("3. Go to https://share.streamlit.io")
    print("4. Connect your GitHub repo")
    print("5. Deploy!")

def choose_model_size():
    """Help user choose appropriate model size"""
    print("\n🤖 Model Selection Guide:")
    print("\n1. LITE (Recommended for free tier):")
    print("   - Model: google/flan-t5-base")
    print("   - Size: ~250MB")
    print("   - Speed: Fast")
    print("   - Quality: Good")
    
    print("\n2. STANDARD:")
    print("   - Model: microsoft/phi-2")
    print("   - Size: ~2.7GB")
    print("   - Speed: Moderate")
    print("   - Quality: Better")
    
    print("\n3. PREMIUM:")
    print("   - Model: mistralai/Mistral-7B-Instruct-v0.1")
    print("   - Size: ~14GB")
    print("   - Speed: Slow (needs GPU)")
    print("   - Quality: Best")
    
    choice = input("\nChoose model size (1/2/3) [1]: ").strip() or "1"
    
    if choice == "1":
        print("\n✅ Using LITE version (app_lite.py)")
        print("Run: streamlit run app_lite.py")
    elif choice == "2":
        print("\n✅ Using STANDARD version")
        print("Update line 68 in app.py to use 'microsoft/phi-2'")
    else:
        print("\n✅ Using PREMIUM version")
        print("Default configuration in app.py")

def main():
    print("🎯 Prompt Optimizer - Deployment Helper")
    print("=" * 40)
    
    check_python_version()
    
    while True:
        print("\n📋 Menu:")
        print("1. Setup project structure")
        print("2. Install dependencies")
        print("3. Test locally")
        print("4. Prepare for Hugging Face")
        print("5. Prepare for Streamlit Cloud")
        print("6. Choose model size")
        print("7. Run lite version")
        print("0. Exit")
        
        choice = input("\nSelect option: ").strip()
        
        if choice == "1":
            create_project_structure()
        elif choice == "2":
            install_dependencies()
        elif choice == "3":
            test_local()
        elif choice == "4":
            prepare_huggingface()
        elif choice == "5":
            prepare_streamlit_cloud()
        elif choice == "6":
            choose_model_size()
        elif choice == "7":
            print("\n🚀 Starting lite version...")
            subprocess.run(["streamlit", "run", "app_lite.py"])
        elif choice == "0":
            print("\n👋 Goodbye!")
            break
        else:
            print("❌ Invalid option")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 Deployment helper stopped")
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        sys.exit(1)
