"""
Google Colab execution helper script for RP-BestA.
This script sets up the Colab environment, installs dependencies, clones the private repository,
and executes the evaluation pipeline.

To run this in Google Colab:
1. Open a new Google Colab notebook.
2. Create a cell and run:
   !wget -O run_colab.py https://raw.githubusercontent.com/adarsh912/RP-BestA/main/run_colab.py
   !python run_colab.py
"""

import os
import sys
import subprocess

def check_colab():
    try:
        import google.colab
        return True
    except ImportError:
        return False

def install_dependencies():
    print("Installing required packages on Google Colab...")
    packages = [
        "numpy>=1.20.0",
        "scipy>=1.7.0",
        "pandas>=1.3.0",
        "scikit-learn>=1.0.0",
        "xgboost>=1.5.0",
        "lightgbm>=3.3.0",
        "catboost>=1.0.0",
        "fastdtw>=0.3.4",
        "ruptures>=1.1.5",
        "matplotlib>=3.4.0",
        "seaborn>=0.11.0",
        "tabulate>=0.8.9",
        "aeon>=0.7.0"
    ]
    subprocess.check_call([sys.executable, "-m", "pip", "install"] + packages)
    print("Dependencies successfully installed.")

def setup_private_repo():
    print("\nSetting up private repository access...")
    print("Since the repository is private, you will need a GitHub Personal Access Token (PAT).")
    print("You can generate one at: https://github.com/settings/tokens")
    
    # Prompt the user for GitHub Username and Token in Colab
    try:
        from google.colab import userdata
        # Try getting from Colab secrets first
        username = userdata.get('GITHUB_USERNAME')
        token = userdata.get('GITHUB_TOKEN')
        print("Using GitHub credentials from Colab Secrets.")
    except Exception:
        # Fall back to interactive input
        import getpass
        username = input("Enter your GitHub Username (default: adarsh912): ").strip() or "adarsh912"
        token = getpass.getpass("Enter your GitHub Personal Access Token (PAT): ").strip()

    if not token:
        print("Error: GitHub Token is required to clone the private repository.")
        sys.exit(1)

    repo_url = f"https://{username}:{token}@github.com/adarsh912/RP-BestA.git"
    repo_dir = "RP-BestA"

    if os.path.exists(repo_dir):
        print(f"Directory {repo_dir} already exists. Pulling latest changes...")
        subprocess.check_call(["git", "-C", repo_dir, "pull"])
    else:
        print(f"Cloning private repository into {repo_dir}...")
        subprocess.check_call(["git", "clone", repo_url, repo_dir])

    # Add cloned directory to python path
    sys.path.append(os.path.abspath(repo_dir))
    os.chdir(repo_dir)
    print(f"Working directory changed to {os.getcwd()}")

def run_sanity_check():
    print("\nRunning nested CV sanity check on GunPoint dataset...")
    # Import modules from the cloned repository
    try:
        from src.evaluation.tuning import run_nested_cv
        # Run a quick 2-fold nested CV to verify compilation and execution
        run_nested_cv("GunPoint", n_outer_folds=2, n_inner_folds=2)
        print("\nSanity check passed successfully!")
    except Exception as e:
        print(f"\nSanity check failed: {e}")

def main():
    if not check_colab():
        print("This script is designed to run inside Google Colab.")
        choice = input("Are you sure you want to run this locally? (y/n): ").lower()
        if choice != 'y':
            sys.exit(0)

    # 1. Install packages
    install_dependencies()

    # 2. Clone repo and cd
    setup_private_repo()

    # 3. Verify execution
    run_sanity_check()

if __name__ == "__main__":
    main()
