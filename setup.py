#!/usr/bin/env python

"""
Setup Script for Course Environment
----------------------------------
This script sets up the required environment for the course,
including creating virtual environment and installing dependencies.

Usage:
    python setup.py [--full] [--env ENV_NAME]

Options:
    --full      Install full requirements (for complete course)
    --env       Specify environment name (default: venv)

Author: Dr. Jody-Ann S. Jones
GitHub: https://github.com/dasdatasensei
"""

import os
import sys
import argparse
import subprocess
import platform
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Setup script for course environment")
    parser.add_argument("--full", action="store_true", help="Install full requirements")
    parser.add_argument(
        "--env", default="venv", help="Environment name (default: venv)"
    )
    parser.add_argument(
        "--setup-git", action="store_true", help="Initialize git repository"
    )
    args = parser.parse_args()

    # Check Python version
    min_version = (3, 9)
    if sys.version_info < min_version:
        sys.exit(f"Python {min_version[0]}.{min_version[1]} or later is required.")

    # Create virtual environment
    print(f"Creating virtual environment: {args.env}")
    subprocess.run([sys.executable, "-m", "venv", args.env], check=True)

    # Determine activation script and pip paths
    if platform.system() == "Windows":
        activate_script = os.path.join(args.env, "Scripts", "activate")
        pip_path = os.path.join(args.env, "Scripts", "pip")
    else:
        activate_script = os.path.join(args.env, "bin", "activate")
        pip_path = os.path.join(args.env, "bin", "pip")

    print(f"To activate the environment, run:")
    if platform.system() == "Windows":
        print(f"    {args.env}\Scripts\activate")
    else:
        print(f"    source {args.env}/bin/activate")

    # Upgrade pip
    print("Upgrading pip, setuptools, and wheel")
    subprocess.run(
        [pip_path, "install", "--upgrade", "pip", "setuptools", "wheel"], check=True
    )

    # Install requirements
    req_file = "requirements-full.txt" if args.full else "requirements.txt"
    if os.path.exists(req_file):
        print(f"Installing dependencies from {req_file}")
        subprocess.run([pip_path, "install", "-r", req_file], check=True)
        print("Environment setup complete!")
    else:
        print(f"Error: {req_file} not found!")
        sys.exit(1)

    # Setup Git if requested
    if args.setup_git and not os.path.exists(".git"):
        print("Initializing Git repository...")
        subprocess.run(["git", "init"], check=True)
        print("Git repository initialized!")

        # Create initial .gitignore if it doesn't exist
        if not os.path.exists(".gitignore"):
            with open(".gitignore", "w") as f:
                f.write("# Python\n")
                f.write("__pycache__/\n")
                f.write("*.py[cod]\n")
                f.write("*$py.class\n\n")
                f.write("# Virtual Environment\n")
                f.write("venv/\n")
                f.write("env/\n\n")
                f.write("# Jupyter\n")
                f.write(".ipynb_checkpoints\n\n")
                f.write("# Models\n")
                f.write("models/*.gguf\n\n")
                f.write("# Environment variables\n")
                f.write(".env\n")
            print("Created basic .gitignore file")

        print("To make your first commit, run:")
        print("    git add .")
        print('    git commit -m "Initial course setup"')


if __name__ == "__main__":
    main()
