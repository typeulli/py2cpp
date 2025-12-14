import os
import shutil
import subprocess
import sys

def remove_directory(path: str):
    """Remove a directory if it exists."""
    if os.path.exists(path):
        shutil.rmtree(path)
        print(f"Removed {path}")

def run_command(cmd: str, description: str):
    """Run a command and handle errors."""
    print(f"\n[*] {description}")
    result = subprocess.run(cmd, shell=True)
    if result.returncode != 0:
        print(f"Error: {description} failed.")
        sys.exit(1)

def main():
    print("[1] Cleaning previous build artifacts...")
    remove_directory("build")
    remove_directory("dist")
    remove_directory("py2cpp.egg-info")
    
    print("\n[2] Building source and wheel distributions...")
    run_command("python setup.py sdist bdist_wheel", "Building distributions")
    
    print("\n[3] Uploading package via twine...")
    run_command("python -m twine upload dist/*", "Uploading to PyPI")
    
    print("\n[4] Cleaning up after upload...")
    remove_directory("build")
    remove_directory("dist")
    remove_directory("py2cpp.egg-info")
    
    print("\nDone.")

if __name__ == "__main__":
    main()