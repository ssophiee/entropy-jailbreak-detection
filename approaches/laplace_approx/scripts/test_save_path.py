#!/usr/bin/env python3
"""Simple test script to create a save path and write a test file.

Usage:
    python scripts/test_save_path.py --save_dir saved_models --model_name my_model

This will create (if needed) `saved_models/my_model` under the current working directory
(or absolute path if you pass an absolute `--save_dir`) and write `test.txt` with a timestamp.
"""

import argparse
import os
from datetime import datetime


def main():
    parser = argparse.ArgumentParser(description="Test save path writer")
    parser.add_argument("--save_dir", type=str, default="saved_models", help="Base directory to save into")
    parser.add_argument("--model_name", type=str, default=None, help="Subfolder name for this test (default: timestamp)")
    args = parser.parse_args()

    # Resolve and create path
    base_dir = os.path.abspath(args.save_dir)
    name = args.model_name or f"test_save_{int(datetime.now().timestamp())}"
    save_path = os.path.join(base_dir, name)
    os.makedirs(save_path, exist_ok=True)

    # File to write
    test_file = os.path.join(save_path, "test.txt")
    now = datetime.now().isoformat()
    content = f"Test file created at {now}\nCWD: {os.path.abspath(os.getcwd())}\nSave path: {save_path}\n"

    with open(test_file, "w", encoding="utf-8") as f:
        f.write(content)

    print("Wrote test file:", os.path.abspath(test_file))
    print("Contents:\n", content)


if __name__ == "__main__":
    main()
