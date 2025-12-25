#!/usr/bin/env python3
"""
Script to update all Target references to use PTV (Planning Target Volume)
in examples and documentation.
"""

import json
import re
from pathlib import Path

def update_python_file(file_path):
    """Update a Python file to replace Target references with PTV."""
    with open(file_path, 'r') as f:
        content = f.read()
    
    original_content = content
    
    # Replace common patterns
    replacements = {
        '"Target"': '"PTV"',
        "'Target'": "'PTV'",
        'Target.nii.gz': 'PTV.nii.gz',
        '# target': '# ptv',
        'target_volume': 'ptv_volume',
        'target_file': 'ptv_file',
        'target_mask': 'ptv_mask',
        'target_array': 'ptv_array',
        'target_list': 'ptv_list',
        'target_image': 'ptv_image',
        'for target in': 'for ptv in',
    }
    
    for old, new in replacements.items():
        content = content.replace(old, new)
    
    # Only write if changed
    if content != original_content:
        with open(file_path, 'w') as f:
            f.write(content)
        return True
    return False

def update_notebook(notebook_path):
    """Update a Jupyter notebook to replace Target references with PTV."""
    with open(notebook_path, 'r') as f:
        notebook = json.load(f)
    
    modified = False
    replacements = {
        '"Target"': '"PTV"',
        "'Target'": "'PTV'",
        'Target mask': 'PTV mask',
        'Target region': 'PTV region',
        'In Target region': 'In PTV region',
        'target_mask': 'ptv_mask',
        'target_mask_np': 'ptv_mask_np',
        'actual_in_target': 'actual_in_ptv',
        'predicted_in_target': 'predicted_in_ptv',
    }
    
    for cell in notebook['cells']:
        if 'source' in cell:
            for i, line in enumerate(cell['source']):
                original_line = line
                for old, new in replacements.items():
                    line = line.replace(old, new)
                if line != original_line:
                    cell['source'][i] = line
                    modified = True
    
    if modified:
        with open(notebook_path, 'w') as f:
            json.dump(notebook, f, indent=1)
        return True
    return False

def main():
    repo_root = Path(__file__).parent.parent
    
    # Update Python examples
    examples_dir = repo_root / 'examples'
    python_files = list(examples_dir.glob('*.py'))
    
    print("Updating Python examples...")
    updated_count = 0
    for py_file in python_files:
        if update_python_file(py_file):
            print(f"  ✓ Updated {py_file.name}")
            updated_count += 1
    
    print(f"\nUpdated {updated_count} Python files")
    
    # Update notebooks
    notebooks_dir = repo_root / 'docs' / 'notebooks'
    notebooks = list(notebooks_dir.glob('*.ipynb'))
    
    print("\nUpdating Jupyter notebooks...")
    updated_count = 0
    for notebook in notebooks:
        if update_notebook(notebook):
            print(f"  ✓ Updated {notebook.name}")
            updated_count += 1
    
    print(f"\nUpdated {updated_count} notebooks")
    print("\n✓ All Target references updated to PTV!")

if __name__ == '__main__':
    main()
