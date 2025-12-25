#!/usr/bin/env python3
"""
Update all documentation notebooks to use the correct HuggingFace dataset
and structure names.
"""

import json
from pathlib import Path


def update_cell_content(cell_source, replacements):
    """Update cell source with replacements."""
    if isinstance(cell_source, list):
        content = ''.join(cell_source)
    else:
        content = cell_source
    
    for old, new in replacements.items():
        content = content.replace(old, new)
    
    # Return as list of lines (preserving notebook format)
    if isinstance(cell_source, list):
        return [content]
    return content


def update_notebook(notebook_path, replacements):
    """Update a Jupyter notebook with the given replacements."""
    print(f"Updating {notebook_path.name}...")
    
    with open(notebook_path, 'r', encoding='utf-8') as f:
        notebook = json.load(f)
    
    changes_made = 0
    for cell in notebook.get('cells', []):
        if 'source' in cell:
            old_source = cell['source']
            new_source = update_cell_content(old_source, replacements)
            if new_source != old_source:
                cell['source'] = new_source
                changes_made += 1
    
    if changes_made > 0:
        with open(notebook_path, 'w', encoding='utf-8') as f:
            json.dump(notebook, f, indent=2, ensure_ascii=False)
        print(f"  ✓ Updated {changes_made} cells")
    else:
        print(f"  - No changes needed")


def main():
    """Main function to update all notebooks."""
    docs_dir = Path(__file__).parent.parent / 'docs' / 'notebooks'
    
    # Common replacements for all notebooks
    replacements = {
        'contouraid/dosemetrics-examples': 'contouraid/dosemetrics-data',
        'data_path / "test_subject"': 'data_path / "longitudinal" / "time_point_1"',
        'data_path / "compare_plans" / "first"': 'data_path / "longitudinal" / "time_point_1"',
        'data_path / "compare_plans" / "last"': 'data_path / "longitudinal" / "time_point_2"',
        '"Target"': '"PTV"',
        "'Target'": "'PTV'",
        'target_mask': 'ptv_mask',
        'Target ': 'PTV ',
    }
    
    # Update each notebook
    notebooks = [
        'computing-metrics.ipynb',
        'exporting-results.ipynb',
    ]
    
    for notebook_name in notebooks:
        notebook_path = docs_dir / notebook_name
        if notebook_path.exists():
            update_notebook(notebook_path, replacements)
        else:
            print(f"  ✗ Not found: {notebook_path}")
    
    print("\n✓ All notebooks updated!")


if __name__ == '__main__':
    main()
