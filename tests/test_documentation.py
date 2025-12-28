"""
Tests for documentation notebooks to ensure they run without errors.
"""

import pytest
import nbformat
from nbconvert.preprocessors import ExecutePreprocessor
from nbclient.exceptions import CellExecutionError
from pathlib import Path


NOTEBOOKS_DIR = Path(__file__).parent.parent / "docs" / "notebooks"
TIMEOUT = 600  # 10 minutes per notebook

# Notebooks that are known to need user customization or have external dependencies
SKIP_NOTEBOOKS = {"comparing-plans"}


@pytest.fixture
def executor():
    """Create a notebook executor."""
    return ExecutePreprocessor(timeout=TIMEOUT, kernel_name="python3")


def get_notebooks():
    """Get all documentation notebooks."""
    notebooks = list(NOTEBOOKS_DIR.glob("*.ipynb"))
    return [(nb, nb.stem) for nb in notebooks if not nb.stem.startswith(".")]


@pytest.mark.parametrize("notebook_path,notebook_name", get_notebooks())
def test_notebook_execution(notebook_path, notebook_name):
    """
    Test that a documentation notebook runs without errors.

    Args:
        notebook_path: Path to the notebook file
        notebook_name: Name of the notebook (for reporting)
    """
    if notebook_name in SKIP_NOTEBOOKS:
        pytest.skip(f"Notebook {notebook_name} requires user customization")

    print(f"\n{'='*60}")
    print(f"Testing: {notebook_name}")
    print(f"{'='*60}")

    # Read the notebook
    with open(notebook_path, "r", encoding="utf-8") as f:
        nb = nbformat.read(f, as_version=4)

    # Execute the notebook
    notebook_executor = ExecutePreprocessor(timeout=TIMEOUT, kernel_name="python3")
    try:
        notebook_executor.preprocess(
            nb, {"metadata": {"path": str(notebook_path.parent)}}
        )
        print(f"✓ {notebook_name} executed successfully")
    except CellExecutionError as e:
        pytest.fail(f"Notebook {notebook_name} failed with error: {str(e)}")


def test_notebooks_exist():
    """Test that documentation notebooks exist."""
    notebooks = list(NOTEBOOKS_DIR.glob("*.ipynb"))
    assert len(notebooks) > 0, "No notebooks found in docs/notebooks/"

    expected_notebooks = [
        "01-basic-usage.ipynb",
        "02-nifti-io.ipynb",
        "03-dicom-io.ipynb",
        "04-getting-started-own-data.ipynb",
        "05-computing-metrics.ipynb",
        "06-comparing-plans.ipynb",
        "07-exporting-results.ipynb",
    ]

    notebook_names = [nb.name for nb in notebooks]
    for expected in expected_notebooks:
        assert expected in notebook_names, f"Expected notebook {expected} not found"


def test_notebook_structure():
    """Test that notebooks have the expected structure."""
    for notebook_path in NOTEBOOKS_DIR.glob("*.ipynb"):
        with open(notebook_path, "r", encoding="utf-8") as f:
            nb = nbformat.read(f, as_version=4)

        # Check that notebook has cells
        assert len(nb.cells) > 0, f"{notebook_path.name} has no cells"

        # Check that notebook has both markdown and code cells
        cell_types = [cell.cell_type for cell in nb.cells]
        assert "markdown" in cell_types, f"{notebook_path.name} has no markdown cells"
        assert "code" in cell_types, f"{notebook_path.name} has no code cells"

        # Check for title in first cell
        if nb.cells[0].cell_type == "markdown":
            first_cell = nb.cells[0].source
            assert first_cell.startswith(
                "#"
            ), f"{notebook_path.name} first cell is not a title"


def test_notebook_uses_correct_dataset():
    """Test that notebooks use the correct HuggingFace dataset."""
    correct_dataset = "contouraid/dosemetrics-data"

    for notebook_path in NOTEBOOKS_DIR.glob("*.ipynb"):
        with open(notebook_path, "r", encoding="utf-8") as f:
            nb = nbformat.read(f, as_version=4)

        # Check code cells for dataset reference
        for cell in nb.cells:
            if cell.cell_type == "code":
                source = cell.source
                # Check specifically for snapshot_download or similar HF functions
                if "snapshot_download" in source and "repo_id=" in source:
                    assert (
                        correct_dataset in source
                    ), f"{notebook_path.name} uses incorrect dataset in snapshot_download"


def test_notebook_uses_correct_structure_names():
    """Test that notebooks use PTV/CTV/GTV instead of 'Target'."""
    for notebook_path in NOTEBOOKS_DIR.glob("*.ipynb"):
        with open(notebook_path, "r", encoding="utf-8") as f:
            nb = nbformat.read(f, as_version=4)

        # Check code cells for incorrect structure names
        for i, cell in enumerate(nb.cells):
            if cell.cell_type == "code":
                source = cell.source
                # Check for problematic patterns
                if '"Target"' in source or "'Target'" in source:
                    pytest.fail(
                        f"{notebook_path.name} cell {i} uses 'Target' instead of PTV/CTV/GTV"
                    )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
