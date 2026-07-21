# Contributing to DoseMetrics

Changes are welcome through the
[GitHub repository](https://github.com/contouraid/dosemetrics).

## Preview documentation locally

From the repository root, install the documentation dependencies and start
MkDocs:

```bash
python -m pip install -e ".[docs]"
mkdocs serve --dev-addr 127.0.0.1:8000
```

Open [http://127.0.0.1:8000/dosemetrics/](http://127.0.0.1:8000/dosemetrics/).
MkDocs watches `docs/` and `mkdocs.yml`; saving a file rebuilds the preview and
refreshing the browser shows the change.

The main metric page is available directly at
[http://127.0.0.1:8000/dosemetrics/user-guide/quality-metrics/](http://127.0.0.1:8000/dosemetrics/user-guide/quality-metrics/).

Before submitting a documentation change, run the strict build and the
Markdown API check:

```bash
mkdocs build --strict
pytest -q tests/test_documentation.py -k "not notebook_execution"
```

The generated `site/` directory is only a build artifact and should not be
committed.

## Propose a change

- [Open an issue](https://github.com/contouraid/dosemetrics/issues) for a bug,
  unclear definition, or larger documentation proposal.
- [Open a pull request](https://github.com/contouraid/dosemetrics/pulls) for a
  concrete correction.
- Keep public API examples importable; documentation tests inspect every
  Python code block.
