# Notebooks

Kaggle notebook scripts, run in this order for a full pipeline:

| Order | File | Purpose |
|---|---|---|
| 1 | [`kaggle_prototype_notebook.py`](kaggle_prototype_notebook.py) | GT-ILP proof-of-concept — no LLM required, uses ground-truth predicates to validate the symbolic engine end-to-end |
| 2 | [`kaggle_preprocessing_notebook.py`](kaggle_preprocessing_notebook.py) | Runs LLM extraction (Groq / Llama 3.3 70B) over the raw dataset and caches parsed facts to disk |
| 3 | [`kaggle_notebook_per_cell.py`](kaggle_notebook_per_cell.py) | Primary setup notebook — writes the engine package cell-by-cell, trains ILP, evaluates on the test set |
| — | [`kaggle_notebook_complete.py`](kaggle_notebook_complete.py) | Same pipeline as above, bundled into a single self-contained cell (no engine dataset upload needed) |
| — | [`kaggle_evaluation_notebook.py`](kaggle_evaluation_notebook.py) | Standalone evaluation harness — loads a trained engine and dataset, produces metrics + SME report |

See each file's header comment for prerequisites (uploaded datasets, internet/GPU settings).
