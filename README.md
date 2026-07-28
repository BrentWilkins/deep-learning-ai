# AI Course Work

This repository is a personal learning workspace for AI courses, primarily from
DeepLearning.AI. It contains course notebooks, exercises, helper code, and
experiments made while working through the material.

The goal is to preserve the instructional structure of each course while making
the examples runnable on current hardware and software. When an example differs
materially from the original tutorial, the change should be documented in that
course directory.

## Courses

| Directory                                                                       | Topics                                                                                               |
| ------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------- |
| [`build_and_train_an_LLM_with_JAX`](build_and_train_an_LLM_with_JAX/)           | Building and training a small transformer language model with JAX, Flax NNX, Grain, and TinyStories  |
| [`building_agentic_RAG_with_Llamaindex`](building_agentic_RAG_with_Llamaindex/) | Building agentic RAG systems with LlamaIndex, router query engines, Ollama, and local embeddings |

## Project structure

Each course directory is an independent Python project with its own
`pyproject.toml`, `uv.lock`, virtual environment, notebooks, and supporting
files. Dependencies should be installed from inside the course being studied
rather than from the repository root.

For example:

```bash
cd build_and_train_an_LLM_with_JAX
uv sync
```

Then select that directory's `.venv` interpreter as the kernel for its
notebooks.

Repository-wide notebook checks are configured in `.pre-commit-config.yaml`.
Run them from this directory with:

```bash
uv run --project building_agentic_RAG_with_Llamaindex \
  pre-commit run --all-files
```

The hooks strip notebook outputs and run course-specific Ruff and mypy checks
through `nbqa`. An interrupted `nbqa` process can briefly leave a generated
`*_nbqa_ipynb.py` file beside a notebook; these temporary files are ignored and
can be safely removed if they do not disappear when the process exits.

## Notes

- These are learning projects, not production applications.
- API keys and other secrets belong in environment variables or an untracked
  `.env` file; they should never be committed.
- Model checkpoints, downloaded datasets, caches, and other generated artifacts
  may be large and should generally remain outside version control.
- Course materials and included assets remain the property of their respective
  authors and copyright holders.
