# Building Agentic RAG with LlamaIndex

This repository contains a local-model adaptation of the DeepLearning.AI
*Building Agentic RAG with LlamaIndex* course exercises.

Lesson 1 builds a router query engine over the MetaGPT paper. It uses:

- LlamaIndex for document loading, indexing, routing, and response synthesis
- `BAAI/bge-small-en-v1.5` for local CPU embeddings
- `qwen3.6:27b` served locally through Ollama
- Structured function calling when the selected model supports it, with a
  plain-LLM selector fallback

## Prerequisites

- Python 3.13 or newer
- [uv](https://docs.astral.sh/uv/)
- [Ollama](https://ollama.com/)
- Enough memory for the chosen local model

## Setup

If Ollama is not already running as a service, start it in a separate
terminal:

```bash
ollama serve
```

Install and test the model from another terminal:

```bash
ollama pull qwen3.6:27b
ollama run qwen3.6:27b
```

Enter `/bye` after the model responds. Then install the Python environment:

```bash
uv sync --dev
```

Open [`lesson_1/L1_Router_Engine.ipynb`](lesson_1/L1_Router_Engine.ipynb) in
Jupyter or VS Code and run the cells in order. The notebook checks that Ollama,
the configured model, and PDF extraction are working before it builds the
indexes.

## Configuration

The notebook defines `OLLAMA_URL`, `OLLAMA_MODEL`, and the LlamaIndex context
window near the top. Adjust those values if you use a different Ollama model or
hardware configuration.

The included defaults reserve GPU memory for generation by running the small
BGE embedding model on CPU. Local summary generation is serialized, while
specific questions retrieve the five most relevant chunks.

## Project layout

```text
lesson_1/
├── 5488_MetaGPT_Meta_Programming_.pdf
├── L1_Router_Engine.ipynb
├── helper.py
└── utils.py
```

`utils.py` contains reusable router and per-document tool builders. It remains
provider-independent: callers may supply any compatible LlamaIndex LLM and
embedding model.
