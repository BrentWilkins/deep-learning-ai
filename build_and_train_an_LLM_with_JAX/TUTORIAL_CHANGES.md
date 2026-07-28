# Changes from the tutorial

This project intentionally differs from the tutorial implementation. The
changes below are grouped by purpose so it is clear which ones fix correctness
issues and which ones are optional improvements.

## Files needed in Colab

Upload both `L4.ipynb` and `helper.py`. The notebook imports the model, data
pipeline, hardware configuration, and sharding helpers from `helper.py`.
Uploading only the notebook will either fail or use a different helper module.

Select the TPU runtime before importing JAX. The hardware configuration is
created when the configuration cell runs.

## Correctness fixes

### Padding no longer contributes to the loss

The tutorial pads every story to 128 tokens with token ID 0 and then averages
loss over all 128 positions. Short stories therefore contribute many artificial
padding targets, making the reported loss look better without improving story
prediction.

`StoryDataset` now returns three arrays:

- `inputs`: tokens at positions 0 through n-1;
- `targets`: the next token for each input position;
- `loss_mask`: 1 for real targets and 0 for padding.

The loss is now:

```text
sum(token_loss * loss_mask) / sum(loss_mask)
```

The dataset temporarily keeps up to 129 tokens to construct 128 real
input/target pairs. It no longer invents a final padding target.

### The batch layout is explicit

The tutorial returns each story as a Python list. Grain interprets that list as
a pytree of token positions, producing `[sequence, batch]`; the notebook then
transposes it to `[batch, sequence]`.

The updated dataset returns NumPy arrays. Grain now directly produces
`[batch, sequence]`, so the fragile conversion and transpose were removed. The
tutorial transpose was compensating for its list representation; it was not
itself an attention bug.

### Temperature now affects generation

The tutorial divides logits by temperature and then uses `argmax`. Scaling all
logits does not change their ordering, so temperature has no effect.

Generation now uses top-k categorical sampling whenever temperature is greater
than zero. Temperature zero deliberately retains greedy `argmax` decoding.
`top_k` limits sampling to plausible candidates, and `seed` makes a run
reproducible.

## Hardware adaptations

`MiniGPT.transformer_blocks` selects `nnx.List` when the installed Flax exposes
it and falls back to a regular Python list otherwise. Newer Flax requires
`nnx.List` for a module collection, while the older Flax bundled with some
Colab TPU runtimes does not expose it. This compatibility selection preserves
the same sequence of transformer blocks without changing the architecture.

`get_hardware_config()` detects the JAX backend and all visible devices. Its
conservative defaults are:

| Backend | Per-device batch |
| --- | ---: |
| CPU | 4 |
| GPU | 16 |
| TPU | 32 |

The global batch is `per-device batch * device count`. Set
`per_device_batch_size` in L4 to an integer to override the default. Lower it
after an out-of-memory error; increase it only after a successful run shows
ample memory.

The model state is replicated on all devices, and the leading batch axis is
sharded across them. This lets an eight-core TPU process eight batch shards
instead of leaving seven cores idle.

The tutorial learning rate remains the explicit `3e-4`. Automatically changing
both batch size and learning rate would make it harder to isolate what each
experiment teaches.

## Deliberately unchanged model architecture

The tutorial's `TransformerBlock` accepts `feed_forward_dim` but does not use
it; each block contains only residual self-attention. This is unusual compared
with a standard GPT block, but it remains unchanged here so the padding,
sampling, and hardware experiments do not become an unrelated architecture
rewrite. Adding normalization and a feed-forward network is best treated as a
separate experiment with its own baseline.

## Training-loop and checkpoint changes

- The full Hugging Face split is requested with `max_stories=None`.
- Hugging Face loading uses its normal disk-backed Arrow cache instead of
  remote streaming. The first run downloads and prepares TinyStories; later
  runs reuse the cached source dataset. Tokenization still happens on demand
  in `StoryDataset`, keeping that tutorial step visible.
- The training step no longer returns the full logits tensor as unused
  auxiliary output, reducing accelerator memory pressure.
- Training logs are limited to roughly 100 reports per epoch so notebook
  outputs remain manageable.
- The learning-rate log uses a global step rather than appearing to restart
  each epoch.
- The corrected run is saved as `model_checkpoint_v2.orbax`.

The architecture is unchanged, but the version 2 name prevents the corrected
masked-loss run from overwriting or being confused with the earlier checkpoint.
Rerun L4 and use the version 2 checkpoint in L5.

## Output stripping and pre-commit

Repository-wide hooks strip saved notebook outputs and run course-specific
Ruff and mypy checks through `nbqa`. Run all checks from the repository root:

```bash
uv run --project building_agentic_RAG_with_Llamaindex \
  pre-commit run --all-files
```

To check only this course's notebooks, pass their paths with `--files`. The
required development tools are recorded in the course projects rather than
installed globally.

If `nbqa` is interrupted, it can briefly leave a generated
`*_nbqa_ipynb.py` file beside a notebook. These files are temporary, ignored by
Git, and safe to remove after confirming no check is still running.
