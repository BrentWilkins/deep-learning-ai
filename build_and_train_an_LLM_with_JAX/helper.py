from dataclasses import dataclass
from pathlib import Path

import flax.nnx as nnx
import grain.python as pygrain
import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec
import numpy as np
import optax
import tiktoken
from datasets import load_dataset

tokenizer = tiktoken.get_encoding("gpt2")
end_token_id = tokenizer.encode(
    "<|endoftext|>", allowed_special={"<|endoftext|>"}
)[0]
pad_token_id = 0

vocab_size = tokenizer.n_vocab
num_transformer_blocks = 6
maxlen = 128
embed_dim = 192
num_heads = 6
feed_forward_dim = int(2/3 * 4 * embed_dim)
batch_size = 24
num_epochs = 3


@dataclass(frozen=True)
class HardwareConfig:
    """Training defaults derived from the JAX devices visible to this process."""

    platform: str
    device_count: int
    per_device_batch_size: int
    batch_size: int


def get_hardware_config(per_device_batch_size=None):
    """Choose conservative defaults while allowing an explicit batch override.

    The returned ``batch_size`` is global: it is the per-device batch multiplied
    by every visible JAX device. TPU and multi-GPU notebooks therefore scale out
    automatically instead of silently using only a single accelerator.
    """
    platform = jax.default_backend()
    device_count = jax.device_count()
    default_per_device = {
        "cpu": 4,
        "gpu": 16,
        "tpu": 32,
    }.get(platform, 8)
    per_device_batch_size = (
        default_per_device
        if per_device_batch_size is None
        else int(per_device_batch_size)
    )
    if per_device_batch_size <= 0:
        raise ValueError("per_device_batch_size must be greater than zero")

    global_batch_size = per_device_batch_size * device_count
    return HardwareConfig(
        platform=platform,
        device_count=device_count,
        per_device_batch_size=per_device_batch_size,
        batch_size=global_batch_size,
    )


def create_data_shardings():
    """Return batch and replicated shardings spanning all visible devices."""
    mesh = Mesh(np.asarray(jax.devices()), ("data",))
    batch_sharding = NamedSharding(mesh, PartitionSpec("data", None))
    replicated_sharding = NamedSharding(mesh, PartitionSpec())
    return batch_sharding, replicated_sharding


def load_stories_from_file(
    file_path,
    max_stories = None
):
    """
    Efficiently load stories from a text file.
    Each story ends with <|endoftext|>.
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"Data file not found: {file_path}")

    print(f"Loading stories from {file_path}...")
    stories = []
    current_story = []

    with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
        for line in f:
            if '<|endoftext|>' in line:
                parts = line.split('<|endoftext|>')
                for part in parts[:-1]:
                    current_story.append(part)
                    story_text = ''.join(current_story).strip()
                    if story_text:
                        stories.append(story_text + '<|endoftext|>')
                        if max_stories and len(stories) >= max_stories:
                            break
                    current_story = []
                if parts[-1].strip():
                    current_story = [parts[-1]]
                else:
                    current_story = []
                if max_stories and len(stories) >= max_stories:
                    break
            else:
                current_story.append(line)
        if current_story and (not max_stories or len(stories) < max_stories):
            story_text = ''.join(current_story).strip()
            if story_text:
                stories.append(story_text + '<|endoftext|>')

    print(f"Loaded {len(stories):,} stories")
    return stories


def load_stories_from_hf(
    split="train",
    max_stories=10_000,
    shuffle=False,
    seed=42,
    cache_dir=None,
):
    """Load stories through the Hugging Face on-disk dataset cache.

    Set ``max_stories`` to an integer to bound the number of examples, or to
    ``None`` to consume the complete requested split. Hugging Face stores the
    downloaded and prepared dataset in its cache, so later runs can reuse it.
    ``cache_dir=None`` uses the standard Hugging Face cache location.
    """
    if max_stories is not None and max_stories <= 0:
        raise ValueError("max_stories must be greater than zero")

    amount = "all" if max_stories is None else f"up to {max_stories:,}"
    print(
        f"Loading {amount} stories from roneneldan/TinyStories ({split}); "
        "Hugging Face disk cache enabled..."
    )
    dataset = load_dataset(
        "roneneldan/TinyStories",
        split=split,
        cache_dir=cache_dir,
    )

    if shuffle:
        dataset = dataset.shuffle(seed=seed)

    stories = []
    rows = (
        dataset
        if max_stories is None
        else dataset.select(range(min(max_stories, len(dataset))))
    )
    for row in rows:
        story = row["text"].strip()
        if story:
            stories.append(story + "<|endoftext|>")

    if not stories:
        raise ValueError("No valid stories found in the Hugging Face dataset")

    print(f"Loaded {len(stories):,} stories from Hugging Face")
    return stories



class TransformerBlock(nnx.Module):

    def __init__(self, embed_dim, num_heads, ff_dim, *, rngs):
        self.attention = nnx.MultiHeadAttention(
            num_heads=num_heads,
            in_features=embed_dim,
            qkv_features=embed_dim,
            out_features=embed_dim,
            decode=False,
            rngs=rngs
        )

    def __call__(self, x, mask=None):
        attn_out = self.attention(x, mask=mask)
        x = x + attn_out
        return x

class TokenAndPositionEmbedding(nnx.Module):
    def __init__(self, maxlen, vocab_size, embed_dim, *, rngs):
        self.token_emb = nnx.Embed(vocab_size, embed_dim, rngs=rngs)
        self.pos_emb = nnx.Embed(maxlen, embed_dim, rngs=rngs)

    def __call__(self, x):
        seq_len = x.shape[1]
        positions = jnp.arange(seq_len)[None, :]
        return self.token_emb(x) + self.pos_emb(positions)

class MiniGPT(nnx.Module):

    def __init__(self, maxlen=maxlen, vocab_size=vocab_size, embed_dim=embed_dim, num_heads=num_heads,
                 feed_forward_dim=feed_forward_dim, num_transformer_blocks=num_transformer_blocks, *, rngs=nnx.Rngs(0)):

        self.maxlen = maxlen

        self.embedding = TokenAndPositionEmbedding(maxlen, vocab_size, embed_dim, rngs=rngs)

        # New Flax requires nnx.List for a module collection, while the older
        # Flax bundled with some Colab runtimes supports plain lists and does
        # not expose nnx.List.
        module_list = getattr(nnx, "List", list)
        self.transformer_blocks = module_list([
            TransformerBlock(embed_dim, num_heads, feed_forward_dim, rngs=rngs)
            for _ in range(num_transformer_blocks)
        ])

        self.output_layer = nnx.Linear(embed_dim, vocab_size, use_bias=False, rngs=rngs)

    def causal_attention_mask(self, seq_len):
        return jnp.tril(jnp.ones((seq_len, seq_len)))

    def __call__(self, token_ids):
        seq_len = token_ids.shape[1]
        mask = self.causal_attention_mask(seq_len)

        x = self.embedding(token_ids)

        for block in self.transformer_blocks:
            x = block(x, mask=mask)

        logits = self.output_layer(x)

        return logits

def generate_text(
    model,
    start_tokens,
    max_new_tokens=50,
    temperature=0.8,
    top_k=50,
    seed=42,
):
    """Generate tokens with greedy decoding at zero or top-k sampling otherwise."""
    temperature = float(temperature)
    max_new_tokens = int(max_new_tokens)
    seed = int(seed)
    if temperature < 0:
        raise ValueError("temperature must be non-negative")
    if max_new_tokens < 0:
        raise ValueError("max_new_tokens must be non-negative")
    if top_k is not None and top_k <= 0:
        raise ValueError("top_k must be positive or None")

    tokens = list(start_tokens)
    if not tokens:
        raise ValueError("start_tokens must contain at least one token")
    key = jax.random.PRNGKey(seed)

    for _ in range(max_new_tokens):
        context = tokens[-model.maxlen:]

        # RIGHT-pad to match training (not left-pad!)
        actual_len = len(context)
        if actual_len < model.maxlen:
            context = context + [0] * (model.maxlen - actual_len)

        context_array = jnp.array(context)[None, :]
        logits = model(context_array)

        next_token_logits = logits[0, actual_len - 1, :]
        if temperature == 0:
            next_token = int(jnp.argmax(next_token_logits))
        else:
            key, sample_key = jax.random.split(key)
            if top_k is None:
                next_token = int(
                    jax.random.categorical(
                        sample_key, next_token_logits / temperature
                    )
                )
            else:
                k = min(int(top_k), next_token_logits.shape[-1])
                top_logits, top_indices = jax.lax.top_k(next_token_logits, k)
                sampled_index = jax.random.categorical(
                    sample_key, top_logits / temperature
                )
                next_token = int(top_indices[sampled_index])

        if next_token == end_token_id:
            break

        tokens.append(next_token)

    return tokenizer.decode(tokens)



def generate_story(
    model,
    story_prompt,
    temperature=0.8,
    max_new_tokens=50,
    top_k=50,
    seed=42,
):
    start_tokens = tokenizer.encode(story_prompt)[-model.maxlen:]
    generated = generate_text(
        model,
        start_tokens,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        seed=seed,
    )
    return generated


class StoryDataset:
    def __init__(self, stories, maxlen, tokenizer):
        self.stories = stories
        self.maxlen = maxlen
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.stories)

    def __getitem__(self, idx):
        story = self.stories[idx]
        tokens = self.tokenizer.encode(story, allowed_special={'<|endoftext|>'})

        # Keep one extra token so inputs and next-token targets can both have
        # maxlen positions without inventing a padding target at the end.
        tokens = tokens[: self.maxlen + 1]
        inputs = tokens[:-1]
        targets = tokens[1:]
        loss_mask = [1.0] * len(targets)

        padding = self.maxlen - len(inputs)
        inputs.extend([pad_token_id] * padding)
        targets.extend([pad_token_id] * padding)
        loss_mask.extend([0.0] * padding)

        # Returning arrays makes Grain produce [batch, sequence] arrays rather
        # than transposing a Python list as a pytree of sequence positions.
        return {
            "inputs": np.asarray(inputs, dtype=np.int32),
            "targets": np.asarray(targets, dtype=np.int32),
            "loss_mask": np.asarray(loss_mask, dtype=np.float32),
        }


def _create_dataloader(
    stories,
    batch_size,
    maxlen,
    num_epochs=1,
    shuffle=False,
    seed=42,
):
    """Build the existing Grain pipeline from already-loaded stories."""
    if not stories:
        raise ValueError("No valid stories found in the dataset")

    estimated_batches_per_epoch = len(stories) // batch_size
    print(f"Estimated batches per epoch: {estimated_batches_per_epoch:,}")

    dataset = StoryDataset(stories, maxlen, tokenizer)
    sampler = pygrain.IndexSampler(
        num_records=len(dataset),
        shuffle=shuffle,
        seed=seed,
        shard_options=pygrain.NoSharding(),
        num_epochs=num_epochs,
    )
    dataloader = pygrain.DataLoader(
        data_source=dataset,
        sampler=sampler,
        operations=[
            pygrain.Batch(batch_size=batch_size, drop_remainder=True)
        ],
    )

    print(f"Created DataLoader with batch_size={batch_size}, maxlen={maxlen}")
    return dataloader, estimated_batches_per_epoch


def load_and_preprocess_hf_data(
    batch_size,
    maxlen,
    split="train",
    max_stories=100_000,
    num_epochs=1,
    shuffle=False,
    seed=42,
    cache_dir=None,
):
    """Load a cached Hugging Face split and create a Grain DataLoader.

    Set ``max_stories=None`` to use every example in the split.
    """
    stories = load_stories_from_hf(
        split=split,
        max_stories=max_stories,
        shuffle=False,
        seed=seed,
        cache_dir=cache_dir,
    )
    return _create_dataloader(
        stories,
        batch_size=batch_size,
        maxlen=maxlen,
        num_epochs=num_epochs,
        shuffle=shuffle,
        seed=seed,
    )



def load_and_preprocess_data(
    file_path,
    batch_size,
    maxlen,
    max_stories = 100_000,
    num_epochs = 1,
    shuffle = False,
    seed = 42
):
    """
    Load and preprocess TinyStories data with memory-efficient chunk reading.

    Args:
        file_path: Path to the text file
        batch_size: Batch size for training
        maxlen: Maximum sequence length
        max_stories: Maximum number of stories to load (for memory efficiency)
        num_epochs: Number of training epochs
        shuffle: Whether to shuffle the data
        seed: Random seed for reproducibility

    Returns:
        Tuple of (Grain DataLoader, estimated_batches_per_epoch)
    """

    # Load and validate file
    file_path = file_path

    print(f"Loading data from {file_path} (max {max_stories:,} stories)")

    # Read file in chunks to avoid loading entire file into memory
    stories = []
    current_story = []

    with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
        for line in f:
            if '<|endoftext|>' in line:
                # Split on end token and process parts
                parts = line.split('<|endoftext|>')
                for i, part in enumerate(parts[:-1]):  # All but last part have end tokens
                    current_story.append(part)
                    story_text = ''.join(current_story).strip()
                    if story_text:
                        stories.append(story_text + '<|endoftext|>')
                        if len(stories) >= max_stories:
                            break
                    current_story = []

                # Last part becomes start of next story
                if parts[-1].strip():
                    current_story = [parts[-1]]

                if len(stories) >= max_stories:
                    break
            else:
                current_story.append(line)

        # Don't forget the last story if file doesn't end with end token
        if current_story and len(stories) < max_stories:
            story_text = ''.join(current_story).strip()
            if story_text:
                stories.append(story_text + '<|endoftext|>')

    print(f"Loaded {len(stories):,} stories")
    return _create_dataloader(
        stories,
        batch_size=batch_size,
        maxlen=maxlen,
        num_epochs=num_epochs,
        shuffle=shuffle,
        seed=seed,
    )
