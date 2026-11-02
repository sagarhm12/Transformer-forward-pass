# Understanding the Transformer --- From Scratch with NumPy

> A hands-on learning project that breaks the original Transformer
> encoder--decoder architecture into small, understandable pieces.

Transformers power many modern language models, but the architecture can
feel complicated when you first encounter it. This repository approaches
it step by step: each notebook explores one building block, explains the
calculations, and helps you see how the pieces fit together.

The implementations use **NumPy** rather than hiding the details behind
a deep-learning framework. The focus is learning how a Transformer
forward pass works---not building a production-ready language model.

## What you will learn

-   How text becomes token IDs and then embedding vectors.
-   How positional encoding represents token order.
-   How queries, keys, and values are used in attention.
-   How multi-head attention, feed-forward networks, residual
    connections, and LayerNorm work together.
-   How an encoder represents a source sequence.
-   Why decoder self-attention needs a causal mask.
-   How cross-attention lets the decoder use information from the
    encoder.
-   How the encoder and decoder combine into a complete forward pass
    that produces vocabulary logits.

## Transformer at a glance

``` text
SOURCE TOKEN IDs
       |
       v
Token embeddings + positional encoding
       |
       v
ENCODER: self-attention -> Add & Norm -> FFN -> Add & Norm
       |
       v
Encoder output ----------------------------------+
                                                 |
TARGET TOKEN IDs                                 |
       |                                         |
       v                                         |
Token embeddings + positional encoding           |
       |                                         |
       v                                         |
DECODER: masked self-attention -> Add & Norm      |
       |                                         |
       v                                         |
Cross-attention <--------------------------------+
       |
       v
Add & Norm -> FFN -> Add & Norm
       |
       v
Linear projection to vocabulary
       |
       v
LOGITS (scores for possible next tokens)
```

This is a simplified view of the original encoder--decoder Transformer.
The notebooks unpack each step and let you inspect the underlying
arrays.

## Repository notebooks

  -----------------------------------------------------------------------
  Notebook                            Purpose
  ----------------------------------- -----------------------------------
  `Tokenization.ipynb`                Introduces converting text into
                                      token IDs.

  `positional_encoding.ipynb`         Implements sinusoidal positional
                                      encoding.

  `self_attention.ipynb`              Explores scaled dot-product
                                      self-attention.

  `Multi-Head Attention.ipynb`        Splits attention into heads and
                                      combines their outputs.

  `FFN.ipynb`                         Implements the position-wise
                                      feed-forward network.

  `Layer_Norm.ipynb`                  Explains normalization over the
                                      feature dimension.

  `Complete_Encoder.ipynb`            Brings the encoder components
                                      together.

  `Encoder.ipynb`                     Explores the encoder.

  `masked self attention.ipynb`       Adds a causal mask to prevent
                                      looking at future positions.

  `Decoder MHA.ipynb`                 Implements masked multi-head
                                      attention for the decoder.

  `Cross_attention.ipynb`             Shows how decoder queries attend to
                                      encoder keys and values.

  `Decoder Block.ipynb`               Combines masked attention,
                                      cross-attention, FFN, residuals,
                                      and normalization.

  `Transformer.ipynb`                 Assembles the main encoder--decoder
                                      forward pass.

  `Demo.ipynb`                        Provides a small demonstration.
  -----------------------------------------------------------------------

Some concepts appear in both standalone notebooks and the complete
Transformer notebook. That repetition is intentional: first learn each
component in isolation, then see it used in the full model.

## Recommended learning order

1.  **Tokenization** --- convert text into token IDs.
2.  **Embeddings and positional encoding** --- turn IDs into vectors and
    represent token order.
3.  **Self-attention** --- understand Q, K, V and attention weights.
4.  **Multi-head attention** --- learn why attention is split into
    multiple heads.
5.  **FFN and LayerNorm** --- explore the other key parts of a
    Transformer block.
6.  **Encoder** --- combine self-attention and FFN with residual
    connections.
7.  **Masked self-attention** --- prevent decoder positions from seeing
    future tokens.
8.  **Cross-attention** --- connect the decoder to encoder output.
9.  **Decoder block** --- combine the decoder operations.
10. **Full Transformer** --- follow data from token IDs to vocabulary
    logits.

**Learning tip:** Run the cells yourself. Print intermediate arrays and
their shapes, change small input values, and check what changes. Shape
tracking is one of the best ways to understand attention.

## Core ideas, explained simply

### Tokenization and embeddings

A neural network cannot directly process ordinary words. Tokenization
represents text as token IDs, and an embedding table maps each ID to a
vector. A sentence with three tokens therefore becomes a sequence of
three vectors.

### Positional encoding

Attention does not automatically tell the model the order in which
tokens appeared. Positional encoding adds position-dependent information
to token embeddings. This project explores the sinusoidal approach used
by the original Transformer.

### Self-attention

Self-attention lets each token gather information from other tokens in
the same sequence. The input is projected into three representations:

-   **Query (Q):** what a token is looking for.
-   **Key (K):** what a token can be matched on.
-   **Value (V):** the information a token contributes.

Queries are compared with keys to produce scores. Softmax converts those
scores into weights, and the weights combine the values.

``` text
Attention(Q, K, V) = softmax(Q Kᵀ / √dₖ) V
```

`dₖ` is the key dimension for one attention head. Scaling by its square
root helps keep attention scores at a useful scale.

### Multi-head attention

Rather than calculate attention only once, the model uses multiple heads
with separate learned projections. The results from all heads are
concatenated and passed through an output projection. In this
implementation, `d_model` must be divisible by `num_heads`, and
`d_head = d_model / num_heads`.

### Feed-forward network (FFN)

The FFN transforms each token representation independently. In the
original Transformer design used here, it applies a linear
transformation, ReLU activation, and a second linear transformation.
Attention mixes information across positions; the FFN processes the
features at each position.

### Residual connections and LayerNorm

A residual connection adds a block's input to its output, giving
information a direct path through the network. LayerNorm normalizes each
token representation across its feature dimension and then applies
learned scale and bias parameters. The notebooks use the
post-normalization pattern: add the residual first, then normalize.

### Encoder

The encoder reads the source sequence and builds contextual
representations. Each encoder block contains self-attention followed by
a residual connection and LayerNorm, then an FFN followed by another
residual connection and LayerNorm. The encoder output is passed to
decoder cross-attention.

### Masked self-attention

During next-token prediction, the decoder must not use future target
tokens. A causal mask blocks attention to positions to the right. This
prevents information from a future token leaking into a prediction that
should only depend on earlier tokens.

### Cross-attention

Cross-attention allows the decoder to consult the encoded source
sequence. **Queries come from the decoder; keys and values come from the
encoder.** Because source and target lengths can differ, cross-attention
scores are generally rectangular rather than square.

### Logits and predictions

The decoder output is projected into vocabulary space. If the vocabulary
contains `vocab_size` tokens, the logits usually have shape:

``` text
(batch_size, target_seq_len, vocab_size)
```

Each position has one score per vocabulary token. Softmax can convert
those scores to probabilities. Logits are scores, not probabilities
themselves.

## Understanding tensor shapes

  -----------------------------------------------------------------------------------------------
  Tensor                  Typical shape                                   Meaning
  ----------------------- ----------------------------------------------- -----------------------
  Source IDs              `(batch_size, source_seq_len)`                  Source token IDs.

  Target IDs              `(batch_size, target_seq_len)`                  Target tokens supplied
                                                                          to the decoder.

  Embeddings              `(batch_size, seq_len, d_model)`                A vector for each
                                                                          token.

  Q, K, V before          `(batch_size, seq_len, d_model)`                Projected attention
  splitting heads                                                         representations.

  Q, K, V after splitting `(batch_size, num_heads, seq_len, d_head)`      Separate representation
  heads                                                                   for each head.

  Attention scores        `(batch_size, num_heads, query_len, key_len)`   Scores between queries
                                                                          and keys.

  Encoder output          `(batch_size, source_seq_len, d_model)`         Contextual source
                                                                          representations.

  Decoder output          `(batch_size, target_seq_len, d_model)`         Contextual target
                                                                          representations.

  Vocabulary logits       `(batch_size, target_seq_len, vocab_size)`      Scores for each
                                                                          vocabulary token at
                                                                          each target position.
  -----------------------------------------------------------------------------------------------

For cross-attention, `query_len` is normally the target length and
`key_len` is the source length.

## Requirements

-   Python 3
-   NumPy
-   Jupyter Notebook or JupyterLab

Install the basic dependencies:

``` bash
python -m pip install numpy notebook
```

## Getting started

### 1. Clone the repository

``` bash
git clone https://github.com/sagarhm12/Transformer-forward-pass.git
cd Transformer-forward-pass
```

### 2. (Recommended) Create a virtual environment

Windows:

``` bash
python -m venv .venv
.venv\Scripts\activate
```

macOS/Linux:

``` bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies and launch Jupyter

``` bash
python -m pip install numpy notebook
jupyter notebook
```

Open a notebook in the browser and run its cells from top to bottom.
Start with `Tokenization.ipynb`, then follow the recommended learning
order.

> **Note:** This repository is a collection of educational notebooks
> rather than a packaged Python library. Some notebooks may depend on
> variables or functions defined in earlier cells. If you see an
> undefined-variable error, check that the notebook's setup cells have
> been run in order.

## What does "forward pass" mean?

A forward pass sends input data through a model to calculate its output.
Here, token IDs are converted into embeddings, processed by the encoder
and decoder, and projected into vocabulary logits.

A forward pass alone does **not** train a model. Training also requires
a loss function, backpropagation to calculate gradients, and an
optimizer to update the parameters. The repository focuses on
understanding and assembling forward-pass components; do not assume that
every notebook includes training or a complete backward pass.

## Scope and limitations

**This project is:**

-   A step-by-step learning resource for Transformer mechanics.
-   A NumPy-based implementation that exposes the calculations.
-   A starting point for experiments with training and text generation.

**This project is not:**

-   A pretrained language model.
-   A production-ready Transformer library.
-   A guarantee of meaningful text generation without training.
-   A substitute for optimized implementations in deep-learning
    frameworks.

The notebooks prioritize clarity over speed, abstraction, and production
safeguards.

## Possible next steps

Once the forward pass makes sense, useful extensions include:

1.  Add a small end-to-end example and explain every intermediate tensor
    shape.
2.  Add teacher forcing and document the target-shifting convention.
3.  Add cross-entropy loss and verify the output probabilities.
4.  Implement backpropagation and validate gradients with finite
    differences.
5.  Train on a tiny synthetic sequence task and track the loss.
6.  Add a simple text-generation loop.
7.  Build a decoder-only, GPT-style model.
8.  Explore BPE tokenization, RoPE, RMSNorm, SwiGLU, grouped-query
    attention, and a KV cache.

These are suggested future improvements, not features that are
guaranteed to be present in the current repository.

## References

-   Original paper: [Attention Is All You
    Need](https://arxiv.org/abs/1706.03762)
-   Project repository: [Transformer-forward-pass on
    GitHub](https://github.com/sagarhm12/Transformer-forward-pass)

## Contributing

If you find a bug or have a clearer explanation, consider opening an
issue or submitting a pull request. Helpful contributions include
comments about tensor shapes, small numerical examples, and tests for
cases such as different source and target sequence lengths.

## Final note

The best way to learn a Transformer is to build it one piece at a time.
Run the code, inspect the shapes, test your assumptions, and make sure
you can explain each operation in your own words. Once the forward pass
feels clear, move on to loss, gradients, optimization, and training.
