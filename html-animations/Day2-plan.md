# Day 2 Plan: How a Transformer Computes the Next Token

## 0. Opening bridge and orientation
- Title
- Explicit bridge from Day 1:
  - Day 1 = why behavior emerges
  - Day 2 = how the computation happens
- Learning outcomes
- One visual map of the complete forward pass
- Agenda: text -> tokens -> vectors -> attention/MLP stack -> logits -> next token

## 1. The full forward pass at a glance
- Input text enters
- Text becomes token IDs
- Token IDs become embeddings
- Positional information is injected
- Representations pass through repeated decoder blocks
- Final hidden state becomes logits
- Logits become probabilities
- A token is selected
- The token is appended and the loop repeats

## 2. Tokenization: turning text into workable units
- Why raw text cannot go directly into the network
- What a vocabulary is
- Why tokenization exists at all
- The design tradeoff: token granularity vs efficiency
- Why word-level tokenization breaks down
- Why character-level tokenization is inefficient
- Why byte-level tokenization is robust but costly
- Why subword tokenization is the practical middle ground
- BPE / merge-based intuition
- Token IDs as the interface between text and model
- Tokenization quirks: spaces, punctuation, rare words, multilingual text
- Interactive tokenization mini-game

## 3. Embeddings: from token IDs to dense vectors
- Why integer IDs are not meaningful by themselves
- Embedding lookup as the first learned representation
- One-hot intuition vs dense vector intuition
- High-dimensional spaces: what they are
- Why high-dimensional spaces are useful
- How semantic and syntactic structure can emerge in embedding space
- Embeddings as compressed learned knowledge
- Limits of simple analogy examples
- Interactive embedding placement intuition checkpoint

## 4. Positional information: giving order to the sequence
- Why order matters in language
- Why the model needs explicit position information
- Why self-attention alone does not encode order
- What goes wrong if order is missing
- Basic positional encoding / positional embedding intuition
- Combining token meaning with token position
- "Same tokens, different order" contrast
- Interactive order puzzle

## 5. The decoder-only transformer block: the main computational unit
- Shift from the original Transformer to decoder-only LLMs
- Why decoder-only is the right framing for next-token prediction
- One block's top-level anatomy
- The residual stream as the running state
- Attention sublayer
- Feed-forward sublayer
- Residual connections
- Normalization
- Many copies of the same block stacked deeply
- Macro view before the deep zoom

## 6. Self-attention setup: what each token computes
- A sequence of hidden vectors enters the block
- Each token is projected into query, key, and value
- Why there are three separate projections
- Learned projection matrices as specialized views
- Tokens comparing themselves to other tokens
- The causal constraint: no looking ahead
- Introducing the causal mask
- Interactive "who can this token see?" checkpoint

## 7. Dot products: the core operation
- Attention scores come from dot products
- What the dot product is measuring here
- Similarity in a learned feature space
- Why simple linear algebra is enough to build rich behavior
- Small by-hand dot product example
- Reframing attention as repeated similarity computation
- Transition from raw scores to attention weights

## 8. Scaled dot-product attention: from scores to context
- Building the score matrix
- Why scaling is needed
- Applying the causal mask
- Softmax converts scores into attention weights
- Weighted sum over values
- Producing a contextualized representation
- How one token becomes context-aware
- Interactive "predict the attention pattern" moment

## 9. Multi-head attention: parallel relational views
- Why one attention pattern is not enough
- Splitting into multiple heads
- Each head gets its own Q/K/V projections
- Each head computes its own attention pattern
- Heads run in parallel
- Concatenating head outputs
- Output projection back into model space
- Intuition for head specialization
- Visual comparison across heads

## 10. Residual paths and normalization: making deep transformers work
- Why the model preserves a residual stream
- Sublayers write updates into the stream
- Why residual connections matter for depth
- Why normalization matters for stable computation
- LayerNorm / RMSNorm as the conceptual role
- Read -> compute -> write-back framing
- Clean block schematic

## 11. Feed-forward networks: per-token computation after attention
- Attention mixes information across tokens
- FFNs transform each token independently
- Expansion into a larger hidden space
- Nonlinear transformation / gating
- Projection back to model dimension
- Why FFNs add capacity beyond attention
- FFN as the other major half of the block

## 12. Stacking blocks: how depth builds richer representations
- One block is only one step of refinement
- Repeated blocks progressively enrich the representation
- Information integration across layers
- Early vs middle vs later layer intuition
- Why depth matters
- Returning from one block to the full stack view
- Reconnecting to the full forward pass

## 13. The output head: turning hidden states into vocabulary scores
- Final hidden state at the current position
- Final normalization
- Projection into vocabulary space
- Unembedding / logits
- One score for every possible next token
- Softmax to probabilities
- Tie-back to Day 1's next-token distribution

## 14. Decoding: turning probabilities into an actual token
- Choosing a token from the distribution
- Argmax vs sampling
- Temperature
- Top-k / top-p
- Why decoding changes behavior without changing weights
- Appending the chosen token
- Autoregressive loop repeats

## 15. One complete end-to-end replay
- Pick one running example
- Show the full path from text to next token
- Tokenization
- Embeddings
- Position
- One decoder block pass
- Repeated stack effect
- Logits
- Decoding
- Produced token

## 16. Modern upgrades and exciting variants
- Why the baseline story is not the full frontier
- RoPE: position encoded inside attention
- Why RoPE is a useful upgrade over simpler positional methods
- MoE: sparse expert routing instead of one dense FFN
- Why MoE increases capacity efficiently
- Where these variants fit into the same baseline pipeline
- Optional quick mentions: other modern refinements (only as side notes)

## 17. Closing synthesis
- Recap the full computation chain
- Re-anchor the big idea:
  - the model computes next-token probabilities through structured linear algebra
- Reconnect to Day 1:
  - same next-token objective, now mechanistically explained
- Final visual of the full forward pass
- Closing takeaway
