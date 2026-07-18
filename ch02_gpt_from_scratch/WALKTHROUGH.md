# Chapter 2 Walkthrough — Build GPT From Scratch

Companion to `gpt_from_scratch.ipynb`. This doc is for *understanding*, not reference — read it
alongside the code, not instead of it. `RUNBOOK.md` (repo root) has the milestone log, why-we-did-it
notes, and interview angles; this file has the line-by-line mechanics.

---

## Stage 2: The Bigram Language Model

### Why build this first?
It's deliberately the dumbest possible language model — one character predicts the next, with zero
memory of anything earlier. We build and train it *before* self-attention specifically so the failure
mode is visible and motivated, not just asserted. You'll feel exactly why attention is necessary.

### Setup — `__init__`
```python
self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)
```
- **What it is:** a lookup table, one row per character in the vocabulary.
- **The content of each row:** raw scores (logits) for "what character comes next," one score per
  possible next character.
- **Size:** if `vocab_size = 65`, this is a 65×65 grid — 65 rows (one per input character), 65 columns
  (one score per possible next character).

Note this is *not* a normal embedding used for meaning — because `nn.Embedding(vocab_size, vocab_size)`
maps each token directly to a full row of next-token scores, the "embedding" and the "prediction" are
the same table. That's what makes this a bigram model rather than a real language model: there's no
intermediate representation, no way to combine information from more than one character.

### Forward pass — `forward`
```python
logits = self.token_embedding_table(idx)
```
Feed in a batch of current-character indices (`idx`); for each one, pull out its row of next-character
scores. Nothing more happens — no attention, no mixing across positions.

```python
B, T, C = logits.shape
logits = logits.view(B*T, C)
targets = targets.view(B*T)
loss = F.cross_entropy(logits, targets)
```
`F.cross_entropy` expects a flat list of predictions and a flat list of correct answers, not a 3D
tensor — `.view()` flattens `(batch, time, vocab)` into `(batch*time, vocab)` so every character
position across every sequence in the batch becomes one independent training example.

### Text generation — `generate`
```python
logits = logits[:, -1, :]
```
The bigram model only ever cares about the single most recent character, so this discards every
earlier prediction and keeps only the scores computed from the last token in the sequence.

```python
probs = F.softmax(logits, dim=-1)
idx_next = torch.multinomial(probs, num_samples=1)
```
`softmax` turns raw scores into probabilities that sum to 1. `torch.multinomial` then samples one
character from that probability distribution — a weighted random draw, not "always pick the highest
score" (that would be greedy decoding, which we're not using here).

```python
idx = torch.cat((idx, idx_next), dim=1)
```
Appends the newly sampled character onto the sequence so the next loop iteration can use it.

### Reading the sanity-check print
```python
print(f"Expected loss: {-torch.log(torch.tensor(1.0/vocab_size)):.4f}")
```
An untrained model's weights are random, so it's guessing uniformly at random over the vocabulary. If
there are 65 possible characters, a correct random guess happens ~1/65 of the time — this line
computes the cross-entropy loss that *pure random guessing* would produce. If your model's actual
initial loss matches this number, that confirms initialization is correct (nothing is broken, biased,
or leaking information it shouldn't have yet).

### Decoding `B`, `T`, `C`
These three letters show up constantly in transformer code — worth memorizing cold:
- **B (Batch):** how many independent sequences are processed simultaneously.
- **T (Time):** how many tokens are in each sequence (this is `block_size` at training time).
- **C (Channels):** the width of the score/feature vector at each position — here, `vocab_size`,
  because the bigram model's "features" are literally next-token scores. In later stages `C` becomes
  `n_embd` (the embedding dimension) once we introduce real embeddings.

### Worked mini-example
Toy vocabulary: `["cat", "sat", "mat"]` → `vocab_size = 3`, so `C = 3`.

Toy 3×3 embedding table (random init):
```
Row 0 (cat): [1.0, 4.0, -1.0]   → model currently guesses "sat" is likely next
Row 1 (sat): [-2.0, -3.0, 5.0]  → model currently guesses "mat" is likely next
Row 2 (mat): [0.0, 0.0, 0.0]    → no preference yet
```

Two sequences processed together (`B = 2`), two tokens each (`T = 2`):
```
idx = [[0, 1],   # "cat sat"
       [1, 2]]   # "sat mat"
# shape (B, T) = (2, 2)
```

The lookup replaces every index with its row from the table:
```
logits = [
  [[1.0, 4.0, -1.0], [-2.0, -3.0, 5.0]],   # sequence 1: cat, sat
  [[-2.0, -3.0, 5.0], [0.0, 0.0, 0.0]]     # sequence 2: sat, mat
]
# shape (B, T, C) = (2, 2, 3)
```

### Why `generate` slices `logits[:, -1, :]`
For sequence 1, the last token processed was "sat" — we don't need a prediction for what follows
"cat" anymore (that's in the past); we only want what follows "sat". Slicing out index `-1` along the
time dimension collapses `(B, T, C)` → `(B, C)`:
```
logits[:, -1, :] = [
  [-2.0, -3.0, 5.0],   # latest prediction for sequence 1 (after "sat")
  [0.0, 0.0, 0.0]      # latest prediction for sequence 2 (after "mat")
]
# shape (B, C) = (2, 3)
```

---

## Training loop, explained with an analogy

**The mental model — a blind hiker on a foggy mountain**, looking for the lowest point (the point of
best predictions):
- **The hiker** = the model's current weights (its position on the "loss landscape")
- **The slope underfoot** = the gradient, computed by `loss.backward()` — which direction is downhill
- **The guide dog** = the AdamW optimizer — remembers recent steps and nudges the hiker smoothly
  toward the bottom rather than lurching around

### Walking through one training step
```python
xb, yb = get_batch('train')
```
Grab `batch_size` (32) random snippets. `xb` = current characters, `yb` = the actual next characters
that followed them in the real text — the ground truth.

```python
logits, loss = model(xb, yb)
```
Forward pass: the model predicts, then compares its predictions to `yb` and computes a penalty score.
Early on (step 0) this loss is close to the random-guessing baseline (~4.7 for a 65-character
vocabulary) — the model knows nothing yet.

```python
optimizer.zero_grad(set_to_none=True)
```
PyTorch accumulates gradients by default (adds new ones to old ones) — this wipes the slate before
computing new gradients, so last step's calculation doesn't contaminate this one.

```python
loss.backward()
```
Computes the gradient: for every number in the embedding table, how much would nudging it up or down
change the loss? This is "feeling the slope."

```python
optimizer.step()
```
AdamW uses that slope (plus its memory of recent steps) to actually update the embedding table's
numbers, moving them toward better predictions.

### What actually happens over 10,000 steps
Loss drops from ~4.7 (random) to ~2.4-2.5. The model is tuning its lookup table so that common
character pairs (like `t` → `h`) get higher scores. That's *all* it can learn — pairwise statistics,
nothing longer-range.

### Why AdamW specifically
"Adaptive Moment Estimation with Weight Decay" — the default optimizer for nearly all modern deep
learning, including GPT-4-class models, because it combines three things:
1. **Momentum** — like a heavy ball rolling downhill, it carries through small bumps/local
   irregularities in the loss landscape instead of getting stuck.
2. **Adaptive per-parameter learning rates** — frequent characters (like `e`) get smaller, more
   careful updates; rare characters (like `z`) get larger updates so they actually learn from the few
   examples they appear in.
3. **Weight decay (the "W")** — a gentle penalty that keeps weights small and smooth, preventing the
   model from over-fitting to any single pattern it's seen.

### Why the output is still gibberish
Generated text: `CEng ay, Theyomy t blll,...` — slightly more English-shaped than pure noise, but
still broken. **The bigram model has zero memory beyond one character.**

Concrete example: to spell "t-h-e," after printing `t`, the model wipes its memory and looks only at
`h` to decide the next letter. But `h` alone is ambiguous — "that" wants `a` next, "this" wants `i`,
"the" wants `e`. Since the model can't see that a `t` came before the `h`, it's guessing blind at every
single step, and the result is a chaotic mix of fragments from different words.

**This is exactly the problem self-attention solves** — letting every character look back at
everything before it, not just the immediately preceding one. That's Part 9 onward.
