# Chapter 2 Walkthrough — Build GPT From Scratch

Companion to `gpt_from_scratch.ipynb`. This doc is for *understanding*, not reference — read it
alongside the code, not instead of it. `RUNBOOK.md` (repo root) has the milestone log, why-we-did-it
notes, and interview angles; this file has the line-by-line mechanics.

**One example runs through this entire document**, so numbers stay consistent and comparable
end-to-end: the sentence `"the cat sat on the mat. the cat ran to the mat. the dog sat too."`
(64 characters, 14-character vocabulary: `[' ', '.', 'a', 'c', 'd', 'e', 'g', 'h', 'm', 'n', 'o', 'r', 's', 't']`).
All numbers below were actually executed, not hand-computed — you can reproduce every one of them.

---

## Stage 2: The Bigram Language Model

### Why build this first?
It's deliberately the dumbest possible language model — one character predicts the next, with zero
memory of anything earlier. We build and train it *before* self-attention specifically so the failure
mode is visible and motivated, not just asserted. You'll feel exactly why attention is necessary.

### The model's only layer
```python
self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)
```
- **What it is:** a lookup table, one row per character in the vocabulary.
- **The content of each row:** raw scores (logits) for "what character comes next," one score per
  possible next character.
- **Size:** with our 14-character vocab, this is a 14x14 grid -- 14 rows (one per input character), 14
  columns (one score per possible next character). In the real Shakespeare notebook, 65x65.

This is *not* a normal embedding used for meaning -- because `nn.Embedding(vocab_size, vocab_size)`
maps each token directly to a full row of next-token scores, the "embedding" and the "prediction" are
the same table. That's what makes this a bigram model rather than a real language model: no
intermediate representation, no way to combine information from more than one character.

> **Common point of confusion, worth stating explicitly:** the embedding table's starting values are
> **not targets**. They're the model's randomly-initialized *weights/parameters* -- before training,
> PyTorch fills every row with small, arbitrary random numbers via a specific initialization scheme.
> **Targets** are the actual next-characters from the real text; they never change, and they're what
> predictions get compared against. Training is the process of nudging the random starting weights,
> step by step, so that looking up a row starts pointing toward the *real* target for that character.

---

## The whole pipeline, once, at a glance

Before the line-by-line dive, here's the full path from text to a trained (if still weak) model, with
real executed output at every stage, so you have the big picture before zooming in.

### Step 1 -- the sequence
64 characters, encoded via `stoi`; round-trip through `decode(encode(text))` confirms the mapping is
lossless.

### Steps 2-3 -- train/val split
90/10 -> 57 training characters, 7 validation characters. (This toy example is tiny for
illustration -- the real notebook splits ~1M/~110K characters the same way.)

### Step 4 -- forming a batch
```
xb (inputs)  shape: (4, 8)   -> B=4 sequences, T=8 characters each
yb (targets) shape: (4, 8)   -> same shape, shifted 1 character later

First sequence: input ' the cat' -> target 'the cat '
```
`yb` is **not** a longer continuation of `xb` -- it's the exact same length, just each character shifted
one position later in the source text. `yb[t]` is always "whatever character comes right after `xb[t]`
in the real text," position by position.

### Step 5 -- logits shape and the flatten
Real (not cherry-picked) randomly-initialized embedding rows before any training:
```
Row 0 (' '): [1.927, 1.487, 0.901, -2.106, 0.678, ...]
Row 1 ('.'): [-0.769, 0.762, 1.642, -0.160, -0.497, ...]
```
Small arbitrary floats -- exactly what "random init" looks like; nothing meaningful yet.
```
Raw logits shape: (4, 8, 14)  -> (B, T, C)
After view():     logits (32, 14), targets (32,)      [4*8 = 32 independent examples]
```

### Step 6 -- cross entropy
```
Loss (untrained model):        3.3848
Theoretical random-guessing:   2.6391   (= -log(1/14))
```
Same ballpark (small-batch sampling noise keeps them from matching exactly with only 32 examples) --
confirms the model starts knowing nothing, as expected.

### Step 7 -- real training, loss over iterations
```
Step    0: loss = 3.0829
Step  200: loss = 1.3798
Step  400: loss = 0.8465
Step  600: loss = 0.7110
Step  800: loss = 0.7360
Step 1600: loss = 0.8172
Step 2000: loss = 0.7943
```
Drops fast, then bounces around rather than smoothly decreasing forever -- normal here, for two
reasons: (1) every step samples a fresh random batch, so per-step "difficulty" varies, and (2) this
corpus is so small (57 characters) the model nearly memorizes it, so it's oscillating near its actual
floor. On the real Shakespeare corpus (1.1M characters) the curve is much smoother because per-batch
noise averages out over far more data.

### Step 8 -- generation after training
```
' cat t. t don mato ratog t sat. do mat. t rat to t o san mato'
```
Recognizable fragments (`cat`, `mato`, `sat.`) because strong character-pair statistics were learned
from this small repetitive corpus -- but still no real coherence. Why, exactly, is the subject of the
rest of this document.

---

## `forward()`, line by line

### When/how it actually gets called
You never call `model.forward(xb, yb)` directly. Writing `model(xb, yb)` triggers Python's `__call__`
protocol; `nn.Module` implements `__call__` internally and that in turn invokes your `forward()` method
(plus autograd bookkeeping). Concretely: `model(xb, yb)` -> `nn.Module.__call__` -> `forward(xb, yb)`.
This happens in three places in the notebook: the sanity-check cell (one pass), every step of the
10,000-step training loop, and inside `generate()` (with `targets=None`, so the loss branch is skipped).

### The lookup -- `logits = self.token_embedding_table(idx)`
Zooming into one specific example for exact tracing: take the 7-character slice `xb = "the cat"`
(indices 0-6 of the source text) with target `yb = "he cat "` (indices 1-7 -- same length, shifted by
one, per the Step 4 clarification above).
```
Full text:  t  h  e     c  a  t     s  a  t
index:      0  1  2  3  4  5  6  7  8  9  10

xb = "the cat"     (indices 0-6, 7 characters)
yb = "he cat "     (indices 1-7, SAME 7 characters, each one position later)
```
This line does **not** treat `"the cat"` as one unit. It looks up each of the 7 characters
*individually*, and returns each one's own row from the embedding table:
```
position t=0  char='t' -> logits row: [0.479, 1.354, -0.159, ...]   (14 scores, one per possible next char)
position t=1  char='h' -> logits row: [0.785, 0.029, 0.641, ...]
position t=2  char='e' -> logits row: [-1.055, 1.278, -0.172, ...]
... (7 rows total, one per input character)
```
`logits` shape is `(B=1, T=7, C=14)` -- literally "for each of the 7 positions, here are the 14 scores
for what character comes next."

### `B, T, C` -- the glossary
These three letters show up constantly in transformer code -- worth memorizing cold:
- **B (Batch):** how many independent sequences are processed simultaneously.
- **T (Time):** how many tokens are in each sequence (this is `block_size` at training time).
- **C (Channels):** the width of the score/feature vector at each position -- here, `vocab_size`,
  because the bigram model's "features" are literally next-token scores. In later stages `C` becomes
  `n_embd` (the embedding dimension) once we introduce real embeddings.

### Why `B*T` when flattening?
```python
B, T, C = logits.shape
logits = logits.view(B*T, C)
targets = targets.view(B*T)
loss = F.cross_entropy(logits, targets)
```
For our traced example: `B=1, T=7 -> B*T = 7`. There are 7 *individual characters* in this batch, and
each one needs its own "was the prediction correct?" score. `cross_entropy` doesn't understand
"batch" vs. "time" as separate concepts -- it just wants a flat list of `(prediction, correct_answer)`
pairs. Flattening turns `(1, 7, 14)` into `(7, 14)` -- 7 independent classification examples, each with
14 candidate scores. (In the batched Step 4-6 example above, the same flatten turns `(4, 8, 14)` into
`(32, 14)` -- 32 independent examples, one per character position across all 4 sequences.)

### The actual cross-entropy math, per position
For each position: turn the 14 raw scores into probabilities with softmax, then take `-log()` of the
probability assigned to the *actual correct answer*:
```
pos 0: 't' -> predicting 'h'   P(correct)=0.1330  -> loss = 2.0178
pos 1: 'h' -> predicting 'e'   P(correct)=0.0281  -> loss = 3.5732
pos 2: 'e' -> predicting ' '   P(correct)=0.0240  -> loss = 3.7277
pos 3: ' ' -> predicting 'c'   P(correct)=0.0049  -> loss = 5.3222
pos 4: 'c' -> predicting 'a'   P(correct)=0.0147  -> loss = 4.2197
pos 5: 'a' -> predicting 't'   P(correct)=0.1441  -> loss = 1.9372
pos 6: 't' -> predicting ' '   P(correct)=0.0737  -> loss = 2.6077

Manual average of these 7:        3.3436
F.cross_entropy(logits, targets): 3.3436   <- exact match
```
That match is the key confirmation: **`F.cross_entropy` is literally this loop**, done in one
vectorized call, averaging the 7 per-position losses into the single number `.backward()` uses. Every
one of these 7 losses is high right now (untrained, random weights) -- training's whole job is to push
each one down.

### The mechanism, restated precisely
Cross-entropy does **not** compare row `'t'` against row `'h'`. When the input character is `'t'`, the
model pulls out row `'t'` -- 14 scores, one slot per possible next character, including a slot for
`'h'`. Softmax normalizes *within that row*; cross-entropy asks "how much probability mass, relative
to the other 13 scores in this same row, landed on the `'h'` slot?" Training's job is to raise that one
slot's value *relative to its row-mates* -- which, because softmax is zero-sum within a row, also means
pushing the other 13 scores in that row down. Row `'t'` and row `'h'` remain otherwise unrelated,
independently-updated rows; nothing here creates a relationship between different rows.

### Sanity-check print
```python
print(f"Expected loss: {-torch.log(torch.tensor(1.0/vocab_size)):.4f}")
```
An untrained model's weights are random, so it's guessing uniformly at random over the vocabulary. This
line computes the cross-entropy loss that *pure random guessing* would produce (`2.6391` for our
14-character vocab, `~4.17` for the real 65-character Shakespeare vocab). If the model's actual initial
loss matches this number, that confirms correct initialization -- nothing broken, biased, or leaking
information it shouldn't have yet.

---

## `generate()`, line by line

```python
logits = logits[:, -1, :]
```
In the notebook, generation always starts from a single seed context (`context = torch.zeros((1, 1))`,
i.e. `B=1`) and grows one character at a time. At each step, the full sequence so far is re-run through
the model, producing `logits` of shape `(1, T, 14)` where `T` grows every iteration. The bigram model
only ever cares about the single most recent character, so this slice discards every earlier
prediction and keeps only the last time step: `(1, T, 14) -> (1, 14)`.

```python
probs = F.softmax(logits, dim=-1)
idx_next = torch.multinomial(probs, num_samples=1)
```
`softmax` turns the 14 raw scores into probabilities summing to 1. `torch.multinomial` then samples one
character from that distribution -- a **weighted random draw**, not "always pick the highest score"
(that would be greedy decoding, which we're not using here -- it's why generated text differs run to
run even from the same trained model).

```python
idx = torch.cat((idx, idx_next), dim=1)
```
Appends the newly sampled character onto the sequence, so the next loop iteration's forward pass sees
one more character of context (even though the bigram model still only *uses* the last one).

---

## Training, traced step by step on the real example

### Real traced example: training on `xb="the cat"`, `yb="he cat "` for 6 steps
Continuing the same 7-character slice from above, watch the score in the `'h'` slot of row `'t'` (the
one predicting what follows `'t'`) evolve as we train on this one example repeatedly:
```
Step | loss   | score['t'->'h'] | P(next='h'|'t')
   0 | 3.3436 | +1.1676          | 0.1552
   1 | 3.1621 | +1.2662          | 0.1799
   2 | 2.9842 | +1.3643          | 0.2066
   3 | 2.8104 | +1.4617          | 0.2351
   4 | 2.6412 | +1.5579          | 0.2647
   5 | 2.4771 | +1.6527          | 0.2950
```
Loss falls, the `'h'` slot's score rises, `P(correct)` climbs from 13% -> 30% in 6 steps.

**The whole row moves, not just the target slot** (softmax cross-entropy pushes the correct slot up and
every other slot in the same row down, since they all compete for the same probability mass):
```
slot ' ': +0.479 -> +1.071      slot 'm': +1.307 -> +0.705
slot '.': +1.354 -> +0.752      slot 'n': +0.460 -> -0.136
slot 'a': -0.159 -> -0.752      slot 'o': +0.262 -> -0.333
slot 'c': -0.425 -> -1.016      slot 'r': -0.760 -> -1.349
slot 'd': +0.944 -> +0.345      slot 's': -2.046 -> -2.628
slot 'e': -0.185 -> -0.777      slot 't': -1.529 -> -2.114
slot 'g': +0.185 -> -0.409
slot 'h': +1.069 -> +1.653   <-- target slot, the only one that rises
```

**Rows for characters never seen in this batch are nearly, but not perfectly, frozen** -- e.g. row `'g'`
(never appears in `"the cat"`) shifts by a tiny amount even though its gradient was exactly zero:
```
before: [0.536, 0.525, 1.141, ...]
after:  [0.533, 0.521, 1.134, ...]   (tiny shrink toward zero)
```
This is **AdamW's weight decay** (see below) acting on every parameter every step regardless of
whether it received a real gradient this batch -- a small stabilizing pull toward zero, distinct from
the loss-driven signal that moves rows with actual gradient.

### The training loop, code line by line
```python
xb, yb = get_batch('train')
```
Grab `batch_size` (32, in the real training loop) random snippets. `xb` = current characters, `yb` =
the actual next characters that followed them in the real text -- the ground truth.

```python
logits, loss = model(xb, yb)
```
Forward pass: the model predicts, then compares its predictions to `yb` and computes a penalty score.
Early on (step 0) this loss is close to the random-guessing baseline -- the model knows nothing yet.

```python
optimizer.zero_grad(set_to_none=True)
```
PyTorch accumulates gradients by default (adds new ones to old ones) -- this wipes the slate before
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

### The mental model -- a blind hiker on a foggy mountain
Looking for the lowest point (the point of best predictions):
- **The hiker** = the model's current weights (its position on the "loss landscape")
- **The slope underfoot** = the gradient, computed by `loss.backward()` -- which direction is downhill
- **The guide dog** = the AdamW optimizer -- remembers recent steps and nudges the hiker smoothly
  toward the bottom rather than lurching around

### Why AdamW specifically
"Adaptive Moment Estimation with Weight decay" -- the default optimizer for nearly all modern deep
learning, including GPT-4-class models, because it combines three things:
1. **Momentum** -- like a heavy ball rolling downhill, it carries through small bumps/local
   irregularities in the loss landscape instead of getting stuck.
2. **Adaptive per-parameter learning rates** -- frequent characters (like `e`) get smaller, more
   careful updates; rare characters (like `z`) get larger updates so they actually learn from the few
   examples they appear in.
3. **Weight decay (the "W")** -- a gentle penalty that keeps weights small and smooth, shrinking every
   parameter slightly toward zero every step regardless of gradient (demonstrated above with row
   `'g'`) -- this is what prevents the model from over-fitting to any single pattern it's seen.

### What actually happens over 10,000 real training steps
Loss drops from ~4.7 (random, on the full 65-character Shakespeare vocab) to ~2.4-2.5. The model is
tuning its lookup table so that common character pairs (like `t` -> `h`) get higher scores. That's *all*
it can learn -- pairwise statistics, nothing longer-range.

---

## Why the output is still gibberish

Generated text (from the real Shakespeare-trained model): `CEng ay, Theyomy t blll,...` -- slightly more
English-shaped than pure noise, but still broken. **The bigram model has zero memory beyond one
character.**

Concrete example: to spell "t-h-e," after printing `t`, the model wipes its memory and looks only at
`h` to decide the next letter. But `h` alone is ambiguous -- "that" wants `a` next, "this" wants `i`,
"the" wants `e`. Since the model can't see that a `t` came before the `h`, it's guessing blind at every
single step, and the result is a chaotic mix of fragments from different words.

**This is exactly the problem self-attention solves** -- letting every character look back at
everything before it, not just the immediately preceding one. That's Part 9 onward.
