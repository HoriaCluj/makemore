# makemore

A step-by-step reproduction of Andrej Karpathy's [makemore](https://github.com/karpathy/makemore): a character-level language model that learns from ~32k names ([names.txt](names.txt)) and generates new, name-like strings.

Everything lives in [makemore.ipynb](makemore.ipynb), with explanations alongside the code.

## Implementation

Both models are **bigram** models: they predict the next character using only the current one. Each name is padded with a special `.` token marking its start and end (`.emma.` → `.e`, `em`, `mm`, `ma`, `a.`), which gives a vocabulary of 27 tokens (26 letters + `.`).

### 1. Counting bigrams

1. Count every bigram in the dataset into a `27×27` matrix `N`, where `N[i, j]` is how often character `j` follows character `i`.
2. Add 1 to every count (smoothing, so no bigram has probability 0), then normalise each row to get the probability matrix `P`.
3. **Sampling:** start at `.`, draw the next character from row `P[ix]` with `torch.multinomial`, and repeat until `.` comes up again.
4. **Evaluation:** the average negative log-likelihood (NLL) of the training bigrams under `P`, which is **≈ 2.45**.

### 2. Single-layer neural network

The same model, but learned by gradient descent instead of counting:

1. Build the training set of `(x, y)` pairs, one per bigram, and one-hot encode `x` into 27-dimensional vectors.
2. **Forward pass:** `logits = xenc @ W`, where `W` is a single `27×27` weight matrix with no bias and no hidden layer. Softmax (`exp`, then normalise each row) turns the logits into next-character probabilities.
3. **Loss:** the mean NLL of the correct next characters, plus a small L2 penalty on `W` that plays the same role as the count smoothing.
4. **Backward pass and update:** reset `W.grad`, call `loss.backward()`, then `W.data += -lr * W.grad`.

Because a one-hot input simply selects one row of `W`, `exp(W)` ends up playing the role of the count matrix `N`. After training, the network converges to roughly the **same loss (≈ 2.46)** and generates the same kind of names as the counting model. The neural approach matters because it scales to longer contexts, where counting tables blow up.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows  (macOS/Linux: source .venv/bin/activate)
pip install -r requirements.txt
```

Then open `makemore.ipynb` in Jupyter or VS Code, select the `.venv` kernel, and run all the cells.

> If you install packages while a kernel is running, restart the kernel. For example, importing `torch` before `numpy` was installed causes `RuntimeError: Numpy is not available`.
