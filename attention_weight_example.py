"""Now here’s a simple Python code demo to show how attention
works in practice with a small sentence using numpy.
🧠 What it shows:
It creates fake embeddings for words.

Applies attention: each word computes a weighted average over all others.
Prints out the attention weights and the resulting vector for each word.
"""
import numpy as np


# Example sentence
tokens = ["The", "animal", "crossed", "the", "road"]

# Create random "embeddings" for simplicity
np.random.seed(42)
embeddings = np.random.rand(len(tokens), 4)  # (tokens x embedding_dim)

# Create simple Q, K, V (in real transformers, these are learned projections)
Q = embeddings @ np.random.rand(4, 4)
K = embeddings @ np.random.rand(4, 4)
V = embeddings @ np.random.rand(4, 4)

# Calculate attention scores
scores = Q @ K.T
scores = scores / np.sqrt(K.shape[-1])  # scale
weights = np.exp(scores) / np.sum(np.exp(scores), axis=1, keepdims=True)  # softmax

# Apply attention to values
attended = weights @ V

# Show results
for i, token in enumerate(tokens):
    print(f"Token: {token}")
    print(f"Attention Weights: {np.round(weights[i], 2)}")
    print(f"Attended Vector: {np.round(attended[i], 2)}\n")



"""✅ The heatmap you saw:
✔️ Shows real attention weights — computed from real math logic
(but based on fake embeddings for simplicity).

How?
The code simulates real Transformer behavior:

Generates random token embeddings.

Computes Q, K, V vectors.

Calculates scaled dot-product attention.

Applies softmax to get normalized attention weights.

So even though the token embeddings are random, the attention weights themselves 
are real — just like a real Transformer would compute them.

✅ They represent how much each token "attends to" the others, 
based on the dot products of their query/key vectors.

❗ The vectors and meaning:
❌ The attention weights in this toy example don't "mean" anything semantically, because...

We used np.random.rand() for the embeddings.

There’s no learned model, no trained weights, and no language knowledge.

So while the attention math is real, it’s acting on random noise.

🧠 In a real Transformer:
The embeddings come from a trained vocabulary, and the Q/K/V matrices are learned parameters.

The resulting attention weights truly reflect relationships like:

Pronoun resolution (e.g. "it" → "animal")

Verb-object linking

Long-range dependencies

👁️ Summary:
Component	                    Our Example	    Real Transformer	 Notes
Embeddings	                    ❌ Random	    ✅ Learned	         Real models use pretrained word vectors
Attention weights logic (QKV)	✅ Real math	✅ Real math	     We used the real attention formula
Meaningful weights	            ❌ No	        ✅ Yes	             Needs training and real language data

Let me know if you’d like to see this with actual pre-trained vectors (like GloVe or BERT), 
or want to extend it into a full mini transformer!
"""

