from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize
import matplotlib.pyplot as plt
from transformers import GPT2Tokenizer, GPT2Model
import torch

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token  # ✅ Fix padding issue
model = GPT2Model.from_pretrained("gpt2")


words = ["king", "queen", "man", "woman", "cat", "dog", "freedom", "love", "robot", "human"]
inputs = tokenizer(words, return_tensors="pt", padding=True, truncation=True, add_special_tokens=False)

with torch.no_grad():
    outputs = model(**inputs).last_hidden_state

embeddings = outputs[:, 0, :].numpy()
embeddings_normalized = normalize(embeddings)
reduced = PCA(n_components=2).fit_transform(embeddings_normalized)

plt.figure(figsize=(10, 7))
for i, word in enumerate(words):
    x, y = reduced[i]
    plt.scatter(x, y)
    plt.text(x + 0.01, y + 0.01, word, fontsize=12)

plt.title("2D Visualization of GPT-2 Token Embeddings")
plt.grid(True)
plt.show()


"""
🎯 First, what you're seeing:
You’re seeing a 2D projection (via PCA) of word embeddings from GPT-2.

Each dot is a word (token) like "king" or "queen"

The position reflects how similar their high-dimensional meanings are, as understood by the model

The words are labeled, and their relative distances tell you something about semantic similarity

🧠 But! PCA is a lossy projection
Remember:

GPT-2 works in 768 dimensions — your 2D map is like flattening a 3D sculpture onto a piece of paper.

So:

Some relationships will be preserved (e.g., "queen" near "woman")

Others may be distorted or compressed (e.g., "king" and "man" might look unrelated on the map but may still be similar in full 768D space)

✅ How to read the map:
✔️ 1. Words close together = semantically similar
"queen" and "woman" are close?
✅ GPT-2 sees them as related in some conceptual space — likely around gender, royalty, or role.

"robot" and "human" are near each other?
✅ GPT-2 might see them both as "entities", maybe connected by function or role in context.

❌ 2. Far apart = may not be unrelated in reality
"king" and "man" look far apart?
🛑 That doesn't mean GPT doesn't connect them — PCA just couldn’t preserve all 768D relationships.

You could use cosine similarity to measure how close they really are in original space.

🔍 Want to test similarity in 768D directly?
Here’s how to check how close "king" and "man" actually are:

python
Copy
Edit
from sklearn.metrics.pairwise import cosine_similarity
"""
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Assume you have `embeddings` from earlier (768D vectors)
idx_king = words.index("king")
idx_man = words.index("man")

cos_sim = cosine_similarity([embeddings[idx_king]], [embeddings[idx_man]])
print(f"Cosine similarity between 'king' and 'man': {cos_sim[0][0]:.3f}")

"""
A value close to 1.0 = very similar

Around 0.5 = somewhat related

Near 0.0 = not related

🧘 TL;DR:
What you see in the 2D plot	What it really means
Words near each other	Likely similar in meaning (in the original model)
Words far apart	May or may not be truly unrelated (PCA might distort)
Straight lines or axes?	No true axes — just compressed high-dimensional meaning
Trust the distances between labels	But don’t overinterpret exact X/Y direction

Would you like me to help you calculate cosine similarity between any pair from the plot — like king vs man vs queen?

"""