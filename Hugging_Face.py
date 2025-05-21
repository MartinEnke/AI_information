from transformers import pipeline

# 2. Load a pre-trained text generation pipeline
generator = pipeline("text-generation", model="gpt2")

# 3. Provide a prompt
prompt = "The human voice is the most complex"

# 4. Generate text
outputs = generator(prompt, max_length=50, truncation=True, num_return_sequences=1)

# 5. Print the generated text
for output in outputs:
    print(output["generated_text"])


'''
🧠 Why Is the Output Repetitive?
The output:

"In order to understand... In order to understand... The human voice is the voice of many people..."

This repetition happens because:

GPT-2 is an older model (released in 2019) and not as good at coherence as newer ones like GPT-3.5 or GPT-4.

No stop sequences or penalties are applied:

You can tweak repetition_penalty or no_repeat_ngram_size to reduce this.

Short context: The prompt isn’t very detailed or directive, so GPT-2 falls back into generic patterns.

Low creativity setting (default temperature=1.0, but still behaves conservatively unless nudged).

🔧 How to Improve the Results
Here’s a better setup to reduce repetition and increase quality:
outputs = generator(
    prompt,
    max_length=100,
    num_return_sequences=1,
    do_sample=True,
    temperature=0.9,
    top_k=50,
    top_p=0.95,
    repetition_penalty=1.2,
    no_repeat_ngram_size=3
)
temperature: Controls randomness (0.7–1.0 is good for creativity)

top_k and top_p: Sampling filters to prevent “dull” completions

repetition_penalty: Makes it less likely to repeat phrases

no_repeat_ngram_size: Avoids repeating n-grams (e.g. bigrams or trigrams)

📘 TL;DR
Concept	What's Happening
Model	You're using GPT-2 (older, weaker than ChatGPT)
Output quality	Decent but repetitive, due to default settings and model limitations
Why it repeats	No penalty for repetition, short/dull prompt, model tendency
How to improve	Use repetition_penalty, top_k, top_p, or switch to a newer model like gpt2-medium, gpt2-xl, or even LLaMA/ChatGPT via Hugging Face


Option 1: 🔌 Run on CPU instead of GPU
Force the model to run on CPU like this:

generator = pipeline("text-generation", model="gpt2", device=-1)
This avoids the whole MPS (GPU) backend and will not crash — it’s slower, but safe.

Option 2: 🧠 Reduce load
Use lower max_new_tokens and less sampling complexity:

outputs = generator(
    prompt,
    max_new_tokens=80,  # instead of 256
    do_sample=True,
    temperature=0.8,
    top_k=40,
    top_p=0.9,
    repetition_penalty=1.1,
    no_repeat_ngram_size=2
)
You can also remove max_length completely — it conflicts with max_new_tokens.

Option 3: 🧪 Try a smaller model
Try a smaller or distilled version of GPT-2:

generator = pipeline("text-generation", model="distilgpt2", device=-1)
It’s lighter, uses less memory, and still performs pretty well.

✅ Bonus: Detect & Warn About MPS Issues
You can also include a quick runtime check to switch to CPU if MPS is unstable:

import torch
from transformers import pipeline

device = "mps" if torch.backends.mps.is_available() else "cpu"
device_id = 0 if device == "mps" else -1

generator = pipeline("text-generation", model="gpt2", device=device_id)
TL;DR
Your crash is caused by GPU memory exhaustion on macOS with Metal/MPS.

Use device=-1 to force CPU and prevent crashing.

Reduce max_new_tokens or try distilgpt2 for lower memory use.

Remove max_length if you're using max_new_tokens.
'''

