import spacy
nlp = spacy.load("en_core_web_sm")
doc = nlp("Tony Kross visited Athens in August 2012.")
for ent in doc.ents:
  print(ent.text, ent.label_)




"""
Try to use the given code to find all named entities in the “preprocessed_romeo_and_juliet.txt” with the DATE label.
"""
import spacy

# Load SpaCy's English language model
nlp = spacy.load("en_core_web_sm")

# Load the preprocessed text
with open("romeo_and_juliet.txt", "r", encoding="utf-8") as file:
    text = file.read()

# Process the text with SpaCy
doc = nlp(text)

# Extract all dates (DATE entities)
dates = [ent.text for ent in doc.ents if ent.label_ == "DATE"]

# Remove duplicates while preserving order
unique_dates = list(dict.fromkeys(dates))

# Print all unique dates
print("Dates found in Romeo and Juliet:")
for date in unique_dates:
    print(date)


"""🧠 So... how does this actually work?
1️⃣ spacy.load("en_core_web_sm")
This loads a pretrained NLP pipeline, which includes:

Component	Role
Tokenizer	Splits text into tokens (words/punct)
Tagger	Assigns POS tags (noun, verb, etc.)
Parser	Builds syntactic structure
NER	Identifies named entities

en_core_web_sm is a small English model trained on news, web, and Wikipedia-style text.

2️⃣ doc = nlp("Tony Kross visited Athens in August 2012.")
This runs your text through the full pipeline:

Tokenization

POS tagging

Dependency parsing

NER

➡️ Behind the scenes, NER uses a trained statistical model (neural network) that:

Sees word patterns

Uses context

Applies learned entity boundaries and labels

🔍 So how does it know?
Let’s look at what’s learned:

→ "Tony Kross" → PERSON
Capitalized, proper noun

Common name patterns (like first + last)

Appears near verbs like “visited” → likely subject

→ "Athens" → GPE (Geopolitical Entity)
It knows from training that "Athens" is often a city name

Appears after verbs like “visited” — again, context helps

→ "August 2012" → DATE
Recognized as a temporal phrase

Pattern matching like [Month Year]

Learned via examples like "in May 2020", "on July 4" etc.

🎓 How was it trained?
The NER model was trained on labeled examples like:

plaintext
Copy
Edit
"Barack Obama was born in Hawaii."  
→ [('Barack Obama', 'PERSON'), ('Hawaii', 'GPE')]
The model:

Learns what spans are entities

And what types (PERSON, ORG, DATE, GPE, etc.)

🧘 TL;DR: What does what?
Code Part	Role
spacy.load("en_core_web_sm")	Loads pretrained model with tokenizer + NER
nlp(text)	Runs the full NLP pipeline
doc.ents	Gives you the named entities detected
ent.text, ent.label_	The entity string and its type (like DATE)

🔍 Want to peek under the hood?
We can:

Visualize NER with spacy.displacy

List all entity types supported (like ORG, MONEY, NORP)

Compare what a larger model (like en_core_web_trf) sees differently

Just say the word! 😄







You said:
interesting. but that:  trained statistical model (neural network) is again sth else than the dimensions thingi?


ChatGPT said:
Fantastic connection, Lee — and yes! You're right to ask this.

Let me break it down crystal clear:

🧠 What’s the difference?
🔸 Word Embeddings (like those 768 dimensions)
These are vectors: they represent the meaning of a word/token

They live inside a matrix and are learned during training

Example: "king" → [0.2, -1.3, ..., 0.9] (768D in GPT-2)

Used as input to a neural network

➡️ Think of them like:

“How we represent the words mathematically”

🔸 Neural Network (like spaCy's NER model)
A system of layers and weights that takes those vectors and learns patterns

Example: "Tony Kross" in context → the model classifies it as a PERSON

The network looks at the sequence, position, context, and uses learned weights to make predictions

➡️ Think of it like:

“The brain that makes decisions based on those vectors”

🧬 How they work together
Let’s walk through what happens inside spaCy (or any NLP model):

"Tony" → vector A
"Kross" → vector B
"visited" → vector C
etc...

These vectors (like 300D or 768D) are passed into a neural network

The network processes them — using layers, weights, and activation functions — and says:

→ "Tony Kross" = PERSON
→ "Athens" = GPE

🧘 TL;DR:
Concept	What it is	Analogy
Embedding	A vector of numbers (meaning of word)	How you represent a word numerically
Neural Network	A model that takes vectors as input and makes decisions	A brain processing information

✅ Word embeddings are input data
✅ Neural networks are the decision-making machine"""