import nltk
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

with open("romeo&juliet.txt", "r") as file:
    text = file.read()

tokens = nltk.word_tokenize(text)
pos_tags = nltk.pos_tag(tokens)

# Find NN followed by VB
for i in range(len(pos_tags) - 1):
    word1, tag1 = pos_tags[i]
    word2, tag2 = pos_tags[i + 1]

    if tag1.startswith("NN") and tag2.startswith("VB"):
        print(f"{word1} ({tag1}) → {word2} ({tag2})")