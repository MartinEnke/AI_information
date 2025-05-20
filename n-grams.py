import nltk
from nltk.util import ngrams
from collections import Counter

nltk.download('punkt')

# Sample text data
#text = "The quick brown fox jumps over the lazy dog"
with open("RAG.txt", "r") as file:
    text = file.read()


# Tokenize the text into words
tokens = text.lower().split()

# Generate Unigrams (1-gram)
unigrams = list(ngrams(tokens, 1))
print("Unigrams:")
print(unigrams)

# Generate Bigrams (2-gram)
bigrams = list(ngrams(tokens, 2))
print("\nBigrams:")
print(bigrams)

# Generate Trigrams (3-gram)
trigrams = list(ngrams(tokens, 3))
print("\nTrigrams:")
print(trigrams)

# Count frequency of each n-gram (for demonstration)
unigram_freq = Counter(unigrams)
most_common_unigram = unigram_freq.most_common(1)
print(most_common_unigram)
bigram_freq = Counter(bigrams)
most_common_bigram = bigram_freq.most_common(1)
print(most_common_bigram)
trigram_freq = Counter(trigrams)
most_common_trigram = trigram_freq.most_common(1)
print(most_common_trigram)
# Print frequencies (optional)
print("\nUnigram Frequencies:")
print(unigram_freq)
print("\nBigram Frequencies:")
print(bigram_freq)
print("\nTrigram Frequencies:")
print(trigram_freq)


'''
.most_common() is a built-in method of Python's Counter class from the collections module. It's incredibly useful for frequency analysis.

🔍 What is Counter?
Counter is a subclass of dict that counts the frequency of elements in a list or iterable.

Example:

python
Copy
Edit
from collections import Counter

words = ['a', 'b', 'a', 'c', 'b', 'a']
counter = Counter(words)

print(counter)
# Output: Counter({'a': 3, 'b': 2, 'c': 1})
💡 .most_common(n)
You can call .most_common(n) on a Counter to get the n most frequent items, sorted by frequency.

python
Copy
Edit
top_2 = counter.most_common(2)
print(top_2)
# Output: [('a', 3), ('b', 2)]
So when you did:

python
Copy
Edit
trigram_freq = Counter(trigrams)
trigram_freq.most_common(1)
You got:

python
Copy
Edit
[(('the', 'fair', 'juliet'), 9)]
Which means:

The trigram ('the', 'fair', 'juliet') appeared 9 times

And it was the most common trigram in your text

🧠 Why this is awesome:
You can use it for:

Most common words, bigrams, trigrams

Most clicked URLs

Most frequent errors

Anything where counting and ranking is useful

Let me know if you want to do something like:

Visualize the top 10 trigrams in a bar chart

Filter out common stop word trigrams

Use bigrams for text classification
'''