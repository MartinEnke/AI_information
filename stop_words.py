import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
import re

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')

# Sample text
text = "My dog is named LeBron and enjoys chewing sticks!"
# Lowercasing
text_lower = text.lower()
print("Lowercased text:", text_lower)

# Removing punctuation
text_no_punct = re.sub(r'[^\w\s]', '', text_lower)
print("Text without punctuation:", text_no_punct)

# Tokenization
words = nltk.word_tokenize(text_no_punct)
print("Tokenized words:", words)

# Removing stop words
stop_words = set(stopwords.words('english'))
words = [word for word in stop_words if not word in ("a", "an", "the", "and", "but", "or", "to", "in", "on", "at", "with", "he", "she", "it", "they", "is", "am", "are", "was", "were", "be", "being", "been")]

# update code so that words_no_stop don't have any stop words
words_no_stop = words

# Stemming
ps = PorterStemmer()
words_stemmed = [ps.stem(word) for word in words_no_stop]
print("Stemmed words:", words_stemmed)

# Lemmatization
lemmatizer = WordNetLemmatizer()
words_lemmatized = [lemmatizer.lemmatize(word) for word in words_no_stop]
print("Lemmatized words:", words_lemmatized)

