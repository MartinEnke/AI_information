import nltk
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Download stopwords
nltk.download("stopwords")
from nltk.corpus import stopwords

# Load English stopwords
stop_words = stopwords.words("english")

# Sample data
texts = [
    "I love this movie, it was amazing!",
    "This film was horrible and boring.",
    "Great acting and beautiful cinematography.",
    "Worst movie I've seen in years.",
    "It was just okay, not the best.",
    "I absolutely loved it!",
    "Fantastic story and visuals.",
    "Amazing and superb.",
    "It was bad and boring.",
    "Awful movie. Never again.",
]

labels = [1, 0, 1, 0, 0, 1, 1, 1, 0, 0]  # 1 = positive, 0 = negative sentiment

# Step-by-step pipeline
pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(lowercase=True, stop_words=stop_words, ngram_range=(1, 2))),
    ("clf", LogisticRegression())
])

# Train the model
pipeline.fit(texts, labels)

# Test it
print(pipeline.predict(["A wonderful and emotional performance."]))  # Expected: [1]
print(pipeline.predict(["Terrible plot and bad acting."]))           # Expected: [0]



from sklearn.feature_extraction.text import TfidfVectorizer

docs = ["The dog barked.", "The cat sat on the mat.", "The cat sat on the bed."]
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(docs)

print(vectorizer.get_feature_names_out())
print(X.toarray())