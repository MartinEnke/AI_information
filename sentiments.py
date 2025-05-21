text = "I absolutely love this phone, but the battery life is terrible"
positive_words = ["love", "amazing", "great", "awesome"]
negative_words = ["bad", "terrible", "horrible", "awful"]
# Convert text to lowercase and split into words
words = text.lower().split()

# Check if each word is in the positive words list
positive_matches = [word in positive_words for word in words]
negative_matches =[word in negative_words for word in words]
# Sum up the number of matches
score = sum(positive_matches) - sum(negative_matches)

if score > 0:
    print("Positive")
else:
    print("Negative or Neutral")