import nltk
import spacy
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from textblob import TextBlob
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import string

# -------------------- Setup --------------------
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Load spaCy English model
nlp = spacy.load("en_core_web_sm")

# -------------------- 1. Text Preprocessing --------------------
print("🔤 TEXT PREPROCESSING (NLTK)\n")
text = "Natural Language Processing is fun and powerful!"
print("Original Text:", text)

# Tokenization
tokens = word_tokenize(text)
print("\n1️⃣ Tokens:", tokens)

# Stopword Removal
stop_words = set(stopwords.words('english'))
filtered = [w for w in tokens if w.lower() not in stop_words]
print("2️⃣ After Removing Stopwords:", filtered)

# Lemmatization
lemmatizer = WordNetLemmatizer()
lemmatized = [lemmatizer.lemmatize(word) for word in filtered]
print("3️⃣ Lemmatized:", lemmatized)

# -------------------- 2. Sentiment Analysis --------------------
print("\n😊 SENTIMENT ANALYSIS (TextBlob)\n")
sent = "I love learning new things, but sometimes it gets overwhelming."
blob = TextBlob(sent)
print(f"Text: {sent}")
print("Polarity:", blob.sentiment.polarity)
print("Subjectivity:", blob.sentiment.subjectivity)

# -------------------- 3. Named Entity Recognition --------------------
print("\n📍 NAMED ENTITY RECOGNITION (spaCy)\n")
ner_text = "Apple is looking at buying a U.K. startup for $1 billion."
doc = nlp(ner_text)
print("Text:", ner_text)
for ent in doc.ents:
    print(f"{ent.text} -> {ent.label_}")

# -------------------- 4. Text Classification --------------------
print("\n🧠 TEXT CLASSIFICATION (Scikit-learn)\n")

# Expanded training data with clear positive & negative examples
texts = [
    "I love this product",
    "This is an amazing place",
    "I love this book",
    "Such a wonderful day",
    "Absolutely amazing",
    "I hate this movie",
    "This is a terrible book",
    "What a horrible experience",
    "This was a bad idea",
    "I really dislike this",
    "Awful performance"
    
]
labels = [1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0]

# Preprocess text by lowercasing and removing punctuation
def preprocess(text):
    text = text.lower()
    return text.translate(str.maketrans('', '', string.punctuation))

texts = [preprocess(t) for t in texts]

# Vectorization with TfidfVectorizer
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(texts)

# Train Model
model = MultinomialNB()
model.fit(X, labels)

print("Enter sentences to classify sentiment (type 'exit' to quit):")
while True:
    user_input = input("Input: ").strip()
    if user_input.lower() == 'exit':
        print("Exiting sentiment classification.")
        break
    processed = preprocess(user_input)
    X_test = vectorizer.transform([processed])
    pred = model.predict(X_test)[0]
    sentiment = "Positive" if pred == 1 else "Negative"
    print(f"Sentiment: {sentiment}\n")
