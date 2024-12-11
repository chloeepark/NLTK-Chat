# Contains utility functions
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.corpus import stopwords
from langdetect import detect, DetectorFactory
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from langdetect import detect, DetectorFactory
from langdetect.lang_detect_exception import LangDetectException

DetectorFactory.seed = 0
analyzer = SentimentIntensityAnalyzer()

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Sentiment analysis function
def analyze_sentiment(text):
    sentiment_scores = analyzer.polarity_scores(text)
    return sentiment_scores


stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()

# Language detection function
def detect_language(text):  
    try:
        if not text.strip():
            return "en"
        language = detect(text)
    except LangDetectException:
        language = "en"
    return language

# Tokenization
def tokenize_text(text):
    tokens = word_tokenize(text)
    return tokens

# Stopword Removal
stop_words = set(stopwords.words('english'))

def remove_stopwords(tokens):
    filtered = [token for token in tokens if token.lower() not in stop_words]
    return filtered

# Stemming
def stem_tokens(tokens):
    stemmed = [stemmer.stem(token) for token in tokens]
    return stemmed

# Lemmatization
def lemmatize_tokens(tokens):
    lemmatized = [lemmatizer.lemmatize(token) for token in tokens]
    return lemmatized

# Preprocessing Pipeline
def preprocess_input(text):
    tokens = tokenize_text(text)            # Tokenization
    tokens = remove_stopwords(tokens)       # Stopword Removal
    tokens = lemmatize_tokens(tokens)       # Lemmatization
    return tokens
