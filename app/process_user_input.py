import nltk
from nltk.classify import NaiveBayesClassifier
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

def create_training_data():
    training_data = [
        ("hello", "greeting"),
        ("hi there", "greeting"),
        ("how are you?", "greeting"),
        ("what is your name?", "question"),
        ("how does this work?", "question"),
        ("can you help me?", "question"),
        ("this is amazing", "feedback"),
        ("I love this app", "feedback"),
        ("this is not working", "complaint"),
        ("I am unhappy with the service", "complaint"),
        ("thank you", "gratitude"),
        ("thanks a lot", "gratitude"),
    ]
    return training_data

def preprocess_data(training_data):
    stop_words = set(stopwords.words('english'))

    def tokenize_and_filter(sentence):
        tokens = word_tokenize(sentence.lower())
        return [word for word in tokens if word.isalnum() and word not in stop_words]

    return [({word: True for word in tokenize_and_filter(text)}, label) for text, label in training_data]

def train_classifier():
    training_data = create_training_data()
    processed_data = preprocess_data(training_data)
    classifier = NaiveBayesClassifier.train(processed_data)
    return classifier

def classify_input(user_input, classifier):
    stop_words = set(stopwords.words('english'))
    tokens = word_tokenize(user_input.lower())
    filtered_tokens = {word: True for word in tokens if word.isalnum() and word not in stop_words}
    return classifier.classify(filtered_tokens)

if __name__ == "__main__":
    chatbot_classifier = train_classifier()

    user_inputs = [
        "hello",
        "can you help me?",
        "this is not working",
        "thank you",
    ]

    for user_input in user_inputs:
        category = classify_input(user_input, chatbot_classifier)
        print(f"Input: {user_input} -> Category: {category}")
