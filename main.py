import nltk
from nltk.data import find
from app import create_app

def ensure_nltk_data():
    try:
        find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')

    try:
        find('sentiment/vader_lexicon')
    except LookupError:
        nltk.download('vader_lexicon')

ensure_nltk_data()

app = create_app()

if __name__ == '__main__':
    app.run(debug=True)
