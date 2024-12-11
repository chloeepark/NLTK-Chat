# Code handling the main logic for the chatbot
from .utils import detect_language, preprocess_input, analyze_sentiment
import openai
from dotenv import load_dotenv
import os

load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")

def process_user_input(user_input):
    # Language detection
    language = detect_language(user_input)
    if language not in ["en", "ko", "es", "fr", "zh", "de", "ru", "ar", "hi", "ja", "it", "pt", "nl", "sv", "cs", "th", "vi", "tr", "uk"]:
        language = "en"

    # Text Preprocessing
    processed_tokens = preprocess_input(user_input)
    processed_text = ' '.join(processed_tokens) 
    print(f"Original Text: {user_input}")
    print(f"Processed Text: {processed_text}")

    # Sentiment Analysis
    sentiment_scores = analyze_sentiment(user_input)
    sentiment = "neutral"
    if sentiment_scores["compound"] > 0.1:
        sentiment = "positive"
    elif sentiment_scores["compound"] < -0.1:
        sentiment = "negative"
    
    # Emotion-based system message configuration
    if sentiment == "positive":
        system_message = f"You are an assistant that provides cheerful and encouraging responses in {language}."
    elif sentiment == "negative":
        system_message = f"You are an assistant that offers empathetic and supportive advice in {language}."
    else:
        system_message = f"You are an assistant that provides clear and professional information in {language}."

    try:
        # Call ChatGPT API
        gpt_response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": f"Original: {user_input}\nProcessed: {processed_text}"}
            ]
        )
        response = gpt_response["choices"][0]["message"]["content"]
    except Exception as e:
        response = f"An error occurred while connecting to the API: {str(e)}"

    return {
        "language": language,
        "sentiment": sentiment,
        "response": response
    }
