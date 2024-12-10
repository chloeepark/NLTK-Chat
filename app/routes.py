from flask import Blueprint, render_template, jsonify, request
from .chat import process_user_input

main = Blueprint('main', __name__)

@main.route('/')
def home():
    return render_template('chat.html')

@main.route('/chat', methods=['POST'])
def chat():
    user_input = request.json.get('text')
    if not user_input:
        return jsonify({"error": "No input provided"}), 400
    
    result = process_user_input(user_input)
    return jsonify(result)
