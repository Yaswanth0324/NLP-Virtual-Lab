import os
import logging
from flask import Flask, render_template, request, jsonify
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy.orm import DeclarativeBase
from werkzeug.middleware.proxy_fix import ProxyFix
from flask_login import UserMixin
from datetime import datetime
import json
import random

# --- New Imports for .env file and AI Generation ---
# Make sure to install the required libraries:
# pip install python-dotenv google-generativeai requests
from dotenv import load_dotenv

try:
    import requests
except ImportError:
    requests = None

try:
    import google.generativeai as genai
    google_ai_available = True
except ImportError:
    genai = None
    google_ai_available = False
# --- End of New Imports ---

# Load environment variables from .env file
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.DEBUG)

class Base(DeclarativeBase):
    pass

db = SQLAlchemy(model_class=Base)

# Create the app
app = Flask(__name__)
app.secret_key = os.environ.get("SESSION_SECRET", "dev-secret-key-change-in-production")
app.wsgi_app = ProxyFix(app.wsgi_app, x_proto=1, x_host=1)

# Configure the database
app.config["SQLALCHEMY_DATABASE_URI"] = os.environ.get("DATABASE_URL", "sqlite:///nlp_lab.db")
app.config["SQLALCHEMY_ENGINE_OPTIONS"] = {
    "pool_recycle": 300,
    "pool_pre_ping": True,
}

# Initialize the app with the extension
db.init_app(app)


# --- Start of Updated Chatbot Code ---

def search_internet(query):
    """
    Searches the internet for a given query using the DuckDuckGo API.
    """
    if not requests:
        logging.warning("The 'requests' library is not installed. Internet search is disabled.")
        return "Internet search is unavailable because the 'requests' library is missing."

    try:
        search_url = "https://api.duckduckgo.com/"
        params = {
            "q": query,
            "format": "json",
            "no_html": 1,
            "skip_disambig": 1
        }
        response = requests.get(search_url, params=params, timeout=5)
        response.raise_for_status()
        data = response.json()

        results = []
        if data.get("AbstractText"):
            results.append(data["AbstractText"])
        
        related_topics = data.get("RelatedTopics", [])
        for topic in related_topics:
            if topic.get("Text"):
                results.append(topic["Text"])

        if not results:
            return "I couldn't find any direct information on that topic. Could you try rephrasing the question?"
            
        return " ".join(results[:3])

    except requests.exceptions.RequestException as e:
        logging.error(f"Error during internet search: {e}")
        return "Sorry, I'm having trouble connecting to the internet right now."
    except Exception as e:
        logging.error(f"An unexpected error occurred during search: {e}")
        return "An unexpected error occurred while searching."


def generate_gemini_response(query, context):
    """
    Generates a response using the Google Gemini API, based on the provided context.
    """
    if not google_ai_available:
        return "The Google AI backend is not installed. Please run 'pip install google-generativeai'."

    # The API key is now loaded securely from the .env file
    api_key = os.environ.get("GEMINI_API_KEY")

    if not api_key:
        logging.warning("Gemini API key not found in .env file. Using fallback response.")
        return "The AI assistant is not configured. Please provide a Gemini API key in a .env file."

    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-1.5-flash-latest')
        
        # --- ** CORRECTED: Language Detection Logic ** ---
        lower_query = query.lower()

        # 1. Detect Language (checking for more specific terms first)
        requested_language = 'Python' # Default
        if 'javascript' in lower_query or ' js ' in f' {lower_query} ' or 'node.js' in lower_query:
            requested_language = 'JavaScript'
        elif 'java' in lower_query:
            requested_language = 'Java'
        elif 'c++' in lower_query or 'cpp' in lower_query:
            requested_language = 'C++'
        elif ' r ' in f' {lower_query} ': # Pad with spaces to match ' r ' safely
            requested_language = 'R'


        # 2. Detect Intent more precisely
        is_code_request = 'code' in lower_query or 'implement' in lower_query or 'snippet' in lower_query
        is_explanation_request = 'explain' in lower_query or 'what is' in lower_query or 'describe' in lower_query or 'how does' in lower_query
        is_code_only_request = 'only code' in lower_query or 'just the code' in lower_query
        is_example_request = 'example' in lower_query

        # 3. Select Prompt based on detected intent
        prompt = ""
        language_specific_instruction = ""
        if requested_language == 'JavaScript':
            language_specific_instruction = "The JavaScript code should be modern and functional, suitable for a Node.js environment. Provide standalone functions where possible. If a common library is standard for the task (e.g., from npm), mention the package and show its usage."

        if is_code_only_request:
            prompt = f"""You are a code generation assistant.
Provide ONLY the {requested_language} code for the following query in a single markdown code block. {language_specific_instruction}
Do NOT add any explanation, introduction, or conclusion.

User's Query: "{query}"
"""
        elif is_code_request and not is_explanation_request:
             prompt = f"""You are a code generation assistant.
Provide a functional {requested_language} code example for the user's query. {language_specific_instruction}
Use the search context for guidance, but generate a standard, working example even if the context is sparse.
Enclose the code in a single markdown code block. Do not add long explanations before or after the code.

Search Context: "{context}"
User's Query: "{query}"
"""
        elif is_explanation_request and not is_code_request:
            prompt = f"""You are an NLP expert. The user is asking for an explanation.
Based on the provided search context, provide a detailed explanation for the user's query.
Use markdown for formatting (bolding, bullet points).
If the user asks for an "example", provide a clear, textual example, not a code snippet.
Do NOT include any code examples.

Search Context: "{context}"
User's Query: "{query}"
"""
        else: # Default case (e.g., "NER", "tokenization code example") - provide both explanation and code
            prompt = f"""You are an expert NLP assistant. Your goal is to provide a direct, helpful, and well-structured answer.

**Instructions:**
1.  Start with a clear, concise explanation of the topic requested by the user.
2.  After the explanation, provide a functional code example in **{requested_language}**. {language_specific_instruction}
3.  Use markdown for formatting (bolding, bullet points, and code blocks).
4.  Do not apologize or say you cannot provide an answer. Use the context to formulate the best possible response.

---
**Search Context:**
{context}
---

**User's Query:**
{query}
---

**Your Expert Answer:**
"""

        response = model.generate_content(prompt)
        return response.text.strip()

    except Exception as e:
        logging.error(f"Error generating Gemini response: {e}")
        return f"Sorry, I encountered an API error: {str(e)}"


@app.route('/api/chatbot', methods=['POST'])
def chatbot_api():
    """
    Handles chat messages by searching the internet and generating a response.
    """
    try:
        data = request.get_json()
        message = data.get('message', '').strip()
        session_id = data.get('session_id', 'default')

        if not message:
            return jsonify({'error': 'Message is required'}), 400

        # 1. Search the internet for context
        search_context = search_internet(message)
        
        # 2. Generate a response based on the search context using Gemini
        final_response = generate_gemini_response(message, search_context)

        return jsonify({'response': final_response, 'session_id': session_id})
        
    except Exception as e:
        logging.error(f"Error in chatbot API: {str(e)}")
        return jsonify({'error': f"Server Error: {str(e)}"}), 500

# --- End of Updated Chatbot Code ---


# Models
class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(64), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(256))
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    progress = db.relationship('UserProgress', backref='user', lazy=True)

class UserProgress(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    module_name = db.Column(db.String(100), nullable=False)
    completed = db.Column(db.Boolean, default=False)
    score = db.Column(db.Integer, default=0)
    attempts = db.Column(db.Integer, default=0)
    last_attempt = db.Column(db.DateTime, default=datetime.utcnow)
    __table_args__ = (db.UniqueConstraint('user_id', 'module_name'),)

class QuizQuestion(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    module_name = db.Column(db.String(100), nullable=False)
    question = db.Column(db.Text, nullable=False)
    options = db.Column(db.JSON, nullable=False)
    correct_answer = db.Column(db.Integer, nullable=False)
    explanation = db.Column(db.Text)

# Lab modules configuration
LAB_MODULES = {
    'text_preprocessing': {
        'title': 'Text Preprocessing',
        'description': 'Learn tokenization, lowercasing, punctuation removal, and stopword removal',
        'icon': 'file-text'
    },
    'pos_tagging': {
        'title': 'Part-of-Speech Tagging',
        'description': 'Identify grammatical categories of words',
        'icon': 'tag'
    },
    'ngram_modeling': {
        'title': 'N-Gram Modeling',
        'description': 'Analyze word sequences and context',
        'icon': 'layers'
    },
    'named_entity_recognition': {
        'title': 'Named Entity Recognition',
        'description': 'Extract entities like names, places, and organizations',
        'icon': 'user'
    },
    'sentiment_analysis': {
        'title': 'Sentiment Analysis',
        'description': 'Determine emotional tone of text',
        'icon': 'heart'
    },
    'text_classification': {
        'title': 'Text Classification',
        'description': 'Categorize text into predefined classes',
        'icon': 'folder'
    },
    'word_embeddings': {
        'title': 'Word Embeddings',
        'description': 'Convert text to numerical representations',
        'icon': 'grid'
    },
    'chunking': {
        'title': 'Chunking & Parsing',
        'description': 'Group words into meaningful phrases',
        'icon': 'git-branch'
    }
}

# Routes
@app.route('/')
def index():
    return render_template('index.html', modules=LAB_MODULES)

@app.route('/readme')
def readme():
    return render_template('readme.html')

@app.route('/chatbot')
def chatbot():
    return render_template('chatbot.html')


@app.route('/lab/<module_name>')
def lab(module_name):
    if module_name not in LAB_MODULES:
        return "Lab module not found", 404
    module_info = LAB_MODULES[module_name]
    return render_template('lab.html', module_name=module_name, module_info=module_info)

@app.route('/quiz/<module_name>')
def quiz(module_name):
    if module_name not in LAB_MODULES:
        return "Quiz not found", 404
    module_info = LAB_MODULES[module_name]
    return render_template('quiz.html', module_name=module_name, module_info=module_info)

@app.route('/api/process', methods=['POST'])
def process_text():
    try:
        from nlp_processor import NLPProcessor
        nlp_processor = NLPProcessor()

        data = request.get_json()
        text = data.get('text', '').strip()
        operation = data.get('operation', '').strip()

        logging.debug(f"[PROCESS] Text: {text[:50]}... | Operation: {operation}")

        if not text or not operation:
            return jsonify({'error': 'Text and operation are required'}), 400

        result = nlp_processor.process(text, operation)
        return jsonify(result)

    except Exception as e:
        logging.exception("Error processing text:")
        return jsonify({'error': f"Server Error: {str(e)}"}), 500

@app.route('/api/quiz/<module_name>')
def get_quiz_questions(module_name):
    try:
        from nlp_processor import NLPProcessor
        nlp_processor = NLPProcessor()
        
        # Get questions from the NLP processor
        questions = nlp_processor.get_quiz_questions(module_name)
        
        # Convert 'correct' key to 'correct_answer' for compatibility with frontend
        formatted_questions = []
        for q in questions:
            formatted_q = q.copy()
            formatted_q['correct_answer'] = formatted_q.pop('correct')
            formatted_questions.append(formatted_q)
        
        return jsonify(formatted_questions)
    except Exception as e:
        logging.exception("Error getting quiz questions:")
        return jsonify([]), 500

@app.route('/test')
def test():
    return "Flask app is working! Routes are properly registered."

# Initialize the database
with app.app_context():
    db.create_all()

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
