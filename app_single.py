import os
import logging
from flask import Flask, render_template, request, jsonify
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy.orm import DeclarativeBase
from werkzeug.middleware.proxy_fix import ProxyFix
from flask_login import UserMixin
from datetime import datetime
import json

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
