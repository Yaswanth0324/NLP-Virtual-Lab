from flask import render_template, request, jsonify, session
from nlp_processor import NLPProcessor
import json

# Initialize NLP processor
nlp_processor = NLPProcessor()

def register_routes(app, db):
    """Register routes with the Flask app"""
    from models import User, UserProgress, QuizQuestion
    
    # Define lab modules
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
    
    @app.route('/')
    def index():
        return render_template('index.html', modules=LAB_MODULES)
    
    @app.route('/lab/<module_name>')
    def lab(module_name):
        if module_name not in LAB_MODULES:
            return "Lab module not found", 404
        
        module_info = LAB_MODULES[module_name]
        return render_template('lab.html',
                             module_name=module_name,
                             module_info=module_info)
    
    @app.route('/quiz/<module_name>')
    def quiz(module_name):
        if module_name not in LAB_MODULES:
            return "Quiz not found", 404
        
        module_info = LAB_MODULES[module_name]
        return render_template('quiz.html',
                             module_name=module_name,
                             module_info=module_info)
    
    @app.route('/api/process', methods=['POST'])
    def process_text():
        data = request.get_json()
        text = data.get('text', '')
        operation = data.get('operation', '')
        
        if not text or not operation:
            return jsonify({'error': 'Text and operation are required'}), 400
        
        try:
            result = nlp_processor.process(text, operation)
            return jsonify(result)
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    @app.route('/api/quiz/<module_name>')
    def get_quiz_questions(module_name):
        # For demo purposes, return hardcoded questions
        # In production, these would come from the database
        questions = nlp_processor.get_quiz_questions(module_name)
        return jsonify(questions)
    
    @app.route('/api/quiz/submit', methods=['POST'])
    def submit_quiz():
        data = request.get_json()
        module_name = data.get('module_name')
        answers = data.get('answers', [])
        
        if not module_name or not answers:
            return jsonify({'error': 'Module name and answers are required'}), 400
        
        try:
            score = nlp_processor.calculate_quiz_score(module_name, answers)
            return jsonify({'score': score})
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    @app.route('/api/progress/<module_name>')
    def get_progress(module_name):
        # For demo purposes, return basic progress
        # In production, this would track actual user progress
        return jsonify({
            'completed': False,
            'score': 0,
            'attempts': 0
        })
    
    @app.route('/api/progress', methods=['POST'])
    def update_progress():
        data = request.get_json()
        module_name = data.get('module_name')
        completed = data.get('completed', False)
        score = data.get('score', 0)
        
        # For demo purposes, just return success
        # In production, this would update the database
        return jsonify({'success': True})
    
    # Chatbot API endpoint
    @app.route('/api/chatbot', methods=['POST'])
    def chatbot():
        data = request.get_json()
        message = data.get('message', '')
        session_id = data.get('session_id', '')
        
        if not message:
            return jsonify({'error': 'Message is required'}), 400
        
        try:
            # Process the message with the chatbot
            response = nlp_processor.chatbot_response(message)
            return jsonify({'response': response, 'session_id': session_id})
        except Exception as e:
            return jsonify({'error': str(e)}), 500

# Define lab modules
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

@app.route('/')
def index():
    return render_template('index.html', modules=LAB_MODULES)

@app.route('/lab/<module_name>')
def lab(module_name):
    if module_name not in LAB_MODULES:
        return "Lab module not found", 404
    
    module_info = LAB_MODULES[module_name]
    return render_template('lab.html', 
                         module_name=module_name, 
                         module_info=module_info)

@app.route('/quiz/<module_name>')
def quiz(module_name):
    if module_name not in LAB_MODULES:
        return "Quiz not found", 404
    
    module_info = LAB_MODULES[module_name]
    return render_template('quiz.html', 
                         module_name=module_name, 
                         module_info=module_info)

@app.route('/api/process', methods=['POST'])
def process_text():
    data = request.get_json()
    text = data.get('text', '')
    operation = data.get('operation', '')
    
    if not text or not operation:
        return jsonify({'error': 'Text and operation are required'}), 400
    
    try:
        result = nlp_processor.process(text, operation)
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/quiz/<module_name>')
def get_quiz_questions(module_name):
    # For demo purposes, return hardcoded questions
    # In production, these would come from the database
    questions = nlp_processor.get_quiz_questions(module_name)
    return jsonify(questions)

@app.route('/api/quiz/submit', methods=['POST'])
def submit_quiz():
    data = request.get_json()
    module_name = data.get('module_name')
    answers = data.get('answers', [])
    
    if not module_name or not answers:
        return jsonify({'error': 'Module name and answers are required'}), 400
    
    try:
        score = nlp_processor.calculate_quiz_score(module_name, answers)
        return jsonify({'score': score})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/progress/<module_name>')
def get_progress(module_name):
    # For demo purposes, return basic progress
    # In production, this would track actual user progress
    return jsonify({
        'completed': False,
        'score': 0,
        'attempts': 0
    })

@app.route('/api/progress', methods=['POST'])
def update_progress():
    data = request.get_json()
    module_name = data.get('module_name')
    completed = data.get('completed', False)
    score = data.get('score', 0)
    
    # For demo purposes, just return success
    # In production, this would update the database
    return jsonify({'success': True})

# Chatbot API endpoint
@app.route('/api/chatbot', methods=['POST'])
def chatbot():
   data = request.get_json()
   message = data.get('message', '')
   session_id = data.get('session_id', '')
   
   if not message:
       return jsonify({'error': 'Message is required'}), 400
   
   try:
       # Process the message with the chatbot
       response = nlp_processor.chatbot_response(message)
       return jsonify({'response': response, 'session_id': session_id})
   except Exception as e:
       return jsonify({'error': str(e)}), 500
