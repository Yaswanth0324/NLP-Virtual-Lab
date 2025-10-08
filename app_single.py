import os
import logging
from flask import Flask, render_template, request, jsonify, send_from_directory
from werkzeug.middleware.proxy_fix import ProxyFix
from datetime import datetime
import json
import random
from transformers import pipeline
from dotenv import load_dotenv
import psycopg2 # For connecting to PostgreSQL
import audioop 
import torch 

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
# Load environment variables from .env file
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.DEBUG)

# Create the app
app = Flask(__name__)
app.secret_key = os.environ.get("SESSION_SECRET", "dev-secret-key-change-in-production")
app.wsgi_app = ProxyFix(app.wsgi_app, x_proto=1, x_host=1)

# Initialize summarization pipeline lazily and prefer PyTorch backend
_summarizer = None

def get_summarizer():
    global _summarizer
    if _summarizer is None:
        try:
            _summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6", framework="pt")
        except Exception:
            # Fallback to default initialization (may require TensorFlow with tf-keras)
            _summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")
    return _summarizer

# Initialize text-generation pipeline lazily and prefer PyTorch backend

_text_generator = None

def get_text_generator():
    global _text_generator
    if _text_generator is None:
        try:
            # Import pipeline from the transformers library
            from transformers import pipeline

            # Attempt to use the 'pt' framework (PyTorch)
            _text_generator = pipeline("text-generation", model="gpt2", framework="pt")
            logging.info("Text generation pipeline loaded successfully with PyTorch.")
        except ImportError:
            # Fallback if PyTorch is not installed
            logging.warning("PyTorch not found. Attempting to load text-generation pipeline without specifying framework.")
            try:
                from transformers import pipeline
                _text_generator = pipeline("text-generation", model="gpt2")
                logging.info("Text generation pipeline loaded successfully without a specified framework.")
            except Exception as e:
                logging.error(f"Failed to load text generation pipeline: {e}")
                _text_generator = None # Ensure it remains None on failure

    if _text_generator is None:
        raise RuntimeError("Failed to initialize text generation pipeline. Please check your transformers and PyTorch installation.")

    return _text_generator

# ---------------- Chatbot Updated Section ---------------- #

def search_internet(query):
    """
    Searches the internet for a given query using the DuckDuckGo API.
    """
    if not requests:
        logging.warning("The 'requests' library is not installed. Internet search is disabled.")
        return ""

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

        return " ".join(results[:3]) if results else ""
    except Exception as e:
        logging.error(f"Search error: {e}")
        return ""

def generate_gemini_response(query, context):
    """
    Generates a response using the Google Gemini API.
    """
    if not google_ai_available:
        return "Google AI backend not installed."

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        return "Gemini API key missing. Please set GEMINI_API_KEY."

    try:
        genai.configure(api_key=api_key)
        # Use a supported model (fixes Render 404 issue)
        model = genai.GenerativeModel("gemini-2.0-flash")

        prompt = f"""
You are an NLP assistant. Answer the user clearly and helpfully.
Search Context: {context}
User Query: {query}
"""
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        logging.error(f"Gemini error: {e}")
        return f"Error from Gemini API: {str(e)}"

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

        # Get search context
        search_context = search_internet(message)
        logging.debug(f"[CHATBOT] Search context: {search_context}")

        # Get Gemini response
        final_response = generate_gemini_response(message, search_context)
        logging.debug(f"[CHATBOT] Final response: {final_response}")

        return jsonify({'response': final_response, 'session_id': session_id})
        
    except Exception as e:
        logging.error(f"Error in chatbot API: {str(e)}")
        return jsonify({'error': f"Server Error: {str(e)}"}), 500

# ---------------- End of Chatbot Updated Section ---------------- #

@app.route('/summarize', methods=['POST'])
def summarize():
    data = request.get_json()
    # Extract input data
    text = data.get('text', '')
    max_length = data.get('max_length', 130)
    min_length = data.get('min_length', 30)

    if not text.strip():
        return jsonify({'error': 'No text provided.'}), 400

    try:
        # Run summarization
        summarizer = get_summarizer()
        result = summarizer(text, max_length=max_length, min_length=min_length, do_sample=False)
        summary = result[0]['summary_text']
        return jsonify({'summary': summary})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/generate', methods=['POST'])
def generate():
    data = request.get_json()
    prompt = data.get('prompt', '')
    max_length = data.get('max_length', 50)
    num_return_sequences = data.get('num_return_sequences', 1)

    if not prompt:
        return jsonify({'error': 'No prompt provided.'}), 400

    try:
        generator = get_text_generator()
        results = generator(
            prompt,
            max_new_tokens=50,
            num_return_sequences=num_return_sequences,
            do_sample=True,
            top_k=50,
            top_p=0.95
        )
        generated_texts = [r.get('generated_text', '') for r in results]
        return jsonify({'generated': generated_texts})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/qa', methods=['POST'])
def question_answering():
    data = request.get_json()
    question = (data.get('question') or '').strip()
    context = (data.get('context') or '').strip()

    if not question or not context:
        return jsonify({'error': 'Both question and context are required.'}), 400

    try:
        from nlp_processor import NLPProcessor
        processor = NLPProcessor()
        result = processor.question_answer(question, context)
        return jsonify(result)
    except Exception as e:
        logging.exception('QA endpoint failed')
        return jsonify({'error': f'QA failed: {str(e)}'}), 200

# Lab modules configuration
LAB_MODULES = {
    'text_preprocessing': {'title': 'Text Preprocessing', 'description': 'Learn tokenization, lowercasing, punctuation removal, and stopword removal', 'icon': 'file-text'},
    'pos_tagging': {'title': 'Part-of-Speech Tagging', 'description': 'Identify grammatical categories of words', 'icon': 'tag'},
    'ngram_modeling': {'title': 'N-Gram Modeling', 'description': 'Analyze word sequences and context', 'icon': 'layers'},
    'named_entity_recognition': {'title': 'Named Entity Recognition', 'description': 'Extract entities like names, places, and organizations', 'icon': 'user'},
    'sentiment_analysis': {'title': 'Sentiment Analysis', 'description': 'Determine emotional tone of text', 'icon': 'heart'},
    'text_classification': {'title': 'Text Classification', 'description': 'Categorize text into predefined classes', 'icon': 'folder'},
    'word_embeddings': {'title': 'Word Embeddings', 'description': 'Convert text to numerical representations', 'icon': 'grid'},
    'chunking': {'title': 'Chunking & Parsing', 'description': 'Group words into meaningful phrases', 'icon': 'git-branch'},
    'machine_translation': {'title': 'Machine Translation', 'description': 'Translate text between languages', 'icon': 'globe'},
    'text_summarization': {'title': 'Text Summarization', 'description': 'Generate concise summaries of long text', 'icon': 'file-text'},
    'text_generation': {'title': 'Text Generation', 'description': 'Generate text continuations from prompts', 'icon': 'type'},
    'topic_modelling': {'title': 'Topic Modelling', 'description': 'Discover latent topics with LDA', 'icon': 'book-open'},
    'question_answering': {'title': 'Question Answering', 'description': 'Find answers from context passages', 'icon': 'help-circle'},
    'speech_processing': {'title': 'Speech Processing', 'description': 'Practice Speech-to-Text and Text-to-Speech in browser', 'icon': 'mic'}
}

# Routes
@app.route('/')
def index():
    return render_template('index.html', modules=LAB_MODULES)

# Favicon routes
@app.route('/favicon.png')
def favicon_png():
    return send_from_directory(os.path.join(app.root_path, 'images'), 'nlp-logo.png', mimetype='image/png')

@app.route('/favicon.ico')
def favicon_ico():
    return send_from_directory(os.path.join(app.root_path, 'images'), 'nlp-logo.png', mimetype='image/png')

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

        if operation == 'translate':
            src_lang = data.get('src_lang') or 'auto'
            dest_lang = data.get('dest_lang') or 'en'
            if isinstance(src_lang, str):
                src_lang = src_lang.strip() or 'auto'
            if isinstance(dest_lang, str):
                dest_lang = dest_lang.strip() or 'en'
            result = nlp_processor.translate_text(text, src_lang=src_lang, dest_lang=dest_lang)
        else:
            result = nlp_processor.process(text, operation)
        return jsonify(result)

    except Exception as e:
        logging.exception("Error processing text:")
        return jsonify({'error': f"Server Error: {str(e)}"}), 500

@app.route('/api/quiz/<module_name>')
def get_quiz_questions(module_name):
    """
    Fetches quiz questions for a given module directly from the PostgreSQL database.
    """
    conn = None
    try:
        # Get the database URL from environment variables
        db_url = os.environ.get("DATABASE_URL")
        if not db_url:
            raise ValueError("DATABASE_URL not found in .env file.")

        # Connect to the PostgreSQL database
        conn = psycopg2.connect(db_url)
        cur = conn.cursor()

        # Execute the query to get questions for the specified module
        cur.execute(
            "SELECT question, options, correct_answer, explanation FROM quiz_question WHERE module_name = %s",
            (module_name,)
        )
        
        # Fetch all results
        questions_from_db = cur.fetchall()
        
        # Format the results into a list of dictionaries
        formatted_questions = []
        for row in questions_from_db:
            formatted_questions.append({
                'question': row[0],
                'options': row[1], # Options are already in JSON format
                'correct_answer': row[2],
                'explanation': row[3]
            })
        
        return jsonify(formatted_questions)

    except (Exception, psycopg2.DatabaseError) as error:
        logging.error(f"Error fetching quiz questions from database: {error}")
        return jsonify({'error': f"Database Error: {str(error)}"}), 500
    finally:
        # Ensure the database connection is closed
        if conn is not None:
            conn.close()


@app.route('/test')
def test():
    return "Flask app is working! Routes are properly registered."


# if __name__ == '__main__':
#     app.run(host='0.0.0.0', port=int(os.environ.get("PORT", 5000)), debug=True)
if __name__ == "__main__":
    app.run()
