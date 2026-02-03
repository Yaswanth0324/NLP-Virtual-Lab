NLP Virtual Lab

Welcome to the NLP Virtual Lab, a comprehensive, web-based platform for exploring and experimenting with core Natural Language Processing (NLP) concepts. This interactive application provides:

🧪 Hands-on labs for various NLP tasks

🤖 Smart chatbot powered by the Gemini API for answering questions

📝 Quizzes to test your knowledge

The backend is powered by a Flask application, with NLP logic in a dedicated processor class and data persisted in a PostgreSQL database hosted on Supabase.

Core Features

This virtual lab covers a wide range of NLP topics:

Text Processing

Text Preprocessing: Tokenization, lowercasing, stopword removal, stemming/lemmatization

Part-of-Speech (POS) Tagging: Identify grammatical categories (nouns, verbs, adjectives)

N-Gram Modeling: Generate unigrams, bigrams, trigrams to understand word context

Named Entity Recognition (NER): Extract names, locations, organizations

Sentiment Analysis: Determine positive, negative, or neutral sentiment

Text Classification: Categorize text into topics like Technology, Sports, Health

Word Embeddings: Visualize numerical vector representations of words

Chunking & Parsing: Identify Noun Phrases (NP) and Verb Phrases (VP), demonstrate CNF transformations

Topic Modeling: Latent Dirichlet Allocation (LDA) for discovering hidden topics

Advanced AI Models

Machine Translation: Translate text between English, Hindi, Telugu

Text Summarization: Condense long articles into concise summaries

Text Generation: Generate creative text from a prompt

Question Answering: Answer questions from a given context

Speech Processing: (Add a description of speech processing features here)

AI Chatbot: Google Gemini API-powered assistant for explanations and code examples

Interactive Quizzes: Questions stored in PostgreSQL database

Tech Stack

Backend: Python 3, Flask

NLP Libraries: NLTK, scikit-learn, Hugging Face Transformers

Database: PostgreSQL (hosted on Supabase)

AI Services: Google Gemini API

Frontend: HTML5, CSS3, JavaScript

Deployment: Platforms like Render or any Python web app service

Setup and Installation
Prerequisites

Python 3.8+

pip (Python package installer)

PostgreSQL database (free tier on Supabase)

Google Gemini API Key

1️⃣ Clone the Repository
git clone <your-repository-url>
cd nlp-virtual-lab

2️⃣ Install Dependencies
pip install -r requirements.txt


Include libraries: Flask, python-dotenv, psycopg2-binary, nltk, scikit-learn, transformers, torch, etc.

3️⃣ Set Up the Database

Create a project on Supabase

Run CREATE TABLE script for quiz_question

Run INSERT script to populate questions

4️⃣ Configure Environment Variables

Create a .env file in the root directory:

# PostgreSQL connection string
DATABASE_URL="postgresql://postgres:[YOUR-PASSWORD]@db.xxxxxxxx.supabase.co:5432/postgres"

# Google Gemini API Key
GEMINI_API_KEY="your_gemini_api_key_here"

# Flask session secret
SESSION_SECRET="a-strong-and-random-secret-key"


Replace placeholders with your actual credentials.

5️⃣ Run the Application
python app_single.py


App runs at: http://127.0.0.1:5000

Project Structure

app_single.py: Main Flask app (routes, API endpoints, templates)

nlp_processor.py: Core NLP logic class

.env: Stores secrets like API keys and database URL

templates/: HTML files for frontend

static/: CSS, JS, and other assets

insert_questions.sql: SQL script for quiz questions

Deployment

Render Example:

Push code to GitHub

Create a new Web Service and connect repo

Build Command: pip install -r requirements.txt && python scripts/download_nltk_data.py

Start Command: gunicorn app_single:app (add gunicorn to requirements.txt)

Add DATABASE_URL and GEMINI_API_KEY to Render environment

Screenshots

Main Dashboard: (Add screenshot)

Interactive Lab View: (Add screenshot)

AI Chatbot Interface: (Add screenshot)

✅ This README provides a complete guide to understanding, setting up, and using the NLP Virtual Lab. Enjoy exploring NLP!
