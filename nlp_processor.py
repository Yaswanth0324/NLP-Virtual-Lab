import logging
import os
import re
import string
import random
import json
from collections import Counter

# --- Core NLP Libraries ---
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tag import pos_tag
from nltk.chunk import ne_chunk
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.util import ngrams

# --- For Machine Learning and NLP Models ---
import numpy as np
import psycopg2 
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import pipeline

# --- Ensure required NLTK data is available ---
def _ensure_nltk_data():
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')

    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords')

    try:
        nltk.data.find('taggers/averaged_perceptron_tagger')
    except LookupError:
        try:
            nltk.download('averaged_perceptron_tagger')
        except Exception:
            pass
    try:
        nltk.data.find('taggers/averaged_perceptron_tagger_eng')
    except LookupError:
        try:
            nltk.download('averaged_perceptron_tagger_eng')
        except Exception:
            pass

    try:
        nltk.data.find('chunkers/maxent_ne_chunker')
    except LookupError:
        nltk.download('maxent_ne_chunker')

    try:
        nltk.data.find('corpora/words')
    except LookupError:
        nltk.download('words')

    try:
        nltk.data.find('corpora/wordnet')
    except LookupError:
        nltk.download('wordnet')

    try:
        nltk.data.find('sentiment/vader_lexicon')
    except LookupError:
        nltk.download('vader_lexicon')

# Additional compatibility downloads for newer NLTK versions
# punkt_tab (required by newer tokenizers)
try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    try:
        nltk.download('punkt_tab')
    except Exception:
        pass

# New tagger package name in recent NLTK releases
try:
    nltk.data.find('taggers/averaged_perceptron_tagger_eng')
except LookupError:
    try:
        nltk.download('averaged_perceptron_tagger_eng')
    except Exception:
        pass

# WordNet dependency for lemmatization in some environments
try:
    nltk.data.find('corpora/omw-1.4')
except LookupError:
    try:
        nltk.download('omw-1.4')
    except Exception:
        pass


class NLPProcessor:
    def __init__(self):
        self.stemmer = PorterStemmer()
        self.lemmatizer = WordNetLemmatizer()
        self.sentiment_analyzer = SentimentIntensityAnalyzer()
        self.stop_words = set(stopwords.words('english'))
        self._summarizer = None
        self._text_generator = None
        
    def process(self, text, operation):
        """Process text based on the specified operation"""
        try:
            if operation == 'tokenize':
                return self.tokenize(text)
            elif operation == 'preprocess':
                return self.preprocess(text)
            elif operation == 'pos_tag':
                return self.pos_tag(text)
            elif operation == 'ngrams':
                return self.generate_ngrams(text)
            elif operation == 'ner':
                return self.named_entity_recognition(text)
            elif operation == 'sentiment':
                return self.sentiment_analysis(text)
            elif operation == 'stem':
                return self.stem_text(text)
            elif operation == 'lemmatize':
                return self.lemmatize_text(text)
            elif operation == 'chunk':
                return self.chunk_text(text)
            elif operation == 'summarize':
                return self.summarize_text(text, max_length=130, min_length=30)
            elif operation == 'generate_text':
                return self.generate_text(text, max_new_tokens=50)
            elif operation == 'translate_en_hi':
                return self.translate_text(text, 'en_hi')
            elif operation == 'translate_hi_en':
                return self.translate_text(text, 'hi_en')
            elif operation == 'translate_en_te':
                return self.translate_text(text, 'en_te')
            elif operation == 'translate_hi_te':
                return self.translate_text(text, 'hi_te')
            elif operation == 'translate_te_en':
                return self.translate_text(text, 'te_en')
            elif operation == 'translate_te_hi':
                return self.translate_text(text, 'te_hi')
            elif operation == 'qa':
                return self.answer_question(text)
            elif operation == 'topic_model':
                return self.topic_modeling(text)
            elif operation == 'attention_demo':
                return self.attention_demo(text)
            else:
                return {'error': 'Unknown operation'}
        except Exception as e:
            logging.error(f"Error processing text: {e}")
            return {'error': str(e)}
    
    def tokenize(self, text):
        words = word_tokenize(text)
        sentences = sent_tokenize(text)
        return {
            'words': words,
            'sentences': sentences,
            'word_count': len(words),
            'sentence_count': len(sentences),
            'visualization': {'type': 'tokens', 'data': words}
        }
    
    def preprocess(self, text):
        original = text
        lowercased = text.lower()
        no_punct = lowercased.translate(str.maketrans('', '', string.punctuation))
        tokens = word_tokenize(no_punct)
        filtered_tokens = [word for word in tokens if word not in self.stop_words]
        stemmed = [self.stemmer.stem(word) for word in filtered_tokens]
        return {
            'original': original,
            'lowercased': lowercased,
            'no_punctuation': no_punct,
            'tokens': tokens,
            'filtered_tokens': filtered_tokens,
            'stemmed': stemmed,
            'steps': [
                {'name': 'Original', 'text': original},
                {'name': 'Lowercased', 'text': lowercased},
                {'name': 'No Punctuation', 'text': no_punct},
                {'name': 'Tokenized', 'text': ' '.join(tokens)},
                {'name': 'Stopwords Removed', 'text': ' '.join(filtered_tokens)},
                {'name': 'Stemmed', 'text': ' '.join(stemmed)}
            ]
        }
    
    def pos_tag(self, text):
        tokens = word_tokenize(text)
        tagged = pos_tag(tokens)
        pos_groups = {}
        for word, pos in tagged:
            pos_groups.setdefault(pos, []).append(word)
        return {
            'tagged': tagged,
            'pos_groups': pos_groups,
            'total_words': len(tokens),
            'unique_pos': len(pos_groups),
            'visualization': {'type': 'pos_chart', 'data': pos_groups}
        }
    
    def generate_ngrams(self, text, n=2):
        tokens = word_tokenize(text.lower())
        unigrams = list(ngrams(tokens, 1))
        bigrams = list(ngrams(tokens, 2))
        trigrams = list(ngrams(tokens, 3))
        unigram_freq = Counter(unigrams)
        bigram_freq = Counter(bigrams)
        trigram_freq = Counter(trigrams)
        return {
            'unigrams': {
                'grams': [' '.join(gram) for gram in unigrams],
                'frequencies': {' '.join(gram): freq for gram, freq in unigram_freq.most_common(10)}
            },
            'bigrams': {
                'grams': [' '.join(gram) for gram in bigrams],
                'frequencies': {' '.join(gram): freq for gram, freq in bigram_freq.most_common(10)}
            },
            'trigrams': {
                'grams': [' '.join(gram) for gram in trigrams],
                'frequencies': {' '.join(gram): freq for gram, freq in trigram_freq.most_common(10)}
            },
            'visualization': {
                'type': 'ngram_chart',
                'data': {
                    'unigrams': {' '.join(gram): freq for gram, freq in unigram_freq.most_common(5)},
                    'bigrams': {' '.join(gram): freq for gram, freq in bigram_freq.most_common(5)},
                    'trigrams': {' '.join(gram): freq for gram, freq in trigram_freq.most_common(5)}
                }
            }
        }
    
    def named_entity_recognition(self, text):
        tokens = word_tokenize(text)
        tagged = pos_tag(tokens)
        entities = ne_chunk(tagged)
        named_entities = []
        for chunk in entities:
            if hasattr(chunk, 'label'):
                entity_words = [token for token, pos in chunk]
                named_entities.append({
                    'entity': ' '.join(entity_words),
                    'label': chunk.label(),
                    'words': entity_words
                })
        entity_groups = {}
        for entity in named_entities:
            entity_groups.setdefault(entity['label'], []).append(entity['entity'])
        return {
            'entities': named_entities,
            'entity_groups': entity_groups,
            'total_entities': len(named_entities),
            'entity_types': list(entity_groups.keys()),
            'visualization': {'type': 'entity_chart', 'data': entity_groups}
        }
    
    def sentiment_analysis(self, text):
        scores = self.sentiment_analyzer.polarity_scores(text)
        if scores['compound'] >= 0.05:
            overall = 'Positive'
        elif scores['compound'] <= -0.05:
            overall = 'Negative'
        else:
            overall = 'Neutral'
        return {
            'scores': scores,
            'overall_sentiment': overall,
            'confidence': abs(scores['compound']),
            'breakdown': {
                'positive': scores['pos'],
                'negative': scores['neg'],
                'neutral': scores['neu']
            },
            'visualization': {
                'type': 'sentiment_chart',
                'data': {
                    'Positive': scores['pos'],
                    'Negative': scores['neg'],
                    'Neutral': scores['neu']
                }
            }
        }
    
    def stem_text(self, text):
        tokens = word_tokenize(text)
        stemmed = [(token, self.stemmer.stem(token)) for token in tokens]
        return {
            'original_tokens': tokens,
            'stemmed_pairs': stemmed,
            'stemmed_text': ' '.join([stem for _, stem in stemmed])
        }
    
    def lemmatize_text(self, text):
        tokens = word_tokenize(text)
        lemmatized = [(token, self.lemmatizer.lemmatize(token)) for token in tokens]
        return {
            'original_tokens': tokens,
            'lemmatized_pairs': lemmatized,
            'lemmatized_text': ' '.join([lemma for _, lemma in lemmatized])
        }
    
    def chunk_text(self, text):
        tokens = word_tokenize(text)
        tagged = pos_tag(tokens)
        grammar = r"""
            NP: {<DT|PP\$>?<JJ>*<NN>}
                {<NNP>+}
        """
        try:
            cp = nltk.RegexpParser(grammar)
            chunks = cp.parse(tagged)
            noun_phrases = []
            for chunk in chunks:
                if hasattr(chunk, 'label') and chunk.label() == 'NP':
                    phrase = ' '.join([word for word, pos in chunk])
                    noun_phrases.append(phrase)
            return {
                'tagged_tokens': tagged,
                'noun_phrases': noun_phrases,
                'chunk_tree': str(chunks),
                'visualization': {'type': 'chunk_tree', 'data': str(chunks)}
            }
        except Exception as e:
            return {
                'error': f'Chunking failed: {str(e)}',
                'tagged_tokens': tagged,
                'noun_phrases': [],
                'chunk_tree': ''
            }
    
    def translate_text(self, text, src_lang='auto', dest_lang='en'):
        """Translate text using googletrans.
        :param text: Text to translate
        :param src_lang: Source language code (default: 'auto')
        :param dest_lang: Destination language code (default: 'en')
        :return: dict with translation result
        """
        try:
            # Import locally to avoid hard dependency at module import time
            from googletrans import Translator
            translator = Translator()
            result = translator.translate(text, src=src_lang or 'auto', dest=dest_lang or 'en')
            return {
                'original_text': text,
                'translated_text': result.text,
                'detected_source': getattr(result, 'src', src_lang or 'auto'),
                'target_language': getattr(result, 'dest', dest_lang or 'en')
            }
        except Exception as e:
            logging.error(f"Translation failed: {e}")
            return {'error': f'Translation failed: {str(e)}'}

    def _get_summarizer(self):
        if self._summarizer is None:
            try:
                self._summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6", framework="pt")
            except Exception as e:
                logging.error(f"Failed to initialize summarizer: {e}")
                self._summarizer = None
        return self._summarizer

    def summarize_text(self, text, max_length=130, min_length=30):
        if not text or not text.strip():
            return {'error': 'No text provided.'}
        try:
            summarizer = self._get_summarizer()
            if summarizer is None:
                return {'error': 'Summarizer not available. Please install transformers and a supported backend (torch or tf-keras).'}
            result = summarizer(text, max_length=max_length, min_length=min_length, do_sample=False)
            summary = result[0]['summary_text']
            return {'summary': summary}
        except Exception as e:
            logging.error(f"Summarization failed: {e}")
            return {'error': f'Summarization failed: {str(e)}'}

    def _get_text_generator(self):
        if self._text_generator is None:
            try:
                self._text_generator = pipeline("text-generation", model="gpt2", framework="pt")
            except Exception as e:
                logging.error(f"Failed to initialize text generator: {e}")
                self._text_generator = None
        return self._text_generator

    def generate_text(self, prompt, max_new_tokens=50, num_return_sequences=1,
                      do_sample=True, top_k=50, top_p=0.95,
                      temperature=0.9, repetition_penalty=1.2):
        if not prompt or not prompt.strip():
            return {'error': 'No prompt provided.'}
        try:
            generator = self._get_text_generator()
            if generator is None:
                return {'error': 'Text generator not available. Please install transformers and a supported backend (torch or tf-keras).'}
            results = generator(
                prompt,
                max_new_tokens=max_new_tokens,
                num_return_sequences=num_return_sequences,
                do_sample=do_sample,
                top_k=top_k,
                top_p=top_p,
                temperature=temperature,
                repetition_penalty=repetition_penalty
            )
            generated_texts = [r.get('generated_text', '') for r in results]
            return {'generated': generated_texts}
        except Exception as e:
            logging.error(f"Text generation failed: {e}")
            return {'error': f'Text generation failed: {str(e)}'}

    def get_quiz_questions(self, module_name):
        conn = None
        try:
            db_url = os.environ.get("DATABASE_URL")
            if not db_url:
                raise ValueError("DATABASE_URL not found in .env file.")

            conn = psycopg2.connect(db_url)
            cur = conn.cursor()

            cur.execute(
                "SELECT question, options, correct_answer, explanation FROM quiz_question WHERE module_name = %s",
                (module_name,)
            )
            
            questions_from_db = cur.fetchall()
            
            formatted_questions = []
            for row in questions_from_db:
                try:
                    options_list = json.loads(row[1])
                except (json.JSONDecodeError, TypeError) as e:
                    logging.error(f"Failed to decode options for question '{row[0]}': {e}. Raw data: {row[1]}")
                    options_list = []
                
                formatted_questions.append({
                    'question': row[0],
                    'options': options_list,
                    'correct_answer': row[2],
                    'explanation': row[3]
                })
            
            return formatted_questions

        except (Exception, psycopg2.DatabaseError) as error:
            logging.error(f"Error fetching quiz questions from database: {error}")
            return []
        finally:
            if conn is not None:
                conn.close()
    
    def calculate_quiz_score(self, module_name, answers):
        questions = self.get_quiz_questions(module_name)
        if not questions:
            return 0
        
        correct = 0
        for i, answer in enumerate(answers):
            try:
                user_answer = int(answer)
                if i < len(questions) and user_answer == questions[i]['correct_answer']:
                    correct += 1
            except (ValueError, TypeError):
                logging.warning(f"Could not convert answer '{answer}' to int.")
                continue
        
        return int((correct / len(questions)) * 100)