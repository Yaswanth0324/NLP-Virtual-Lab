import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tag import pos_tag
from nltk.chunk import ne_chunk
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.util import ngrams
import re
import string
from collections import Counter
import logging
import os
import psycopg2  # Added for database connection


# Download required NLTK data
# (NLTK download checks remain the same)
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
    nltk.download('averaged_perceptron_tagger')

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


class NLPProcessor:
    def __init__(self):
        self.stemmer = PorterStemmer()
        self.lemmatizer = WordNetLemmatizer()
        self.sentiment_analyzer = SentimentIntensityAnalyzer()
        self.stop_words = set(stopwords.words('english'))
        self._summarizer = None
        
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
                # Default lengths if not provided via /summarize route
                return self.summarize_text(text, max_length=130, min_length=30)
            elif operation == 'translate':
                # Fallback support if route does not pass src/dest; defaults used
                return self.translate_text(text, src_lang='auto', dest_lang='en')
            else:
                return {'error': 'Unknown operation'}
        except Exception as e:
            logging.error(f"Error processing text: {e}")
            return {'error': str(e)}
    
    def tokenize(self, text):
        """Tokenize text into words and sentences"""
        words = word_tokenize(text)
        sentences = sent_tokenize(text)
        
        return {
            'words': words,
            'sentences': sentences,
            'word_count': len(words),
            'sentence_count': len(sentences),
            'visualization': {
                'type': 'tokens',
                'data': words
            }
        }
    
    def preprocess(self, text):
        """Complete preprocessing pipeline"""
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
        """Part-of-speech tagging"""
        tokens = word_tokenize(text)
        tagged = pos_tag(tokens)
        
        pos_groups = {}
        for word, pos in tagged:
            if pos not in pos_groups:
                pos_groups[pos] = []
            pos_groups[pos].append(word)
        
        return {
            'tagged': tagged,
            'pos_groups': pos_groups,
            'total_words': len(tokens),
            'unique_pos': len(pos_groups),
            'visualization': {
                'type': 'pos_chart',
                'data': pos_groups
            }
        }
    
    def generate_ngrams(self, text, n=2):
        """Generate n-grams from text"""
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
        """Extract named entities"""
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
            label = entity['label']
            if label not in entity_groups:
                entity_groups[label] = []
            entity_groups[label].append(entity['entity'])
        
        return {
            'entities': named_entities,
            'entity_groups': entity_groups,
            'total_entities': len(named_entities),
            'entity_types': list(entity_groups.keys()),
            'visualization': {
                'type': 'entity_chart',
                'data': entity_groups
            }
        }
    
    def sentiment_analysis(self, text):
        """Analyze sentiment using VADER"""
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
        """Stem words in text"""
        tokens = word_tokenize(text)
        stemmed = [(token, self.stemmer.stem(token)) for token in tokens]
        
        return {
            'original_tokens': tokens,
            'stemmed_pairs': stemmed,
            'stemmed_text': ' '.join([stem for _, stem in stemmed])
        }
    
    def lemmatize_text(self, text):
        """Lemmatize words in text"""
        tokens = word_tokenize(text)
        lemmatized = [(token, self.lemmatizer.lemmatize(token)) for token in tokens]
        
        return {
            'original_tokens': tokens,
            'lemmatized_pairs': lemmatized,
            'lemmatized_text': ' '.join([lemma for _, lemma in lemmatized])
        }
    
    def chunk_text(self, text):
        """Perform noun phrase chunking"""
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
                'visualization': {
                    'type': 'chunk_tree',
                    'data': str(chunks)
                }
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

    # Text Summarization (Transformers)
    def _get_summarizer(self):
        if getattr(self, '_summarizer', None) is None:
            try:
                # Prefer PyTorch backend if available
                try:
                    import torch  # noqa: F401
                    from transformers import pipeline
                    self._summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6", framework="pt")
                except Exception:
                    from transformers import pipeline
                    self._summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")
            except Exception as e:
                logging.error(f"Failed to initialize summarizer: {e}")
                self._summarizer = None
        return self._summarizer

    def summarize_text(self, text, max_length=130, min_length=30):
        """
        Generate a concise summary for the given text.
        Falls back with an error dict if transformers or model is unavailable.
        """
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

    def get_quiz_questions(self, module_name):
        """
        Fetches quiz questions for a given module directly from the PostgreSQL database.
        """
        if psycopg2 is None:
            logging.error("psycopg2 is not installed; quiz questions cannot be fetched.")
            return []
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
                formatted_questions.append({
                    'question': row[0],
                    'options': row[1],
                    'correct': row[2], # Changed back to 'correct' to match original structure
                    'explanation': row[3]
                })
            
            return formatted_questions

        except (Exception, psycopg2.DatabaseError) as error:
            logging.error(f"Error fetching quiz questions from database: {error}")
            return [] # Return an empty list on error
        finally:
            if conn is not None:
                conn.close()
    
    def calculate_quiz_score(self, module_name, answers):
        """Calculate quiz score based on answers"""
        questions = self.get_quiz_questions(module_name)
        if not questions:
            return 0
        
        correct = 0
        for i, answer in enumerate(answers):
            # Ensure the answer is an integer for comparison
            try:
                user_answer = int(answer)
                if i < len(questions) and user_answer == questions[i]['correct']:
                    correct += 1
            except (ValueError, TypeError):
                logging.warning(f"Could not convert answer '{answer}' to int.")
                continue # Skip if the answer is not a valid number
        
        return int((correct / len(questions)) * 100)

