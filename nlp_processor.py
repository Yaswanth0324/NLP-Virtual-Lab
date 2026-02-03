import logging
import os
import re
import string
import random
import json
import copy
import io
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
from nltk.treetransforms import chomsky_normal_form
from nltk.parse import ChartParser
from nltk.grammar import CFG

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

    # Newer NLTK models for NE chunker
    try:
        nltk.data.find('chunkers/maxent_ne_chunker_tab')
    except LookupError:
        try:
            nltk.download('maxent_ne_chunker_tab')
        except Exception:
            pass

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
        _ensure_nltk_data()
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
            elif operation == 'text_classify':
                return self.text_classification(text)
            elif operation == 'word_embeddings':
                return self.word_embeddings(text)
            elif operation == 'topic_modelling':
                return self.topic_modelling(text)
            elif operation == 'stem':
                return self.stem_text(text)
            elif operation == 'lemmatize':
                return self.lemmatize_text(text)
            elif operation == 'chunk':
                return self.chunk_text(text)
            elif operation == 'chunk_vp':
                return self.chunk_text_vp(text)
            elif operation == 'cfg_parse':
                return self.cfg_parse(text)
            elif operation == 'cnf': # ** FIX: Changed from 'cnf_transform' to 'cnf' to match the frontend **
                return self.cnf_transform(text)
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
    
    def text_classification(self, text):
        if not text or not text.strip():
            return {'error': 'No text provided.'}
        # Simple keyword-based topic classifier for educational purposes
        labels_keywords = {
            'Technology': ['software', 'computer', 'ai', 'machine learning', 'algorithm', 'data', 'device', 'internet', 'app', 'programming', 'code', 'robot', 'cloud', 'model', 'neural', 'server'],
            'Sports': ['game', 'match', 'tournament', 'score', 'team', 'player', 'coach', 'league', 'goal', 'win', 'loss', 'cricket', 'football', 'basketball', 'tennis'],
            'Politics': ['election', 'policy', 'government', 'minister', 'parliament', 'vote', 'campaign', 'president', 'prime minister', 'bill', 'law', 'party', 'diplomacy'],
            'Business': ['market', 'revenue', 'profit', 'loss', 'startup', 'company', 'stocks', 'share', 'merger', 'acquisition', 'sales', 'strategy', 'customer', 'growth'],
            'Health': ['doctor', 'disease', 'treatment', 'diet', 'exercise', 'hospital', 'medicine', 'vaccine', 'mental', 'fitness', 'covid', 'symptom', 'therapy'],
            'Education': ['school', 'college', 'university', 'student', 'teacher', 'exam', 'curriculum', 'classroom', 'assignment', 'lecture', 'research', 'degree'],
            'Entertainment': ['movie', 'music', 'song', 'film', 'actor', 'actress', 'series', 'show', 'concert', 'theater', 'celebrity', 'bollywood', 'hollywood'],
            'Science': ['experiment', 'research', 'theory', 'physics', 'chemistry', 'biology', 'laboratory', 'hypothesis', 'astronomy', 'genetics', 'quantum']
        }
        text_lower = text.lower()
        scores = {label: 0 for label in labels_keywords}
        keywords_found = {label: [] for label in labels_keywords}
        for label, kws in labels_keywords.items():
            for kw in kws:
                pattern = r'\b' + re.escape(kw) + r'(s|es)?\b'
                matches = re.findall(pattern, text_lower)
                if matches:
                    count = len(matches)
                    scores[label] += count
                    keywords_found[label].append(kw)
        total = sum(scores.values())
        probabilities = {label: (score / total if total > 0 else 0.0) for label, score in scores.items()}
        if total > 0:
            predicted_label = max(probabilities, key=probabilities.get)
            confidence = probabilities[predicted_label]
        else:
            predicted_label = 'Unknown'
            confidence = 0.0
        return {
            'predicted_label': predicted_label,
            'confidence': confidence,
            'probabilities': probabilities,
            'keywords_found': keywords_found,
            'visualization': {
                'type': 'classification_chart',
                'data': probabilities
            }
        }

    def topic_modelling(self, text, num_topics=3, num_words=5):
        """
        Perform topic modelling using LDA.
        :param text: Input text (string or list of documents)
        :param num_topics: Number of topics to extract
        :param num_words: Number of top words per topic
        :return: dict with topics and their top words
        """
        try:
            # Handle input
            if isinstance(text, str):
                # Prefer splitting by paragraphs for richer topics; fallback to sentences
                docs = [doc.strip() for doc in text.split("\n") if doc.strip()]
                if not docs:
                    docs = sent_tokenize(text)
            else:
                docs = text

            if not isinstance(docs, (list, tuple)) or len(docs) == 0:
                return {'error': 'No documents found. Provide multiple lines (each line = one document) or a longer text.'}

            n_docs = len(docs)
            # Choose safer df thresholds for small corpora to avoid conflicts
            min_df_param = 1 if n_docs < 3 else 2
            max_df_param = n_docs if n_docs < 3 else 0.9

            # First attempt
            try:
                vectorizer = CountVectorizer(stop_words='english', max_df=max_df_param, min_df=min_df_param)
                X = vectorizer.fit_transform(docs)
            except Exception as ve:
                logging.warning(f"CountVectorizer failed with min_df={min_df_param}, max_df={max_df_param}: {ve}. Retrying with min_df=1, max_df=1.0")
                # Fallback attempt for very small or sparse inputs
                vectorizer = CountVectorizer(stop_words='english', max_df=1.0, min_df=1)
                X = vectorizer.fit_transform(docs)

            n_features = X.shape[1]
            if n_features == 0:
                return {'error': 'No vocabulary after vectorization. Provide richer input (avoid only stopwords) or more documents.'}

            # Ensure topics do not exceed available features
            n_topics = max(1, min(int(num_topics or 1), n_features))

            # Fit LDA
            lda = LatentDirichletAllocation(n_components=n_topics, random_state=42)
            doc_topic_matrix = lda.fit_transform(X)
            words = vectorizer.get_feature_names_out()

            # Extract topics
            topics = []
            for idx, topic in enumerate(lda.components_):
                top_indices = topic.argsort()[-int(num_words or 5):][::-1]
                top_words = [words[i] for i in top_indices]
                topics.append({
                    'topic': idx + 1,
                    'words': top_words,
                    'distribution': topic.tolist()
                })

            return {
                'num_topics': n_topics,
                'topics': topics,
                'doc_topic_matrix': doc_topic_matrix.tolist(),
                'visualization': {
                    'type': 'topic_words',
                    'data': topics
                }
            }
        except Exception as e:
            logging.error(f"Topic modelling failed: {e}")
            return {'error': f'Topic modelling failed: {str(e)}'}

    def word_embeddings(self, text, model_name='glove-wiki-gigaword-50'):
        """
        Generate word embeddings for each word in the text using a pre-trained model.
        Falls back to a deterministic hash-based embedding if gensim or its models
        are unavailable.
        :param text: Input text
        :param model_name: Pre-trained model name (default: 'glove-wiki-gigaword-50')
        :return: dict with word and its embedding vector
        """
        tokens = word_tokenize(text.lower())
        # First, try using gensim pretrained vectors if available
        try:
            from gensim.models import KeyedVectors  # noqa: F401
            import gensim.downloader as api
            from sklearn.decomposition import PCA

            model = api.load(model_name)
            embeddings = {}
            vectors = []
            words_for_projection = []
            oov_map = {}
            import hashlib
            # Use unique tokens to avoid duplicates and ensure one point per word
            unique_tokens = []
            for w in tokens:
                if w not in unique_tokens:
                    unique_tokens.append(w)
            for word in unique_tokens:
                if word in getattr(model, 'key_to_index', {}):
                    vec = model[word]
                    embeddings[word] = vec.tolist()
                    oov_map[word] = False
                else:
                    embeddings[word] = None  # Out-of-vocabulary for this model
                    # Deterministic hash-based vector so OOV words still appear in projection
                    h = int(hashlib.md5(word.encode('utf-8')).hexdigest()[:8], 16)
                    rng = np.random.default_rng(h)
                    vec = rng.normal(0, 1, int(getattr(model, 'vector_size', 50))).astype(float)
                    oov_map[word] = True
                vectors.append(vec)
                words_for_projection.append(word)

            projection = None
            if len(vectors) >= 2:
                try:
                    pca = PCA(n_components=2)
                    coords = pca.fit_transform(np.array(vectors))
                    projection = {w: {'x': float(x), 'y': float(y), 'oov': bool(oov_map.get(w, False))} for (w, (x, y)) in zip(words_for_projection, coords)}
                except Exception as pe:
                    logging.warning(f"PCA projection failed: {pe}")

            return {
                'model': model_name,
                'embeddings': embeddings,
                'dimension': int(getattr(model, 'vector_size', 0)),
                'visualization': {
                    'type': 'embedding_projection',
                    'data': projection or {}
                }
            }
        except Exception as e:
            logging.warning(f"Gensim embeddings unavailable, using hash-based fallback: {e}")
            # Fallback: deterministic hash-based embeddings (pure numpy)
            try:
                import hashlib
                from sklearn.decomposition import PCA

                dim = 50  # match default GloVe size for consistency
                embeddings = {}
                vectors = []
                words = []

                for word in tokens:
                    # Deterministic seed per word using md5 hash
                    h = int(hashlib.md5(word.encode('utf-8')).hexdigest()[:8], 16)
                    rng = np.random.default_rng(h)
                    vec = rng.normal(0, 1, dim).astype(float)
                    embeddings[word] = vec.tolist()
                    vectors.append(vec)
                    words.append(word)

                projection = None
                if len(vectors) >= 2:
                    try:
                        pca = PCA(n_components=2)
                        coords = pca.fit_transform(np.vstack(vectors))
                        projection = {w: {'x': float(x), 'y': float(y), 'oov': False} for w, (x, y) in zip(words, coords)}
                    except Exception as pe:
                        logging.warning(f"PCA projection failed: {pe}")

                return {
                    'model': f'hash-embedding-{dim}',
                    'embeddings': embeddings,
                    'dimension': dim,
                    'visualization': {
                        'type': 'embedding_projection',
                        'data': projection or {}
                    }
                }
            except Exception as fe:
                logging.error(f"Word embedding failed in fallback: {fe}")
                return {'error': f'Word embedding failed: {str(fe)}'}

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
            # Build ASCII tree visualization
            ascii_tree = str(chunks)
            try:
                s_buf = io.StringIO()
                chunks.pretty_print(stream=s_buf)
                ascii_tree = s_buf.getvalue()
            except Exception:
                pass
            return {
                'tagged_tokens': tagged,
                'noun_phrases': noun_phrases,
                'chunk_tree': str(chunks),
                'ascii_tree': ascii_tree,
                'visualization': {'type': 'chunk_tree', 'data': str(chunks)}
            }
        except Exception as e:
            return {
                'error': f'Chunking failed: {str(e)}',
                'tagged_tokens': tagged,
                'noun_phrases': [],
                'chunk_tree': ''
            }
    
    def chunk_text_vp(self, text):
        tokens = word_tokenize(text)
        tagged = pos_tag(tokens)
        grammar = r"""
            NP: {<DT|PP\$>?<JJ>*<NN|NNS|NNP|NNPS>+}
            PP: {<IN><NP>}
            VP: {<MD>?<VB.*><RB.*>*<VB.*>*<NP|PP|PRP>*}
        """
        try:
            cp = nltk.RegexpParser(grammar)
            chunks = cp.parse(tagged)
            verb_phrases = []
            for subtree in chunks.subtrees():
                if subtree.label() == 'VP':
                    phrase = ' '.join([word for (word, pos) in subtree.leaves()])
                    verb_phrases.append(phrase)
            # Build ASCII tree visualization
            ascii_tree = str(chunks)
            try:
                s_buf = io.StringIO()
                chunks.pretty_print(stream=s_buf)
                ascii_tree = s_buf.getvalue()
            except Exception:
                pass
            return {
                'tagged_tokens': tagged,
                'verb_phrases': verb_phrases,
                'chunk_tree': str(chunks),
                'ascii_tree': ascii_tree
            }
        except Exception as e:
            return {
                'error': f'VP Chunking failed: {str(e)}',
                'tagged_tokens': tagged,
                'verb_phrases': [],
                'chunk_tree': ''
            }

    def cfg_parse(self, text):
        try:
        # Take first sentence (CFG parsers usually work on single sentences)
            sentences = sent_tokenize(text)
            sentence = sentences[0] if sentences else text
            tokens = [t.lower() for t in word_tokenize(sentence)]
            tagged = pos_tag(tokens)

        # Basic grammar skeleton
            grammar_rules = [
    "S -> NP VP | S CC S | INTJ NP VP",
    "NP -> DT NN | DT JJ NN | DT JJ JJ NN | PRP | NN | NP PP",
    "VP -> VBZ NP | VBD NP | VBP NP | VB NP | VBZ | VBD | VBP | VB | VP RB | VP PP",
    "PP -> IN NP",
    "INTJ -> UH",

    # POS categories with example words (can be extended dynamically)
    "RB -> 'quickly' | 'suddenly' | 'slowly'",
    "CC -> 'and' | 'or' | 'but'",
    "DT -> 'a' | 'an' | 'the'",
    "PRP -> 'i' | 'he' | 'she' | 'they' | 'we'",
    "NN -> 'book' | 'table' | 'dog' | 'singer' | 'man' | 'girl'",
    "JJ -> 'good' | 'big' | 'small' | 'happy'",
    "VBZ -> 'is' | 'likes'",
    "VBD -> 'was' | 'saw' | 'liked'",
    "VBP -> 'are' | 'eat' | 'see'",
    "VB -> 'eat' | 'see' | 'like'",
    "IN -> 'on' | 'in' | 'under' | 'over'",
    "UH -> 'wow' | 'oh' | 'hey'"
]



        # POS → CFG categories
            mapping = {
                "NN": "NN", "NNS": "NN", "NNP": "NN", "NNPS": "NN",
                "JJ": "JJ", "JJR": "JJ", "JJS": "JJ",
                "VB": "VB", "VBD": "VBD", "VBP": "VBP", "VBZ": "VBZ",
                "DT": "DT", "PRP": "PRP", "IN": "IN", "RB": "RB", "UH": "UH", "CC": "CC"
            }

        # Add terminal rules dynamically for all words in the sentence
            for word, pos in tagged:
                if pos in mapping:
                    grammar_rules.append(f"{mapping[pos]} -> '{word.lower()}'")

            grammar = CFG.fromstring("\n".join(grammar_rules))
            parser = ChartParser(grammar)

            trees = list(parser.parse(tokens))
            ascii_tree = ''
            if trees:
                try:
                    s_buf = io.StringIO()
                    trees[0].pretty_print(stream=s_buf)
                    ascii_tree = s_buf.getvalue()
                except Exception:
                    ascii_tree = str(trees[0])

            result = {
            'tokens': tokens,
            'tagged_tokens': tagged,
            'num_trees': len(trees),
            'parse_trees': [str(t) for t in trees[:3]],
            'chunk_tree': str(trees[0]) if trees else '',
            'ascii_tree': ascii_tree
            }
            if not trees:
                result['error'] = 'No parse found with auto-generated CFG for this sentence.'
            return result
        except Exception as e:
            logging.error(f"CFG parse failed: {e}")
            return {'error': f'CFG parse failed: {str(e)}'}

    def cnf_transform(self, text, to_cnf=True):
        try:
            sentence = sent_tokenize(text)[0]
            tokens = [t.lower() for t in word_tokenize(sentence)]
            tagged = pos_tag(tokens)

            # A more flexible base grammar
            grammar_rules = [
    "S -> NP VP | S CC S | INTJ NP VP",
    "NP -> DT NN | DT JJ NN | DT JJ JJ NN | PRP | NN | NP PP",
    "VP -> VBZ NP | VBD NP | VBP NP | VB NP | VBZ | VBD | VBP | VB | VP RB | VP PP",
    "PP -> IN NP",
    "INTJ -> UH",

    # POS categories with example words (can be extended dynamically)
    "RB -> 'quickly' | 'suddenly' | 'slowly'",
    "CC -> 'and' | 'or' | 'but'",
    "DT -> 'a' | 'an' | 'the'",
    "PRP -> 'i' | 'he' | 'she' | 'they' | 'we'",
    "NN -> 'book' | 'table' | 'dog' | 'singer' | 'man' | 'girl'",
    "JJ -> 'good' | 'big' | 'small' | 'happy'",
    "VBZ -> 'is' | 'likes'",
    "VBD -> 'was' | 'saw' | 'liked'",
    "VBP -> 'are' | 'eat' | 'see'",
    "VB -> 'eat' | 'see' | 'like'",
    "IN -> 'on' | 'in' | 'under' | 'over'",
    "UH -> 'wow' | 'oh' | 'hey'"
]



        # POS → CFG categories
            pos_map = {
                "NN": "NN", "NNS": "NN", "NNP": "NN", "NNPS": "NN",
                "JJ": "JJ", "JJR": "JJ", "JJS": "JJ",
                "VB": "VB", "VBD": "VBD", "VBP": "VBP", "VBZ": "VBZ",
                "DT": "DT", "PRP": "PRP", "IN": "IN", "RB": "RB", "UH": "UH", "CC": "CC"
            }

            for word, tag in tagged:
                if tag in pos_map:
                    rule = f"{pos_map[tag]} -> '{word}'"
                    if rule not in grammar_rules: grammar_rules.append(rule)
            
            grammar = CFG.fromstring("\n".join(grammar_rules))
            parser = ChartParser(grammar)
            trees = list(parser.parse(tokens))

            if not trees:
                return {'error': 'Could not parse the sentence with the dynamic grammar.', 'tokens': tokens}
            
            original_tree = trees[0]
            result_tree = original_tree
            
            if to_cnf:
                cnf_tree = copy.deepcopy(original_tree)
                chomsky_normal_form(cnf_tree)
                result_tree = cnf_tree
            
            ascii_tree = ""
            try:
                s_buf = io.StringIO()
                result_tree.pretty_print(stream=s_buf)
                ascii_tree = s_buf.getvalue()
            except: ascii_tree = str(result_tree)

            return {
                'tokens': tokens,
                'original_tree': str(original_tree),
                'final_tree': str(result_tree), # Generic key for both CFG and CNF
                'chunk_tree': str(result_tree),
                'ascii_tree': ascii_tree
            }
        except Exception as e:
            return {'error': f'Parsing failed: {e}'}

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

    
    def question_answer(self, question, context):
        """
        Answer a question given a context using a pre-trained QA model.
        :param question: The question string
        :param context: The context string
        :return: dict with answer and score
        """
        try:
            from transformers import pipeline
            qa_pipeline = pipeline("question-answering", model="distilbert-base-cased-distilled-squad")
            result = qa_pipeline(question=question, context=context)
            return {'answer': result['answer'], 'score': result['score']}
        except Exception as e:
            return {'error': f'Question answering failed: {str(e)}'}
            
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
