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

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

try:
    nltk.data.find('taggers/averaged_perceptron_tagger')
except LookupError:
    nltk.download('averaged_perceptron_tagger')

try:
    nltk.data.find('taggers/averaged_perceptron_tagger_eng')
except LookupError:
    nltk.download('averaged_perceptron_tagger_eng')

try:
    nltk.data.find('chunkers/maxent_ne_chunker')
except LookupError:
    nltk.download('maxent_ne_chunker')

try:
    nltk.data.find('chunkers/maxent_ne_chunker_tab')
except LookupError:
    nltk.download('maxent_ne_chunker_tab')

try:
    nltk.data.find('corpora/words')
except LookupError:
    nltk.download('words')

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

try:
    nltk.data.find('vader_lexicon')
except LookupError:
    nltk.download('vader_lexicon')

class NLPProcessor:
    def __init__(self):
        self.stemmer = PorterStemmer()
        self.lemmatizer = WordNetLemmatizer()
        self.sentiment_analyzer = SentimentIntensityAnalyzer()
        self.stop_words = set(stopwords.words('english'))
        
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
        # Original text
        original = text
        
        # Lowercase
        lowercased = text.lower()
        
        # Remove punctuation
        no_punct = lowercased.translate(str.maketrans('', '', string.punctuation))
        
        # Tokenize
        tokens = word_tokenize(no_punct)
        
        # Remove stopwords
        filtered_tokens = [word for word in tokens if word not in self.stop_words]
        
        # Stem
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
        
        # Group by POS tag
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
        
        # Count frequencies
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
        
        # Extract entities
        named_entities = []
        current_entity = []
        current_label = None
        
        for chunk in entities:
            if hasattr(chunk, 'label'):
                # This is a named entity
                entity_words = [token for token, pos in chunk]
                named_entities.append({
                    'entity': ' '.join(entity_words),
                    'label': chunk.label(),
                    'words': entity_words
                })
            else:
                # This is a regular word
                if current_entity:
                    current_entity = []
                    current_label = None
        
        # Group by entity type
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
        
        # Determine overall sentiment
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
        
        # Define a simple noun phrase chunk pattern
        grammar = r"""
            NP: {<DT|PP\$>?<JJ>*<NN>}
                {<NNP>+}
        """
        
        try:
            cp = nltk.RegexpParser(grammar)
            chunks = cp.parse(tagged)
            
            # Extract noun phrases
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
    
    def get_quiz_questions(self, module_name):
        """Get quiz questions for a specific module"""
        questions = {
            'text_preprocessing': [
                {
                    'question': 'What is the primary purpose of text preprocessing in NLP?',
                    'options': [
                        'To make text more readable for humans',
                        'To convert raw text into a clean, standardized format suitable for analysis',
                        'To increase the size of text data',
                        'To encrypt text for security'
                    ],
                    'correct': 1,
                    'explanation': 'Text preprocessing transforms raw text into a clean, standardized format that is more suitable for NLP algorithms and analysis.'
                },
                {
                    'question': 'Which of the following is NOT a common step in text preprocessing?',
                    'options': [
                        'Tokenization',
                        'Lowercasing',
                        'Image enhancement',
                        'Stopword removal'
                    ],
                    'correct': 2,
                    'explanation': 'Image enhancement is a computer vision technique, not a text preprocessing step. The others are all common text preprocessing steps.'
                },
                {
                    'question': 'What is tokenization in text preprocessing?',
                    'options': [
                        'Removing punctuation marks',
                        'Splitting text into individual words or tokens',
                        'Converting text to lowercase',
                        'Removing common words'
                    ],
                    'correct': 1,
                    'explanation': 'Tokenization is the process of breaking text into individual words, phrases, symbols, or other meaningful elements called tokens.'
                },
                {
                    'question': 'Why is converting text to lowercase a common preprocessing step?',
                    'options': [
                        'To reduce the vocabulary size by treating "Word" and "word" as the same',
                        'To make the text look better',
                        'To comply with grammar rules',
                        'To increase processing speed'
                    ],
                    'correct': 0,
                    'explanation': 'Lowercasing reduces vocabulary size by treating words with different cases as the same token, which helps in reducing sparsity in NLP models.'
                },
                {
                    'question': 'What are stopwords in text preprocessing?',
                    'options': [
                        'Words that are misspelled',
                        'Common words that usually don\'t carry much meaning (e.g., "the", "and", "is")',
                        'Words that are longer than 10 characters',
                        'Words that contain numbers'
                    ],
                    'correct': 1,
                    'explanation': 'Stopwords are common words like "the", "and", "is", "in" that occur frequently but typically don\'t carry much meaning for text analysis.'
                },
                {
                    'question': 'Which of the following is an example of a stopword?',
                    'options': ['machine', 'learning', 'the', 'algorithm'],
                    'correct': 2,
                    'explanation': '"The" is a common stopword that is often removed during text preprocessing because it doesn\'t carry significant meaning.'
                },
                {
                    'question': 'What is the main purpose of removing punctuation in text preprocessing?',
                    'options': [
                        'To make the text shorter',
                        'To reduce noise and focus on meaningful words',
                        'To comply with writing standards',
                        'To make text look cleaner'
                    ],
                    'correct': 1,
                    'explanation': 'Removing punctuation helps reduce noise in the text data and allows algorithms to focus on the meaningful words and their relationships.'
                },
                {
                    'question': 'What is stemming in text preprocessing?',
                    'options': [
                        'Removing the last few characters of a word to get its root form',
                        'Converting all words to uppercase',
                        'Removing duplicate words',
                        'Adding prefixes to words'
                    ],
                    'correct': 0,
                    'explanation': 'Stemming is a technique that removes the suffixes of words to get their root form (e.g., "running" becomes "run").'
                },
                {
                    'question': 'What is lemmatization in text preprocessing?',
                    'options': [
                        'Removing words that are too long',
                        'Using vocabulary and morphological analysis to get the base form of a word',
                        'Combining similar words together',
                        'Removing words with special characters'
                    ],
                    'correct': 1,
                    'explanation': 'Lemmatization uses vocabulary and morphological analysis to convert words to their base or dictionary form (lemma).'
                },
                {
                    'question': 'What is the main difference between stemming and lemmatization?',
                    'options': [
                        'Stemming is faster but less accurate; lemmatization is slower but more accurate',
                        'Stemming always produces real words; lemmatization does not',
                        'There is no difference; they are the same technique',
                        'Stemming is used for English only; lemmatization works for all languages'
                    ],
                    'correct': 0,
                    'explanation': 'Stemming uses simple rule-based approaches that may produce non-real words, while lemmatization uses vocabulary and morphological analysis to produce real dictionary words.'
                },
                {
                    'question': 'Which of the following is a potential drawback of stopword removal?',
                    'options': [
                        'It increases processing time',
                        'It can remove contextually important words in some cases',
                        'It increases the vocabulary size',
                        'It makes text harder to read'
                    ],
                    'correct': 1,
                    'explanation': 'Stopword removal can sometimes remove contextually important words, especially in cases where stopwords like "not" or "no" change the meaning of a sentence.'
                },
                {
                    'question': 'What is the purpose of text normalization in preprocessing?',
                    'options': [
                        'To make all text documents the same length',
                        'To convert text into a standard format to ensure consistency',
                        'To improve the visual appearance of text',
                        'To compress text data'
                    ],
                    'correct': 1,
                    'explanation': 'Text normalization converts text into a standard format to ensure consistency, such as converting all text to lowercase or standardizing date formats.'
                },
                {
                    'question': 'Which preprocessing step would be most appropriate for handling "USA", "U.S.A.", and "United States"?',
                    'options': [
                        'Tokenization',
                        'Stopword removal',
                        'Text normalization',
                        'Stemming'
                    ],
                    'correct': 2,
                    'explanation': 'Text normalization would standardize different representations of the same entity to a single form.'
                },
                {
                    'question': 'What is the effect of text preprocessing on vocabulary size?',
                    'options': [
                        'It always increases vocabulary size',
                        'It has no effect on vocabulary size',
                        'It typically reduces vocabulary size',
                        'It makes vocabulary size unpredictable'
                    ],
                    'correct': 2,
                    'explanation': 'Text preprocessing techniques like lowercasing and removing punctuation typically reduce vocabulary size by standardizing different forms of words.'
                },
                {
                    'question': 'Which technique would be most appropriate for handling URLs in text preprocessing?',
                    'options': [
                        'Stopword removal',
                        'Regular expressions',
                        'Tokenization',
                        'Lemmatization'
                    ],
                    'correct': 1,
                    'explanation': 'Regular expressions are commonly used to identify and handle URLs, email addresses, and other specific patterns in text.'
                },
                {
                    'question': 'What is the main challenge in handling numbers during text preprocessing?',
                    'options': [
                        'Numbers are always important to keep',
                        'Deciding whether to keep, remove, or normalize numbers',
                        'Numbers are difficult to tokenize',
                        'Numbers cannot be lowercased'
                    ],
                    'correct': 1,
                    'explanation': 'The challenge with numbers is deciding whether to keep them, remove them, or normalize them (e.g., converting "2" and "two" to the same representation).'
                },
                {
                    'question': 'Which of the following is a benefit of lemmatization over stemming?',
                    'options': [
                        'Lemmatization is faster',
                        'Lemmatization produces real dictionary words',
                        'Lemmatization is simpler to implement',
                        'Lemmatization works for all languages'
                    ],
                    'correct': 1,
                    'explanation': 'Lemmatization produces real dictionary words as the root form, making the output more meaningful than stemming which may produce non-words.'
                },
                {
                    'question': 'What is the purpose of expanding contractions like "don\'t" in text preprocessing?',
                    'options': [
                        'To make the text longer',
                        'To standardize text representations',
                        'To increase vocabulary size',
                        'To preserve original text format'
                    ],
                    'correct': 1,
                    'explanation': 'Expanding contractions standardizes text by converting abbreviated forms to their full versions, ensuring consistency in text analysis.'
                },
                {
                    'question': 'Which preprocessing technique would help in treating "running", "ran", and "runs" as the same word?',
                    'options': [
                        'Tokenization',
                        'Stopword removal',
                        'Stemming or lemmatization',
                        'Lowercasing'
                    ],
                    'correct': 2,
                    'explanation': 'Both stemming and lemmatization would convert different forms of a word to its root form, treating them as the same word for analysis.'
                },
                {
                    'question': 'What is a common issue that text preprocessing aims to solve?',
                    'options': [
                        'Too much consistency in text data',
                        'Noise and variability in raw text',
                        'Perfect grammar in all texts',
                        'Uniform length of all documents'
                    ],
                    'correct': 1,
                    'explanation': 'Raw text data often contains noise and variability that can interfere with NLP algorithms, and preprocessing aims to reduce these issues.'
                },
                {
                    'question': 'What is the main purpose of using regular expressions in text preprocessing?',
                    'options': [
                        'To count the number of words',
                        'To identify and manipulate specific patterns in text',
                        'To improve grammar in text',
                        'To translate text to another language'
                    ],
                    'correct': 1,
                    'explanation': 'Regular expressions are used to identify and manipulate specific patterns in text, such as removing URLs, email addresses, or other structured patterns.'
                },
                {
                    'question': 'Which of the following is an example of text normalization?',
                    'options': [
                        'Removing all vowels from text',
                        'Converting "USA" and "United States" to the same representation',
                        'Counting word frequencies',
                        'Removing all punctuation'
                    ],
                    'correct': 1,
                    'explanation': 'Converting different representations of the same entity to a single standard form is an example of text normalization.'
                },
                {
                    'question': 'What is the purpose of removing HTML tags during text preprocessing?',
                    'options': [
                        'To make the text look cleaner',
                        'To extract only the textual content from web pages',
                        'To improve the visual appearance of text',
                        'To increase the size of the dataset'
                    ],
                    'correct': 1,
                    'explanation': 'Removing HTML tags helps extract only the textual content from web pages, eliminating markup that is not relevant for text analysis.'
                },
                {
                    'question': 'Which technique would be most appropriate for handling hashtags like #NLP in social media text?',
                    'options': [
                        'Removing them completely',
                        'Separating the # from the word (e.g., #NLP → NLP)',
                        'Converting them to uppercase',
                        'Ignoring them during analysis'
                    ],
                    'correct': 1,
                    'explanation': 'Separating the # from the word allows the content of the hashtag to be analyzed while preserving its meaning.'
                },
                {
                    'question': 'What is the main challenge when dealing with emojis in text preprocessing?',
                    'options': [
                        'They are too colorful',
                        'They are difficult to tokenize and interpret correctly',
                        'They are too small to see',
                        'They slow down processing'
                    ],
                    'correct': 1,
                    'explanation': 'Emojis can be challenging to tokenize and interpret correctly, as they may carry emotional meaning that is important for analysis.'
                },
                {
                    'question': 'Why might you want to preserve some punctuation marks during text preprocessing?',
                    'options': [
                        'To make the text look better',
                        'To maintain sentence structure and meaning (e.g., question marks, exclamation points)',
                        'To increase processing time',
                        'To make the text longer'
                    ],
                    'correct': 1,
                    'explanation': 'Some punctuation marks like question marks and exclamation points carry important meaning and can be useful for sentiment analysis or other tasks.'
                },
                {
                    'question': 'What is the purpose of using a whitelist approach in text preprocessing?',
                    'options': [
                        'To allow only specific characters or patterns to remain in the text',
                        'To remove all text',
                        'To make the text longer',
                        'To increase vocabulary size'
                    ],
                    'correct': 0,
                    'explanation': 'A whitelist approach in text preprocessing involves keeping only specific characters or patterns that are explicitly allowed, removing everything else.'
                },
                {
                    'question': 'Which of the following is a disadvantage of aggressive text preprocessing?',
                    'options': [
                        'It makes text too clean',
                        'It can remove important semantic information',
                        'It takes too little time',
                        'It is too simple to implement'
                    ],
                    'correct': 1,
                    'explanation': 'Aggressive text preprocessing can remove important semantic information, such as negations (not, never) that change the meaning of text.'
                },
                {
                    'question': 'What is the main benefit of using libraries like NLTK or spaCy for text preprocessing?',
                    'options': [
                        'They are slower than custom implementations',
                        'They provide tested, optimized implementations of common preprocessing tasks',
                        'They are more expensive to use',
                        'They only work with English text'
                    ],
                    'correct': 1,
                    'explanation': 'Libraries like NLTK or spaCy provide tested, optimized implementations of common preprocessing tasks, saving development time and ensuring reliability.'
                },
                {
                    'question': 'What is the purpose of handling encoding issues in text preprocessing?',
                    'options': [
                        'To make text look better',
                        'To ensure all characters are correctly represented and processed',
                        'To increase the size of text data',
                        'To remove all special characters'
                    ],
                    'correct': 1,
                    'explanation': 'Handling encoding issues ensures that all characters are correctly represented and processed, preventing errors when working with text from different sources.'
                },
                {
                    'question': 'Which of the following is an example of domain-specific text preprocessing?',
                    'options': [
                        'Lowercasing all text',
                        'Removing medical terminology in medical text analysis',
                        'Tokenizing sentences',
                        'Removing stopwords'
                    ],
                    'correct': 1,
                    'explanation': 'Removing domain-specific terminology (like medical terms in medical text analysis) is an example of domain-specific preprocessing that might be needed for specialized applications.'
                },
                {
                    'question': 'What is the main purpose of spell correction in text preprocessing?',
                    'options': [
                        'To make text look perfect',
                        'To correct typos and improve text quality for analysis',
                        'To increase the length of text',
                        'To remove all vowels'
                    ],
                    'correct': 1,
                    'explanation': 'Spell correction aims to correct typos and improve text quality, which can improve the performance of NLP models.'
                },
                {
                    'question': 'Why might you want to remove rare words during text preprocessing?',
                    'options': [
                        'To increase vocabulary size',
                        'To reduce noise and sparsity in the data',
                        'To make the text shorter',
                        'To improve grammar'
                    ],
                    'correct': 1,
                    'explanation': 'Removing rare words can help reduce noise and sparsity in the data, as they may not contribute meaningfully to analysis and can cause overfitting.'
                },
                {
                    'question': 'What is the effect of text preprocessing on model training time?',
                    'options': [
                        'It always increases training time',
                        'It can reduce training time by reducing data size and complexity',
                        'It has no effect on training time',
                        'It makes models less accurate'
                    ],
                    'correct': 1,
                    'explanation': 'Text preprocessing can reduce training time by reducing data size and complexity, making it easier for models to process the data.'
                },
                {
                    'question': 'Which of the following is a common technique for handling multi-word expressions (MWEs) in preprocessing?',
                    'options': [
                        'Ignoring them completely',
                        'Treating them as single tokens (e.g., "New York" as one token)',
                        'Removing all spaces',
                        'Converting them to uppercase'
                    ],
                    'correct': 1,
                    'explanation': 'Treating multi-word expressions as single tokens helps preserve their meaning and can improve the performance of NLP models.'
                },
                {
                    'question': 'What is the purpose of using language-specific preprocessing techniques?',
                    'options': [
                        'To make text look better',
                        'To account for language-specific characteristics like morphology and syntax',
                        'To increase processing time',
                        'To remove all punctuation'
                    ],
                    'correct': 1,
                    'explanation': 'Language-specific preprocessing techniques account for language-specific characteristics like morphology and syntax, which can improve the quality of text analysis.'
                },
                {
                    'question': 'Which of the following is a potential issue with removing all numbers from text?',
                    'options': [
                        'Numbers are always important',
                        'It might remove contextually important information like dates, ages, or quantities',
                        'It makes text too clean',
                        'It is too difficult to implement'
                    ],
                    'correct': 1,
                    'explanation': 'Removing all numbers might eliminate contextually important information like dates, ages, or quantities that could be relevant for analysis.'
                },
                {
                    'question': 'What is the main advantage of using pipelines in text preprocessing?',
                    'options': [
                        'They are slower than individual steps',
                        'They allow for organized, reproducible sequences of preprocessing steps',
                        'They are more expensive to implement',
                        'They only work with English'
                    ],
                    'correct': 1,
                    'explanation': 'Pipelines allow for organized, reproducible sequences of preprocessing steps, making it easier to apply the same preprocessing to different datasets.'
                },
                {
                    'question': 'Why is it important to consider the order of preprocessing steps?',
                    'options': [
                        'It doesn\'t matter what order you do things in',
                        'Some steps depend on others (e.g., tokenization before stemming)',
                        'It makes processing slower',
                        'It increases memory usage'
                    ],
                    'correct': 1,
                    'explanation': 'The order of preprocessing steps matters because some steps depend on others. For example, tokenization should typically be done before stemming or lemmatization.'
                },
                {
                    'question': 'What is a common approach to handling out-of-vocabulary (OOV) words during preprocessing?',
                    'options': [
                        'Removing them completely',
                        'Replacing them with a special token like <UNK>',
                        'Ignoring them during analysis',
                        'Converting them to uppercase'
                    ],
                    'correct': 1,
                    'explanation': 'Replacing out-of-vocabulary words with a special token like <UNK> allows models to handle unknown words gracefully during training and inference.'
                },
                {
                    'question': 'Which of the following is a benefit of keeping original text alongside preprocessed text?',
                    'options': [
                        'It doubles the storage requirements',
                        'It allows for interpretability and debugging of results',
                        'It slows down processing',
                        'It makes models less accurate'
                    ],
                    'correct': 1,
                    'explanation': 'Keeping original text alongside preprocessed text allows for interpretability and debugging of results, making it easier to understand what the model is doing.'
                },
                {
                    'question': 'What is the purpose of using character-level preprocessing?',
                    'options': [
                        'To make text longer',
                        'To handle spelling errors and morphological variations at the character level',
                        'To remove all punctuation',
                        'To increase vocabulary size'
                    ],
                    'correct': 1,
                    'explanation': 'Character-level preprocessing can help handle spelling errors and morphological variations by working at the character level rather than the word level.'
                },
                {
                    'question': 'Which technique would be most appropriate for handling user mentions like @username in social media text?',
                    'options': [
                        'Removing them completely',
                        'Replacing them with a generic token like <USER>',
                        'Converting them to uppercase',
                        'Ignoring them during analysis'
                    ],
                    'correct': 1,
                    'explanation': 'Replacing user mentions with a generic token preserves the information that a mention occurred while removing potentially sensitive or identifying information.'
                },
                {
                    'question': 'What is the main challenge of preprocessing text in low-resource languages?',
                    'options': [
                        'There are too many tools available',
                        'Limited availability of language-specific tools and resources',
                        'The text is too clean',
                        'Processing is too fast'
                    ],
                    'correct': 1,
                    'explanation': 'Preprocessing text in low-resource languages is challenging due to the limited availability of language-specific tools, resources, and pre-trained models.'
                },
                {
                    'question': 'Why might you want to perform sentence segmentation as part of preprocessing?',
                    'options': [
                        'To make text look better',
                        'To process text at the sentence level for tasks that require it',
                        'To increase the size of text data',
                        'To remove all punctuation'
                    ],
                    'correct': 1,
                    'explanation': 'Sentence segmentation is useful for tasks that require processing text at the sentence level, such as sentiment analysis or information extraction.'
                },
                {
                    'question': 'What is the purpose of using stopword lists in text preprocessing?',
                    'options': [
                        'To increase vocabulary size',
                        'To define which words should be removed based on language or domain',
                        'To make text longer',
                        'To improve grammar'
                    ],
                    'correct': 1,
                    'explanation': 'Stopword lists define which words should be removed based on language or domain, helping to standardize the preprocessing process.'
                },
                {
                    'question': 'Which of the following is a consideration when preprocessing text for topic modeling?',
                    'options': [
                        'Removing too many words might eliminate topic-specific terms',
                        'Making text as short as possible',
                        'Removing all punctuation',
                        'Converting everything to uppercase'
                    ],
                    'correct': 0,
                    'explanation': 'When preprocessing text for topic modeling, it\'s important not to remove too many words, as this might eliminate topic-specific terms that are important for identifying topics.'
                },
                {
                    'question': 'What is the main benefit of using parallel processing in text preprocessing?',
                    'options': [
                        'It makes processing slower',
                        'It can significantly speed up preprocessing of large datasets',
                        'It is more expensive to implement',
                        'It only works with English'
                    ],
                    'correct': 1,
                    'explanation': 'Parallel processing can significantly speed up preprocessing of large datasets by distributing the work across multiple processors or cores.'
                }
            ],
            'pos_tagging': [
                {
                    'question': 'What does POS stand for in NLP?',
                    'options': [
                        'Position of Speech',
                        'Part-of-Speech',
                        'Positive or Negative',
                        'Point of Sale'
                    ],
                    'correct': 1,
                    'explanation': 'POS stands for Part-of-Speech, which identifies the grammatical category of each word.'
                },
                {
                    'question': 'What is the primary purpose of POS tagging?',
                    'options': [
                        'To remove punctuation from text',
                        'To identify the grammatical category of each word',
                        'To convert text to lowercase',
                        'To remove stopwords'
                    ],
                    'correct': 1,
                    'explanation': 'POS tagging identifies the grammatical category (noun, verb, adjective, etc.) of each word in a text.'
                },
                {
                    'question': 'Which POS tag represents a noun in the Penn Treebank tagset?',
                    'options': ['VB', 'JJ', 'NN', 'RB'],
                    'correct': 2,
                    'explanation': 'NN is the POS tag for nouns in the Penn Treebank tagset.'
                },
                {
                    'question': 'What does the POS tag "VB" represent?',
                    'options': ['Noun', 'Adjective', 'Verb', 'Adverb'],
                    'correct': 2,
                    'explanation': 'VB is the POS tag for verbs in the Penn Treebank tagset.'
                },
                {
                    'question': 'Which POS tag would be assigned to the word "quickly"?',
                    'options': ['NN', 'JJ', 'VB', 'RB'],
                    'correct': 3,
                    'explanation': 'RB is the POS tag for adverbs, and "quickly" is an adverb modifying how something is done.'
                },
                {
                    'question': 'What is the POS tag for determiners like "the", "a", "an"?',
                    'options': ['DT', 'IN', 'CC', 'PRP'],
                    'correct': 0,
                    'explanation': 'DT is the POS tag for determiners, which include articles like "the", "a", and "an".'
                },
                {
                    'question': 'Which tag represents prepositions like "in", "on", "at"?',
                    'options': ['DT', 'IN', 'CC', 'PRP'],
                    'correct': 1,
                    'explanation': 'IN is the POS tag for prepositions, which show relationships between words.'
                },
                {
                    'question': 'What does the POS tag "JJ" represent?',
                    'options': ['Noun', 'Verb', 'Adjective', 'Adverb'],
                    'correct': 2,
                    'explanation': 'JJ is the POS tag for adjectives, which describe or modify nouns.'
                },
                {
                    'question': 'Which POS tag would be assigned to "running" in "I am running"?',
                    'options': ['NN', 'VBG', 'VBD', 'VBN'],
                    'correct': 1,
                    'explanation': 'VBG is the POS tag for verb gerunds or present participles, like "running".'
                },
                {
                    'question': 'What is the POS tag for plural nouns?',
                    'options': ['NN', 'NNS', 'NNP', 'NNPS'],
                    'correct': 1,
                    'explanation': 'NNS is the POS tag for plural common nouns.'
                },
                {
                    'question': 'Which tag represents coordinating conjunctions like "and", "but", "or"?',
                    'options': ['DT', 'IN', 'CC', 'PRP'],
                    'correct': 2,
                    'explanation': 'CC is the POS tag for coordinating conjunctions that connect words, phrases, or clauses.'
                },
                {
                    'question': 'What does the POS tag "PRP" represent?',
                    'options': ['Possessive pronouns', 'Personal pronouns', 'Prepositions', 'Proper nouns'],
                    'correct': 1,
                    'explanation': 'PRP is the POS tag for personal pronouns like "I", "you", "he", "she", "it", "we", "they".'
                },
                {
                    'question': 'Which POS tag would be assigned to "John" in "John runs"?',
                    'options': ['NN', 'NNS', 'NNP', 'NNPS'],
                    'correct': 2,
                    'explanation': 'NNP is the POS tag for singular proper nouns, which are specific names of people, places, etc.'
                },
                {
                    'question': 'What is the POS tag for cardinal numbers like "one", "two", "three"?',
                    'options': ['NN', 'CD', 'LS', 'SYM'],
                    'correct': 1,
                    'explanation': 'CD is the POS tag for cardinal numbers.'
                },
                {
                    'question': 'Which tag represents punctuation marks?',
                    'options': ['PUNC', 'SYM', '.', 'PUNCT'],
                    'correct': 1,
                    'explanation': 'SYM is the POS tag for symbols, which includes punctuation marks in some tagsets.'
                },
                {
                    'question': 'What does the POS tag "VBD" represent?',
                    'options': ['Base form verb', 'Past tense verb', 'Present participle', 'Past participle'],
                    'correct': 1,
                    'explanation': 'VBD is the POS tag for past tense verbs.'
                },
                {
                    'question': 'Which POS tag would be assigned to "has" in "She has a book"?',
                    'options': ['VB', 'VBD', 'VBG', 'VBZ'],
                    'correct': 3,
                    'explanation': 'VBZ is the POS tag for third person singular present tense verbs.'
                },
                {
                    'question': 'What is the POS tag for possessive nouns like "John\'s"?',
                    'options': ['NN', 'NNS', 'NNP', 'NNP\'s'],
                    'correct': 2,
                    'explanation': 'NNP is used for proper nouns, and possessive forms are typically tagged as NNP with a POS (possessive ending) tag for the apostrophe s.'
                },
                {
                    'question': 'Which tag represents interjections like "oh", "hey", "wow"?',
                    'options': ['IN', 'CC', 'UH', 'EX'],
                    'correct': 2,
                    'explanation': 'UH is the POS tag for interjections, which are exclamations expressing emotion.'
                },
                {
                    'question': 'What is a key benefit of POS tagging in NLP applications?',
                    'options': [
                        'It increases the size of text data',
                        'It helps in understanding syntactic structure and improving downstream tasks',
                        'It removes the need for machine learning',
                        'It makes text unreadable'
                    ],
                    'correct': 1,
                    'explanation': 'POS tagging helps understand syntactic structure, which improves performance in tasks like parsing, information extraction, and machine translation.'
                }
            ],
            'ngram_modeling': [
                {
                    'question': 'What is an n-gram in NLP?',
                    'options': [
                        'A sequence of n consecutive words or tokens',
                        'A neural network with n layers',
                        'A normalization technique for text',
                        'A type of neural network architecture'
                    ],
                    'correct': 0,
                    'explanation': 'An n-gram is a contiguous sequence of n items (words, characters, etc.) from a given sample of text or speech.'
                },
                {
                    'question': 'What is a unigram?',
                    'options': [
                        'A sequence of one word',
                        'A sequence of two words',
                        'A sequence of three words',
                        'A sequence of four words'
                    ],
                    'correct': 0,
                    'explanation': 'A unigram is an n-gram with n=1, which means it consists of a single word or token.'
                },
                {
                    'question': 'What is a bigram?',
                    'options': [
                        'A sequence of one word',
                        'A sequence of two consecutive words',
                        'A sequence of three consecutive words',
                        'A sequence of four consecutive words'
                    ],
                    'correct': 1,
                    'explanation': 'A bigram is an n-gram with n=2, which means it consists of two consecutive words or tokens.'
                },
                {
                    'question': 'What is a trigram?',
                    'options': [
                        'A sequence of one word',
                        'A sequence of two words',
                        'A sequence of three consecutive words',
                        'A sequence of four words'
                    ],
                    'correct': 2,
                    'explanation': 'A trigram is an n-gram with n=3, which means it consists of three consecutive words or tokens.'
                },
                {
                    'question': 'Which of the following is the primary use of n-gram models?',
                    'options': [
                        'Image recognition',
                        'Text prediction and language modeling',
                        'Audio processing',
                        'Database management'
                    ],
                    'correct': 1,
                    'explanation': 'N-gram models are primarily used for text prediction, language modeling, and calculating probabilities of word sequences.'
                },
                {
                    'question': 'What does the term "smoothing" refer to in n-gram models?',
                    'options': [
                        'Making the text visually appealing',
                        'Technique to handle unseen n-grams with zero probability',
                        'Reducing the size of the model',
                        'Speeding up the model training'
                    ],
                    'correct': 1,
                    'explanation': 'Smoothing is a technique used to handle the problem of zero probability for n-grams that were not seen in the training data.'
                },
                {
                    'question': 'Which n-gram model would be most appropriate for capturing the context "I always ___ coffee in the morning"?',
                    'options': ['Unigram', 'Bigram', 'Trigram', '4-gram'],
                    'correct': 2,
                    'explanation': 'A trigram would be most appropriate as it considers the context of the two preceding words ("always" and "coffee") to predict the next word.'
                },
                {
                    'question': 'What is the main advantage of using higher-order n-grams (like 4-grams or 5-grams)?',
                    'options': [
                        'They require less memory',
                        'They capture longer context dependencies',
                        'They are faster to compute',
                        'They are easier to implement'
                    ],
                    'correct': 1,
                    'explanation': 'Higher-order n-grams capture longer context dependencies, which can improve the accuracy of language models.'
                },
                {
                    'question': 'What is a major drawback of using higher-order n-grams?',
                    'options': [
                        'They are too simple to capture context',
                        'They require more computational resources and suffer from data sparsity',
                        'They are not useful for language modeling',
                        'They cannot be used with neural networks'
                    ],
                    'correct': 1,
                    'explanation': 'Higher-order n-grams require exponentially more memory and data to estimate accurately, leading to data sparsity issues.'
                },
                {
                    'question': 'In n-gram modeling, what does the Markov assumption state?',
                    'options': [
                        'The probability of a word depends on all previous words',
                        'The probability of a word depends only on a fixed number of previous words',
                        'Words are independent of each other',
                        'The probability of a word is always 0.5'
                    ],
                    'correct': 1,
                    'explanation': 'The Markov assumption in n-gram modeling states that the probability of a word depends only on the previous (n-1) words.'
                },
                {
                    'question': 'What is the purpose of add-one smoothing (Laplace smoothing) in n-gram models?',
                    'options': [
                        'To make the model run faster',
                        'To assign a small probability to unseen n-grams',
                        'To remove duplicate n-grams',
                        'To convert words to lowercase'
                    ],
                    'correct': 1,
                    'explanation': 'Add-one smoothing assigns a small probability to n-grams that were not seen in the training data by adding one to all counts.'
                },
                {
                    'question': 'Which of the following is a limitation of n-gram models?',
                    'options': [
                        'They are too complex to implement',
                        'They cannot capture long-range dependencies',
                        'They require too much training data',
                        'They are not probabilistic models'
                    ],
                    'correct': 1,
                    'explanation': 'N-gram models have limited context window and cannot capture long-range dependencies that may be important for understanding text.'
                },
                {
                    'question': 'How is the probability of a sentence calculated in n-gram models?',
                    'options': [
                        'By multiplying the probabilities of individual words',
                        'By multiplying the probabilities of n-grams in the sentence',
                        'By adding the probabilities of n-grams in the sentence',
                        'By averaging the probabilities of all words'
                    ],
                    'correct': 1,
                    'explanation': 'The probability of a sentence in n-gram models is calculated by multiplying the conditional probabilities of each word given its context.'
                },
                {
                    'question': 'What is the main difference between character-level and word-level n-grams?',
                    'options': [
                        'Character-level n-grams are faster to compute',
                        'Word-level n-grams capture meaning better',
                        'Character-level n-grams use characters as tokens instead of words',
                        'There is no difference between them'
                    ],
                    'correct': 2,
                    'explanation': 'Character-level n-grams use individual characters as tokens, while word-level n-grams use words as tokens.'
                },
                {
                    'question': 'Which n-gram model would be most suitable for spell checking applications?',
                    'options': ['Word-level unigrams', 'Word-level bigrams', 'Character-level n-grams', 'Sentence-level n-grams'],
                    'correct': 2,
                    'explanation': 'Character-level n-grams are most suitable for spell checking as they can capture patterns in character sequences to identify misspellings.'
                },
                {
                    'question': 'What is perplexity in the context of n-gram models?',
                    'options': [
                        'A measure of model complexity',
                        'A measure of how well a model predicts sample text',
                        'A measure of training time',
                        'A measure of vocabulary size'
                    ],
                    'correct': 1,
                    'explanation': 'Perplexity is a measure of how well a language model predicts sample text, with lower values indicating better performance.'
                },
                {
                    'question': 'Why are n-gram models still relevant despite the advent of neural language models?',
                    'options': [
                        'They are more accurate than neural models',
                        'They are simpler, faster, and useful as baselines or components in hybrid systems',
                        'They are easier to implement than neural models',
                        'They require less data than neural models'
                    ],
                    'correct': 1,
                    'explanation': 'N-gram models are still relevant because they are computationally efficient, interpretable, and useful as baselines or components in more complex systems.'
                },
                {
                    'question': 'What is the effect of increasing the value of n in n-gram models?',
                    'options': [
                        'Decreases model accuracy',
                        'Increases the context considered for predictions',
                        'Reduces computational requirements',
                        'Makes the model deterministic'
                    ],
                    'correct': 1,
                    'explanation': 'Increasing n increases the context considered for predictions, but also increases data sparsity and computational requirements.'
                },
                {
                    'question': 'Which technique can help mitigate the data sparsity problem in n-gram models?',
                    'options': [
                        'Using only unigrams',
                        'Increasing the training data size or using smoothing techniques',
                        'Reducing the vocabulary size',
                        'Using only bigrams'
                    ],
                    'correct': 1,
                    'explanation': 'Increasing training data size or using smoothing techniques like add-one smoothing can help mitigate data sparsity in n-gram models.'
                },
                {
                    'question': 'What is the primary advantage of n-gram models over more complex language models?',
                    'options': [
                        'They are always more accurate',
                        'They are computationally efficient and easy to understand',
                        'They can capture long-range dependencies',
                        'They require no training data'
                    ],
                    'correct': 1,
                    'explanation': 'N-gram models are computationally efficient and easy to understand, making them practical for many applications despite their limitations.'
                },
                {
                    'question': 'Which of the following is a common application of n-gram models?',
                    'options': [
                        'Image recognition',
                        'Speech recognition and text prediction',
                        'Database management',
                        'Network security'
                    ],
                    'correct': 1,
                    'explanation': 'N-gram models are commonly used in speech recognition systems and text prediction features like autocomplete.'
                },
                {
                    'question': 'What is the main idea behind backoff models in n-gram smoothing?',
                    'options': [
                        'Always use the highest order n-gram available',
                        'Use lower-order n-grams when higher-order ones are not reliable',
                        'Remove all n-grams with low frequencies',
                        'Double the count of all n-grams'
                    ],
                    'correct': 1,
                    'explanation': 'Backoff models use lower-order n-grams (like bigrams or unigrams) when higher-order n-grams have insufficient data or are unreliable.'
                },
                {
                    'question': 'In n-gram modeling, what does the term "out-of-vocabulary" (OOV) refer to?',
                    'options': [
                        'Words that appear too frequently in the training data',
                        'Words that were not seen during training',
                        'Words that are too long to process',
                        'Words that contain special characters'
                    ],
                    'correct': 1,
                    'explanation': 'Out-of-vocabulary words are those that were not present in the training data and therefore have no associated n-gram probabilities.'
                },
                {
                    'question': 'Which technique is used to handle numerical underflow in n-gram probability calculations?',
                    'options': [
                        'Integer arithmetic',
                        'Logarithmic transformations',
                        'Rounding to nearest integer',
                        'Multiplying by a large constant'
                    ],
                    'correct': 1,
                    'explanation': 'Logarithmic transformations are used to handle numerical underflow by converting products to sums, which are more numerically stable.'
                },
                {
                    'question': 'What is the primary benefit of using interpolated smoothing in n-gram models?',
                    'options': [
                        'It completely eliminates zero probabilities',
                        'It combines probabilities from different n-gram orders',
                        'It reduces the model size',
                        'It speeds up training'
                    ],
                    'correct': 1,
                    'explanation': 'Interpolated smoothing combines probabilities from different n-gram orders (unigrams, bigrams, trigrams, etc.) to provide more robust estimates.'
                },
                {
                    'question': 'Which of the following best describes the "curse of dimensionality" in n-gram models?',
                    'options': [
                        'Models become too simple to capture language patterns',
                        'The number of possible n-grams grows exponentially with n',
                        'Models become too fast to be useful',
                        'All n-grams have the same probability'
                    ],
                    'correct': 1,
                    'explanation': 'The curse of dimensionality refers to the exponential growth in the number of possible n-grams as n increases, leading to data sparsity issues.'
                },
                {
                    'question': 'What is the purpose of pruning in n-gram models?',
                    'options': [
                        'To make the model more complex',
                        'To remove infrequent n-grams and reduce model size',
                        'To increase training time',
                        'To add more vocabulary words'
                    ],
                    'correct': 1,
                    'explanation': 'Pruning removes infrequent n-grams from the model to reduce its size and improve efficiency without significantly affecting performance.'
                },
                {
                    'question': 'In the context of n-grams, what is meant by "domain adaptation"?',
                    'options': [
                        'Changing the programming language used',
                        'Adapting a model trained on one domain to work better on another',
                        'Converting text to uppercase',
                        'Removing punctuation marks'
                    ],
                    'correct': 1,
                    'explanation': 'Domain adaptation refers to techniques for adapting an n-gram model trained on one type of text (e.g., news articles) to perform better on another type (e.g., social media text).'
                },
                {
                    'question': 'Which of the following is a disadvantage of using very high-order n-grams (e.g., 10-grams)?',
                    'options': [
                        'They are too simple to capture context',
                        'They suffer from severe data sparsity and require enormous amounts of training data',
                        'They are too fast to compute',
                        'They cannot be used with smoothing techniques'
                    ],
                    'correct': 1,
                    'explanation': 'Very high-order n-grams suffer from severe data sparsity because the number of possible n-grams grows exponentially, requiring enormous amounts of training data to estimate reliably.'
                },
                {
                    'question': 'What does the term "discounting" refer to in n-gram smoothing techniques?',
                    'options': [
                        'Reducing the probability mass of seen n-grams to allocate to unseen ones',
                        'Applying percentage discounts to model prices',
                        'Removing punctuation marks from text',
                        'Converting all words to lowercase'
                    ],
                    'correct': 0,
                    'explanation': 'Discounting refers to reducing the probability mass of observed n-grams slightly to allocate some probability to unseen n-grams, which is a key concept in smoothing techniques.'
                },
                {
                    'question': 'Which of the following is a key consideration when choosing the value of n in an n-gram model?',
                    'options': [
                        'Always use the highest possible value',
                        'Balance between context sensitivity and data requirements',
                        'Use only unigrams for better performance',
                        'Match the number of available processors'
                    ],
                    'correct': 1,
                    'explanation': 'Choosing the value of n involves balancing between capturing enough context for good predictions and managing the data requirements and computational complexity.'
                },
                {
                    'question': 'What is the primary reason for using character-level n-grams instead of word-level n-grams in some applications?',
                    'options': [
                        'They are always more accurate',
                        'They can handle out-of-vocabulary words and spelling variations better',
                        'They require more storage space',
                        'They are easier to implement'
                    ],
                    'correct': 1,
                    'explanation': 'Character-level n-grams can handle out-of-vocabulary words and spelling variations better because they can still make predictions based on character patterns even for unseen words.'
                },
                {
                    'question': 'In n-gram models, what is the purpose of using a "vocabulary cutoff"?',
                    'options': [
                        'To increase the number of unique words',
                        'To replace rare words with a special token to manage vocabulary size',
                        'To remove all punctuation marks',
                        'To convert text to uppercase'
                    ],
                    'correct': 1,
                    'explanation': 'A vocabulary cutoff replaces rare words with a special token (like <UNK>) to manage vocabulary size and handle out-of-vocabulary words more effectively.'
                }
            ],
            'named_entity_recognition': [
                {
                    'question': 'What does NER stand for in NLP?',
                    'options': [
                        'Natural Entity Recognition',
                        'Named Entity Recognition',
                        'Neural Entity Recognition',
                        'Noun Entity Recognition'
                    ],
                    'correct': 1,
                    'explanation': 'NER stands for Named Entity Recognition, which identifies and classifies named entities in text.'
                },
                {
                    'question': 'What is the primary goal of Named Entity Recognition?',
                    'options': [
                        'To remove punctuation from text',
                        'To identify and classify named entities like persons, organizations, and locations',
                        'To convert text to lowercase',
                        'To remove stopwords from text'
                    ],
                    'correct': 1,
                    'explanation': 'The primary goal of NER is to locate and classify named entities mentioned in unstructured text into predefined categories.'
                },
                {
                    'question': 'Which of the following is typically NOT considered a named entity?',
                    'options': ['Barack Obama', 'New York', 'the', 'Google'],
                    'correct': 2,
                    'explanation': 'Common words like "the" are not named entities. Named entities are specific proper nouns like persons, locations, organizations, etc.'
                },
                {
                    'question': 'What category would "John Smith" typically be classified as in NER?',
                    'options': ['LOCATION', 'ORGANIZATION', 'PERSON', 'MISC'],
                    'correct': 2,
                    'explanation': 'John Smith is a person\'s name, so it would be classified as a PERSON entity in most NER systems.'
                },
                {
                    'question': 'What category would "London" typically be classified as in NER?',
                    'options': ['PERSON', 'ORGANIZATION', 'LOCATION', 'MISC'],
                    'correct': 2,
                    'explanation': 'London is a city, which is a geographical location, so it would be classified as a LOCATION entity.'
                },
                {
                    'question': 'What category would "Microsoft" typically be classified as in NER?',
                    'options': ['PERSON', 'ORGANIZATION', 'LOCATION', 'MISC'],
                    'correct': 1,
                    'explanation': 'Microsoft is a company, which is an organization, so it would be classified as an ORGANIZATION entity.'
                },
                {
                    'question': 'Which of the following is a common challenge in Named Entity Recognition?',
                    'options': [
                        'Identifying too many stopwords',
                        'Ambiguity in entity boundaries and types',
                        'Removing punctuation marks',
                        'Converting text to lowercase'
                    ],
                    'correct': 1,
                    'explanation': 'Ambiguity in entity boundaries and types is a common challenge in NER, as the same word can have different meanings in different contexts.'
                },
                {
                    'question': 'What does the term "entity disambiguation" refer to in NER?',
                    'options': [
                        'Removing duplicate entities from text',
                        'Determining the correct category for an entity when multiple interpretations are possible',
                        'Converting entities to lowercase',
                        'Removing entities that are too long'
                    ],
                    'correct': 1,
                    'explanation': 'Entity disambiguation refers to determining the correct category or meaning of an entity when multiple interpretations are possible.'
                },
                {
                    'question': 'Which NER category would "July 4, 1776" typically belong to?',
                    'options': ['PERSON', 'ORGANIZATION', 'LOCATION', 'DATE'],
                    'correct': 3,
                    'explanation': 'July 4, 1776 is a specific date, so it would be classified as a DATE entity in most NER systems.'
                },
                {
                    'question': 'What is a common approach to implementing Named Entity Recognition?',
                    'options': [
                        'Using regular expressions only',
                        'Using machine learning models trained on annotated data',
                        'Removing all punctuation',
                        'Converting all text to uppercase'
                    ],
                    'correct': 1,
                    'explanation': 'NER is commonly implemented using machine learning models trained on annotated data, though rule-based approaches can also be used.'
                },
                {
                    'question': 'Which of the following is an example of a nested named entity?',
                    'options': [
                        'New York',
                        'United States of America',
                        'John Smith',
                        'Google Inc.'
                    ],
                    'correct': 1,
                    'explanation': 'United States of America is an example of a nested entity because it contains multiple words that together form a single entity.'
                },
                {
                    'question': 'What is the main advantage of using pre-trained NER models?',
                    'options': [
                        'They are always more accurate than custom models',
                        'They require less training data and computational resources',
                        'They are easier to implement than rule-based systems',
                        'They work perfectly for all domains'
                    ],
                    'correct': 1,
                    'explanation': 'Pre-trained NER models require less training data and computational resources since they are already trained on large datasets.'
                },
                {
                    'question': 'Which NER category would "$100" typically belong to?',
                    'options': ['PERSON', 'ORGANIZATION', 'LOCATION', 'MONEY'],
                    'correct': 3,
                    'explanation': 'Dollar amounts are typically classified as MONEY entities in NER systems.'
                },
                {
                    'question': 'What is a limitation of rule-based NER approaches?',
                    'options': [
                        'They are too fast',
                        'They require too much training data',
                        'They are difficult to maintain and don\'t generalize well to new domains',
                        'They are too accurate'
                    ],
                    'correct': 2,
                    'explanation': 'Rule-based NER approaches are difficult to maintain and often don\'t generalize well to new domains or languages.'
                },
                {
                    'question': 'In the context of NER, what is "chunking"?',
                    'options': [
                        'Removing chunks of text',
                        'Grouping words into meaningful phrases or entities',
                        'Converting text to lowercase chunks',
                        'Breaking sentences into chunks'
                    ],
                    'correct': 1,
                    'explanation': 'In NER, chunking refers to grouping words into meaningful phrases or entities, which is a key step in identifying named entities.'
                },
                {
                    'question': 'Which of the following is a benefit of Named Entity Recognition in information extraction?',
                    'options': [
                        'It increases the size of documents',
                        'It helps structure unstructured text by identifying important entities',
                        'It removes all punctuation',
                        'It converts text to uppercase'
                    ],
                    'correct': 1,
                    'explanation': 'NER helps structure unstructured text by identifying and categorizing important entities, making it easier to extract and analyze information.'
                },
                {
                    'question': 'What category would "CEO" typically be classified as in NER?',
                    'options': ['TITLE', 'PERSON', 'ORGANIZATION', 'LOCATION'],
                    'correct': 0,
                    'explanation': 'CEO (Chief Executive Officer) is a job title, so it would typically be classified as a TITLE entity in NER systems that recognize this category.'
                },
                {
                    'question': 'Which NLP task is closely related to Named Entity Recognition?',
                    'options': [
                        'Sentiment analysis',
                        'Part-of-speech tagging',
                        'Text summarization',
                        'Machine translation'
                    ],
                    'correct': 1,
                    'explanation': 'Part-of-speech tagging is closely related to NER as both involve identifying and classifying words in text, and POS tags can help inform NER decisions.'
                },
                {
                    'question': 'What is the main difference between Named Entity Recognition and Named Entity Linking?',
                    'options': [
                        'NER identifies entities while linking connects them to knowledge bases',
                        'NER is faster than linking',
                        'NER works only with English text',
                        'NER requires less data than linking'
                    ],
                    'correct': 0,
                    'explanation': 'NER identifies and classifies entities in text, while Named Entity Linking connects these entities to entries in knowledge bases like Wikipedia or DBpedia.'
                },
                {
                    'question': 'Which of the following is a common evaluation metric for NER systems?',
                    'options': [
                        'Accuracy only',
                        'Precision, recall, and F1-score',
                        'Processing speed only',
                        'Vocabulary size'
                    ],
                    'correct': 1,
                    'explanation': 'NER systems are typically evaluated using precision, recall, and F1-score, which measure the accuracy of entity identification and classification.'
                }
            ],
            'sentiment_analysis': [
                {
                    'question': 'What is the primary purpose of sentiment analysis in NLP?',
                    'options': [
                        'To count the number of words in a document',
                        'To determine the emotional tone or attitude expressed in text',
                        'To translate text from one language to another',
                        'To remove punctuation from text'
                    ],
                    'correct': 1,
                    'explanation': 'Sentiment analysis determines the emotional tone or attitude expressed in text, typically classifying it as positive, negative, or neutral.'
                },
                {
                    'question': 'Which of the following is NOT a common sentiment classification?',
                    'options': ['Positive', 'Negative', 'Neutral', 'Grammatical'],
                    'correct': 3,
                    'explanation': 'Grammatical is not a sentiment classification. The common sentiment classifications are positive, negative, and neutral.'
                },
                {
                    'question': 'What does VADER sentiment analysis tool stand for?',
                    'options': [
                        'Visual Analysis of Document Emotional Response',
                        'Valence Aware Dictionary and sEntiment Reasoner',
                        'Very Accurate Determination of Emotional Ratings',
                        'Verifiable Algorithm for Determining Emotional Responses'
                    ],
                    'correct': 1,
                    'explanation': 'VADER stands for Valence Aware Dictionary and sEntiment Reasoner, a sentiment analysis tool.'
                },
                {
                    'question': 'Which type of text is VADER particularly good at analyzing?',
                    'options': [
                        'Formal academic papers',
                        'Social media text with emoticons and slang',
                        'Legal documents',
                        'Scientific research papers'
                    ],
                    'correct': 1,
                    'explanation': 'VADER is specifically designed for social media text and handles emoticons, slang, and informal language well.'
                },
                {
                    'question': 'What is the compound score in VADER sentiment analysis?',
                    'options': [
                        'A measure of text complexity',
                        'A normalized score between -1 (most negative) and +1 (most positive)',
                        'A count of positive words',
                        'A measure of vocabulary diversity'
                    ],
                    'correct': 1,
                    'explanation': 'The compound score in VADER is a normalized score between -1 (most negative) and +1 (most positive), providing an overall sentiment rating.'
                },
                {
                    'question': 'Which of the following is a challenge in sentiment analysis?',
                    'options': [
                        'Processing speed',
                        'Sarcasm and context understanding',
                        'Text formatting',
                        'Word counting'
                    ],
                    'correct': 1,
                    'explanation': 'Sarcasm and context understanding are major challenges in sentiment analysis because the same words can have different meanings in different contexts.'
                },
                {
                    'question': 'What is the typical range for sentiment scores in many sentiment analysis tools?',
                    'options': [
                        '0 to 100',
                        '0 to 1',
                        '-1 to +1',
                        '1 to 10'
                    ],
                    'correct': 2,
                    'explanation': 'Many sentiment analysis tools use a range from -1 (most negative) to +1 (most positive) for their sentiment scores.'
                },
                {
                    'question': 'Which sentiment category would the text "This movie is terrible!" typically be classified as?',
                    'options': ['Positive', 'Negative', 'Neutral', 'Mixed'],
                    'correct': 1,
                    'explanation': 'The text "This movie is terrible!" expresses a negative opinion, so it would be classified as negative sentiment.'
                },
                {
                    'question': 'What is a lexicon-based approach to sentiment analysis?',
                    'options': [
                        'Using machine learning models only',
                        'Using predefined dictionaries of words with sentiment scores',
                        'Using neural networks exclusively',
                        'Using only punctuation analysis'
                    ],
                    'correct': 1,
                    'explanation': 'Lexicon-based approaches use predefined dictionaries of words with sentiment scores to determine the overall sentiment of text.'
                },
                {
                    'question': 'Which of the following is an advantage of machine learning approaches to sentiment analysis?',
                    'options': [
                        'They are faster than lexicon-based approaches',
                        'They can learn from data and adapt to specific domains',
                        'They require no training data',
                        'They are simpler to implement'
                    ],
                    'correct': 1,
                    'explanation': 'Machine learning approaches can learn from training data and adapt to specific domains or types of text.'
                },
                {
                    'question': 'What does the term "polarity" refer to in sentiment analysis?',
                    'options': [
                        'The grammatical structure of sentences',
                        'The positive or negative orientation of text',
                        'The frequency of word usage',
                        'The complexity of vocabulary'
                    ],
                    'correct': 1,
                    'explanation': 'Polarity in sentiment analysis refers to the positive or negative orientation of text.'
                },
                {
                    'question': 'Which sentiment analysis tool is specifically designed to handle social media text?',
                    'options': ['NLTK Sentiment Analyzer', 'VADER', 'TextBlob', 'Stanford NLP'],
                    'correct': 1,
                    'explanation': 'VADER (Valence Aware Dictionary and sEntiment Reasoner) is specifically designed to handle social media text, including emoticons and slang.'
                },
                {
                    'question': 'What is a common preprocessing step in sentiment analysis?',
                    'options': [
                        'Removing all punctuation',
                        'Converting text to uppercase',
                        'Handling negations and intensifiers',
                        'Removing all vowels'
                    ],
                    'correct': 2,
                    'explanation': 'Handling negations (like "not") and intensifiers (like "very") is important in sentiment analysis because they can significantly change the sentiment.'
                },
                {
                    'question': 'Which of the following would likely have a positive sentiment score?',
                    'options': [
                        'I hate this product',
                        'This is the worst experience ever',
                        'I love this new feature',
                        'I am disappointed with the service'
                    ],
                    'correct': 2,
                    'explanation': 'The statement "I love this new feature" expresses a positive opinion and would likely have a positive sentiment score.'
                },
                {
                    'question': 'What is the main difference between binary and ternary sentiment classification?',
                    'options': [
                        'Binary uses machine learning, ternary uses lexicons',
                        'Binary has two categories (positive/negative), ternary has three (positive/negative/neutral)',
                        'Binary is faster than ternary',
                        'Binary is more accurate than ternary'
                    ],
                    'correct': 1,
                    'explanation': 'Binary sentiment classification has two categories (typically positive and negative), while ternary has three categories (positive, negative, and neutral).'
                },
                {
                    'question': 'Which of the following is a limitation of lexicon-based sentiment analysis?',
                    'options': [
                        'It is too fast',
                        'It may not capture context and sarcasm well',
                        'It requires too much training data',
                        'It is too complex to implement'
                    ],
                    'correct': 1,
                    'explanation': 'Lexicon-based approaches may not capture context, sarcasm, or domain-specific language well, as they rely on predefined word scores.'
                },
                {
                    'question': 'What is the purpose of using stopword removal in sentiment analysis?',
                    'options': [
                        'To increase processing time',
                        'To focus on sentiment-bearing words',
                        'To make the text longer',
                        'To preserve all information in the text'
                    ],
                    'correct': 1,
                    'explanation': 'Stopword removal in sentiment analysis helps focus on sentiment-bearing words by removing common words that typically don\'t contribute to sentiment.'
                },
                {
                    'question': 'Which sentiment category would the text "The weather is okay" typically be classified as?',
                    'options': ['Positive', 'Negative', 'Neutral', 'Mixed'],
                    'correct': 2,
                    'explanation': 'The text "The weather is okay" expresses a neutral opinion, neither strongly positive nor negative.'
                },
                {
                    'question': 'What is a common evaluation metric for sentiment analysis models?',
                    'options': [
                        'Vocabulary size',
                        'Accuracy, precision, recall, and F1-score',
                        'Text length',
                        'Number of sentences'
                    ],
                    'correct': 1,
                    'explanation': 'Common evaluation metrics for sentiment analysis models include accuracy, precision, recall, and F1-score.'
                },
                {
                    'question': 'Which of the following is a benefit of sentiment analysis in business applications?',
                    'options': [
                        'It increases data storage requirements',
                        'It helps understand customer opinions and feedback',
                        'It makes text longer',
                        'It removes the need for customer service'
                    ],
                    'correct': 1,
                    'explanation': 'Sentiment analysis helps businesses understand customer opinions and feedback, which can inform product development and customer service improvements.'
                },
                {
                    'question': 'What is the main advantage of using deep learning models for sentiment analysis?',
                    'options': [
                        'They require less computational power',
                        'They can automatically learn complex features from raw text',
                        'They are simpler to implement than rule-based approaches',
                        'They work well with very small datasets'
                    ],
                    'correct': 1,
                    'explanation': 'Deep learning models can automatically learn complex features from raw text, reducing the need for manual feature engineering and often achieving better performance.'
                },
                {
                    'question': 'Which of the following is a common technique for handling negations in sentiment analysis?',
                    'options': [
                        'Ignoring negation words completely',
                        'Using special tokens or modifying sentiment scores for negated words',
                        'Removing all negation words from text',
                        'Treating negation words as positive sentiment'
                    ],
                    'correct': 1,
                    'explanation': 'Handling negations often involves using special tokens or modifying sentiment scores for words that are negated, as negation can reverse the sentiment of a phrase.'
                },
                {
                    'question': 'What is the purpose of using ensemble methods in sentiment analysis?',
                    'options': [
                        'To reduce computational requirements',
                        'To combine predictions from multiple models for improved accuracy',
                        'To simplify model interpretation',
                        'To decrease the amount of training data needed'
                    ],
                    'correct': 1,
                    'explanation': 'Ensemble methods combine predictions from multiple models, which often leads to improved accuracy by reducing variance and capturing different aspects of the data.'
                },
                {
                    'question': 'Which sentiment analysis approach would be most appropriate for analyzing product reviews with domain-specific language?',
                    'options': [
                        'Using a generic pre-trained model without fine-tuning',
                        'Fine-tuning a pre-trained model on domain-specific data',
                        'Using only rule-based approaches',
                        'Removing all domain-specific terms'
                    ],
                    'correct': 1,
                    'explanation': 'Fine-tuning a pre-trained model on domain-specific data allows the model to adapt to the specific language and sentiment patterns of product reviews.'
                },
                {
                    'question': 'What is the main challenge when applying sentiment analysis to multilingual text?',
                    'options': [
                        'All languages have the same sentiment patterns',
                        'Language-specific sentiment expressions and cultural differences',
                        'Multilingual text is always shorter',
                        'Translation is never needed'
                    ],
                    'correct': 1,
                    'explanation': 'Multilingual sentiment analysis is challenging due to language-specific sentiment expressions, cultural differences in expressing opinions, and varying linguistic structures.'
                },
                {
                    'question': 'What is the role of attention mechanisms in neural sentiment analysis models?',
                    'options': [
                        'To reduce the model size',
                        'To focus on the most relevant parts of text for sentiment determination',
                        'To speed up training',
                        'To eliminate the need for preprocessing'
                    ],
                    'correct': 1,
                    'explanation': 'Attention mechanisms help neural models focus on the most relevant parts of text for determining sentiment, improving interpretability and performance.'
                },
                {
                    'question': 'Which of the following is a benefit of aspect-based sentiment analysis?',
                    'options': [
                        'It provides sentiment for the entire document only',
                        'It identifies sentiment toward specific aspects or features of a product/service',
                        'It is simpler to implement than document-level analysis',
                        'It requires less training data'
                    ],
                    'correct': 1,
                    'explanation': 'Aspect-based sentiment analysis identifies sentiment toward specific aspects or features of a product or service, providing more detailed insights than document-level analysis.'
                },
                {
                    'question': 'What is the main purpose of using transfer learning in sentiment analysis?',
                    'options': [
                        'To increase the amount of training data needed',
                        'To leverage knowledge from related tasks or domains to improve performance',
                        'To make models more complex',
                        'To eliminate the need for labeled data'
                    ],
                    'correct': 1,
                    'explanation': 'Transfer learning leverages knowledge from related tasks or domains (like pre-trained language models) to improve sentiment analysis performance, especially with limited labeled data.'
                },
                {
                    'question': 'Which of the following is a common approach to handle sarcasm in sentiment analysis?',
                    'options': [
                        'Ignoring sarcastic text completely',
                        'Using contextual features and specialized models trained on sarcastic examples',
                        'Treating all sarcastic text as positive',
                        'Removing all sarcastic expressions'
                    ],
                    'correct': 1,
                    'explanation': 'Handling sarcasm in sentiment analysis often requires specialized models trained on sarcastic examples and contextual features, as sarcasm can be challenging to detect automatically.'
                },
                {
                    'question': 'What is the primary benefit of using a hybrid approach combining rule-based and machine learning methods for sentiment analysis?',
                    'options': [
                        'It is always faster than using either approach alone',
                        'It can leverage the strengths of both approaches for improved accuracy',
                        'It requires less training data than machine learning alone',
                        'It eliminates the need for preprocessing'
                    ],
                    'correct': 1,
                    'explanation': 'A hybrid approach combining rule-based and machine learning methods can leverage the strengths of both approaches, potentially achieving higher accuracy than either method alone.'
                }
            ],
            'text_classification': [
                {
                    'question': 'What is the primary goal of text classification in NLP?',
                    'options': [
                        'To convert text to speech',
                        'To categorize text documents into predefined classes or categories',
                        'To remove punctuation from text',
                        'To translate text between languages'
                    ],
                    'correct': 1,
                    'explanation': 'Text classification aims to categorize text documents into predefined classes or categories based on their content.'
                },
                {
                    'question': 'Which of the following is a common application of text classification?',
                    'options': [
                        'Image recognition',
                        'Email spam detection',
                        'Audio processing',
                        'Video compression'
                    ],
                    'correct': 1,
                    'explanation': 'Email spam detection is a classic application of text classification where emails are classified as spam or not spam.'
                },
                {
                    'question': 'What is a labeled dataset in the context of text classification?',
                    'options': [
                        'A dataset with no categories',
                        'A dataset where each document is associated with its correct category',
                        'A dataset with only numerical data',
                        'A dataset with encrypted text'
                    ],
                    'correct': 1,
                    'explanation': 'A labeled dataset in text classification is one where each document is associated with its correct category or class label.'
                },
                {
                    'question': 'Which of the following is NOT a typical step in text classification?',
                    'options': [
                        'Feature extraction',
                        'Model training',
                        'Image enhancement',
                        'Prediction'
                    ],
                    'correct': 2,
                    'explanation': 'Image enhancement is a computer vision technique, not a typical step in text classification. Common steps include feature extraction, model training, and prediction.'
                },
                {
                    'question': 'What is the "bag of words" model in text classification?',
                    'options': [
                        'A model that considers word order',
                        'A model that represents text as an unordered collection of words',
                        'A model that only uses stop words',
                        'A model that counts punctuation marks'
                    ],
                    'correct': 1,
                    'explanation': 'The bag of words model represents text as an unordered collection of words, ignoring grammar and word order but keeping track of word frequencies.'
                },
                {
                    'question': 'Which algorithm is commonly used for text classification?',
                    'options': [
                        'K-means clustering',
                        'Naive Bayes',
                        'Linear regression',
                        'Decision tree for regression'
                    ],
                    'correct': 1,
                    'explanation': 'Naive Bayes is a commonly used algorithm for text classification due to its effectiveness with high-dimensional data like text.'
                },
                {
                    'question': 'What is TF-IDF in text classification?',
                    'options': [
                        'Term Frequency-Inverse Document Frequency',
                        'Text Format-Inverse Data Format',
                        'Term Filtering-Inverse Document Filtering',
                        'Text Frequency-Inverse Data Frequency'
                    ],
                    'correct': 0,
                    'explanation': 'TF-IDF stands for Term Frequency-Inverse Document Frequency, a statistical measure used to evaluate the importance of a word in a document.'
                },
                {
                    'question': 'What is overfitting in text classification?',
                    'options': [
                        'When the model is too simple',
                        'When the model performs well on training data but poorly on new data',
                        'When there are too few features',
                        'When the dataset is too small'
                    ],
                    'correct': 1,
                    'explanation': 'Overfitting occurs when a model learns the training data too well, including its noise and outliers, resulting in poor performance on new, unseen data.'
                },
                {
                    'question': 'Which of the following is a binary classification problem?',
                    'options': [
                        'Classifying news articles into politics, sports, and technology',
                        'Classifying emails as spam or not spam',
                        'Classifying movies into genres',
                        'Classifying products into categories'
                    ],
                    'correct': 1,
                    'explanation': 'Classifying emails as spam or not spam is a binary classification problem because there are only two possible classes.'
                },
                {
                    'question': 'What is cross-validation used for in text classification?',
                    'options': [
                        'To increase the dataset size',
                        'To evaluate model performance more reliably by splitting data into multiple train/test sets',
                        'To remove outliers from the data',
                        'To convert text to numerical features'
                    ],
                    'correct': 1,
                    'explanation': 'Cross-validation is used to evaluate model performance more reliably by splitting the data into multiple training and testing sets, providing a more robust estimate of model performance.'
                },
                {
                    'question': 'What is precision in the context of text classification evaluation?',
                    'options': [
                        'The proportion of actual positives correctly identified',
                        'The proportion of predicted positives that are actually positive',
                        'The overall accuracy of the model',
                        'The ratio of true negatives to false positives'
                    ],
                    'correct': 1,
                    'explanation': 'Precision is the proportion of predicted positives that are actually positive, measuring the accuracy of positive predictions.'
                },
                {
                    'question': 'What is recall in the context of text classification evaluation?',
                    'options': [
                        'The proportion of actual positives correctly identified',
                        'The proportion of predicted positives that are actually positive',
                        'The overall accuracy of the model',
                        'The ratio of true negatives to false positives'
                    ],
                    'correct': 0,
                    'explanation': 'Recall is the proportion of actual positives that are correctly identified, measuring the completeness of positive identifications.'
                },
                {
                    'question': 'Which of the following is a challenge in text classification?',
                    'options': [
                        'Having too much labeled data',
                        'Dealing with imbalanced datasets',
                        'Having perfectly clean text data',
                        'Using only numerical features'
                    ],
                    'correct': 1,
                    'explanation': 'Imbalanced datasets, where some classes have significantly more examples than others, are a common challenge in text classification.'
                },
                {
                    'question': 'What is the purpose of feature selection in text classification?',
                    'options': [
                        'To increase the number of features',
                        'To select the most relevant features to improve model performance and reduce complexity',
                        'To remove all textual features',
                        'To standardize feature scales'
                    ],
                    'correct': 1,
                    'explanation': 'Feature selection aims to select the most relevant features to improve model performance, reduce overfitting, and decrease computational complexity.'
                },
                {
                    'question': 'What is the difference between supervised and unsupervised text classification?',
                    'options': [
                        'Supervised uses labeled data, unsupervised uses unlabeled data',
                        'Supervised is faster than unsupervised',
                        'Supervised uses fewer features than unsupervised',
                        'There is no difference'
                    ],
                    'correct': 0,
                    'explanation': 'Supervised text classification uses labeled training data, while unsupervised text classification works with unlabeled data, often using clustering techniques.'
                },
                {
                    'question': 'Which text classification approach groups similar documents together without predefined categories?',
                    'options': [
                        'Supervised classification',
                        'Binary classification',
                        'Clustering',
                        'Multi-class classification'
                    ],
                    'correct': 2,
                    'explanation': 'Clustering is an unsupervised approach that groups similar documents together without using predefined categories.'
                },
                {
                    'question': 'What is the F1-score in text classification?',
                    'options': [
                        'The average of precision and recall',
                        'The harmonic mean of precision and recall',
                        'The product of precision and recall',
                        'The difference between precision and recall'
                    ],
                    'correct': 1,
                    'explanation': 'The F1-score is the harmonic mean of precision and recall, providing a single metric that balances both measures.'
                },
                {
                    'question': 'What is stemming commonly used for in text classification?',
                    'options': [
                        'To increase the dataset size',
                        'To reduce words to their root form to normalize text',
                        'To add more punctuation',
                        'To convert text to uppercase'
                    ],
                    'correct': 1,
                    'explanation': 'Stemming is used to reduce words to their root form, helping to normalize text by treating different forms of the same word as identical.'
                },
                {
                    'question': 'Which of the following is a benefit of using deep learning for text classification?',
                    'options': [
                        'Requires less computational power',
                        'Can automatically learn relevant features from raw text',
                        'Is simpler to implement than traditional methods',
                        'Works well with very small datasets'
                    ],
                    'correct': 1,
                    'explanation': 'Deep learning models can automatically learn relevant features from raw text, reducing the need for manual feature engineering.'
                },
                {
                    'question': 'What is the main advantage of ensemble methods in text classification?',
                    'options': [
                        'They are faster to train',
                        'They combine multiple models to often achieve better performance than individual models',
                        'They require less memory',
                        'They are easier to interpret'
                    ],
                    'correct': 1,
                    'explanation': 'Ensemble methods combine multiple models, often achieving better performance than individual models by reducing variance and bias.'
                },
                {
                    'question': 'What is the primary purpose of feature engineering in text classification?',
                    'options': [
                        'To reduce the size of the dataset',
                        'To create meaningful representations of text data for machine learning models',
                        'To remove all punctuation from text',
                        'To translate text into multiple languages'
                    ],
                    'correct': 1,
                    'explanation': 'Feature engineering involves creating meaningful representations of text data that can be used by machine learning algorithms to make accurate predictions.'
                },
                {
                    'question': 'Which of the following is a common technique for handling class imbalance in text classification?',
                    'options': [
                        'Removing all minority class examples',
                        'Oversampling the minority class or undersampling the majority class',
                        'Using only unigrams as features',
                        'Converting all text to uppercase'
                    ],
                    'correct': 1,
                    'explanation': 'Class imbalance can be addressed through resampling techniques like oversampling the minority class or undersampling the majority class to create a more balanced dataset.'
                },
                {
                    'question': 'What is the purpose of using n-gram features in text classification?',
                    'options': [
                        'To capture only single words',
                        'To capture local word order and phrases that convey meaning',
                        'To reduce the vocabulary size',
                        'To remove stopwords from text'
                    ],
                    'correct': 1,
                    'explanation': 'N-gram features (like bigrams and trigrams) help capture local word order and phrases that can be important for understanding the context and meaning of text.'
                },
                {
                    'question': 'Which evaluation metric is most appropriate when dealing with imbalanced datasets in text classification?',
                    'options': [
                        'Accuracy only',
                        'F1-score, precision, and recall',
                        'Training time',
                        'Vocabulary size'
                    ],
                    'correct': 1,
                    'explanation': 'In imbalanced datasets, accuracy can be misleading. F1-score, precision, and recall provide a better picture of model performance across different classes.'
                },
                {
                    'question': 'What is the main benefit of using word embeddings as features in text classification?',
                    'options': [
                        'They require less computational power',
                        'They capture semantic relationships between words',
                        'They are easier to implement than traditional methods',
                        'They work only with English text'
                    ],
                    'correct': 1,
                    'explanation': 'Word embeddings capture semantic relationships between words, allowing the model to understand that similar words have similar representations, which can improve classification performance.'
                }
            ],
            'word_embeddings': [
                {
                    'question': 'What is the primary purpose of word embeddings in NLP?',
                    'options': [
                        'To compress text data',
                        'To convert words into numerical vectors that capture semantic meaning',
                        'To remove stopwords from text',
                        'To translate text between languages'
                    ],
                    'correct': 1,
                    'explanation': 'Word embeddings convert words into numerical vectors that capture semantic meaning and relationships between words.'
                },
                {
                    'question': 'Which of the following is a key advantage of word embeddings over traditional encoding methods like one-hot encoding?',
                    'options': [
                        'They are faster to compute',
                        'They capture semantic relationships between words',
                        'They require less memory',
                        'They are easier to implement'
                    ],
                    'correct': 1,
                    'explanation': 'Word embeddings capture semantic relationships between words, unlike one-hot encoding which treats all words as equally distant from each other.'
                },
                {
                    'question': 'What does it mean for word embeddings to capture semantic similarity?',
                    'options': [
                        'They can identify grammatical errors',
                        'Similar words have similar vector representations',
                        'They can translate between languages',
                        'They can generate new words'
                    ],
                    'correct': 1,
                    'explanation': 'Semantic similarity in word embeddings means that similar words have similar vector representations, allowing the model to understand relationships between words.'
                },
                {
                    'question': 'Which of the following is a popular pre-trained word embedding model?',
                    'options': [
                        'VADER',
                        'WordNet',
                        'Word2Vec',
                        'NLTK'
                    ],
                    'correct': 2,
                    'explanation': 'Word2Vec is a popular pre-trained word embedding model developed by Google that learns vector representations of words.'
                },
                {
                    'question': 'What is the dimensionality of word embeddings?',
                    'options': [
                        'Always 100 dimensions',
                        'Fixed at 10 dimensions',
                        'Variable, depending on the model and training',
                        'Always equal to the vocabulary size'
                    ],
                    'correct': 2,
                    'explanation': 'The dimensionality of word embeddings is variable and depends on the specific model and training configuration, commonly ranging from 50 to 300 dimensions or more.'
                },
                {
                    'question': 'In Word2Vec, what is the Continuous Bag of Words (CBOW) approach?',
                    'options': [
                        'Predicts target word from context words',
                        'Predicts context words from target word',
                        'Counts word frequencies',
                        'Removes stop words'
                    ],
                    'correct': 0,
                    'explanation': 'CBOW predicts the target word based on its surrounding context words, learning embeddings that capture contextual relationships.'
                },
                {
                    'question': 'In Word2Vec, what is the Skip-gram approach?',
                    'options': [
                        'Predicts target word from context words',
                        'Predicts context words from target word',
                        'Counts word frequencies',
                        'Removes stop words'
                    ],
                    'correct': 1,
                    'explanation': 'Skip-gram predicts the context words given a target word, which is the opposite of CBOW and works well with smaller datasets.'
                },
                {
                    'question': 'What is a key characteristic of word embeddings that enables arithmetic operations?',
                    'options': [
                        'They are always positive numbers',
                        'They capture semantic relationships that allow operations like "king - man + woman ≈ queen"',
                        'They are always integers',
                        'They are sorted alphabetically'
                    ],
                    'correct': 1,
                    'explanation': 'Word embeddings capture semantic relationships that enable arithmetic operations, such as "king - man + woman ≈ queen", demonstrating their ability to encode meaning.'
                },
                {
                    'question': 'Which of the following is a limitation of traditional word embeddings like Word2Vec?',
                    'options': [
                        'They are too fast',
                        'They create a single static representation for each word regardless of context',
                        'They require too little data',
                        'They are too simple to implement'
                    ],
                    'correct': 1,
                    'explanation': 'Traditional word embeddings create a single static representation for each word, which doesn\'t account for different meanings of polysemous words in different contexts.'
                },
                {
                    'question': 'What is the main advantage of contextualized word embeddings over traditional ones?',
                    'options': [
                        'They are faster to compute',
                        'They generate different representations for the same word in different contexts',
                        'They require less training data',
                        'They are easier to visualize'
                    ],
                    'correct': 1,
                    'explanation': 'Contextualized word embeddings generate different representations for the same word depending on its context, addressing the limitation of static embeddings.'
                },
                {
                    'question': 'Which model introduced contextualized word embeddings?',
                    'options': [
                        'Word2Vec',
                        'GloVe',
                        'ELMo',
                        'FastText'
                    ],
                    'correct': 2,
                    'explanation': 'ELMo (Embeddings from Language Models) was one of the first models to introduce contextualized word embeddings that generate different representations based on context.'
                },
                {
                    'question': 'What does GloVe stand for?',
                    'options': [
                        'Global Vectors for Word Representation',
                        'Graphical Learning of Vector Embeddings',
                        'General Language Understanding Vectors',
                        'Gradient-based Learning of Vector Encodings'
                    ],
                    'correct': 0,
                    'explanation': 'GloVe stands for Global Vectors for Word Representation, a word embedding model that combines global matrix factorization with local context window methods.'
                },
                {
                    'question': 'What is a key difference between Word2Vec and GloVe?',
                    'options': [
                        'Word2Vec uses neural networks, GloVe uses matrix factorization',
                        'Word2Vec is faster, GloVe is slower',
                        'Word2Vec works only with English, GloVe works with all languages',
                        'Word2Vec requires more data than GloVe'
                    ],
                    'correct': 0,
                    'explanation': 'Word2Vec uses neural networks to learn embeddings, while GloVe uses matrix factorization techniques to learn from global word-word co-occurrence statistics.'
                },
                {
                    'question': 'What is the purpose of the embedding layer in neural networks?',
                    'options': [
                        'To store the entire dataset',
                        'To convert discrete tokens into continuous vector representations',
                        'To remove punctuation from text',
                        'To perform sentiment analysis'
                    ],
                    'correct': 1,
                    'explanation': 'The embedding layer in neural networks converts discrete tokens (like words) into continuous vector representations that can be processed by the network.'
                },
                {
                    'question': 'Which of the following is a benefit of using pre-trained word embeddings?',
                    'options': [
                        'They are always more accurate than training from scratch',
                        'They save training time and can provide good representations even with limited data',
                        'They are easier to implement than custom embeddings',
                        'They work with any type of data'
                    ],
                    'correct': 1,
                    'explanation': 'Pre-trained word embeddings save training time and can provide good representations even with limited data, as they are trained on large corpora.'
                },
                {
                    'question': 'What is FastText in relation to word embeddings?',
                    'options': [
                        'A method to speed up neural networks',
                        'An extension of Word2Vec that considers subword information',
                        'A technique to remove stop words quickly',
                        'A method for real-time text processing'
                    ],
                    'correct': 1,
                    'explanation': 'FastText is an extension of Word2Vec that considers subword information (character n-grams), allowing it to generate embeddings for out-of-vocabulary words.'
                },
                {
                    'question': 'What is the main advantage of FastText over traditional Word2Vec?',
                    'options': [
                        'It is faster to train',
                        'It can generate embeddings for unseen words using subword information',
                        'It requires less memory',
                        'It is simpler to implement'
                    ],
                    'correct': 1,
                    'explanation': 'FastText can generate embeddings for unseen words by using subword information (character n-grams), which is particularly useful for handling out-of-vocabulary words.'
                },
                {
                    'question': 'Which of the following operations is meaningful with word embeddings?',
                    'options': [
                        'Adding embeddings of unrelated words',
                        'Calculating cosine similarity between embeddings',
                        'Sorting embeddings alphabetically',
                        'Multiplying embeddings randomly'
                    ],
                    'correct': 1,
                    'explanation': 'Calculating cosine similarity between embeddings is meaningful as it measures the semantic similarity between words based on their vector representations.'
                },
                {
                    'question': 'What does it mean for word embeddings to be dense vectors?',
                    'options': [
                        'They contain mostly zero values',
                        'They contain mostly non-zero values',
                        'They are very long vectors',
                        'They are stored in databases'
                    ],
                    'correct': 1,
                    'explanation': 'Dense vectors in word embeddings contain mostly non-zero values, unlike sparse representations like one-hot encoding which have mostly zeros.'
                },
                {
                    'question': 'Which of the following is a common application of word embeddings?',
                    'options': [
                        'Image compression',
                        'Information retrieval and document similarity',
                        'Audio processing',
                        'Video streaming'
                    ],
                    'correct': 1,
                    'explanation': 'Word embeddings are commonly used in information retrieval and document similarity tasks, where they help measure the semantic similarity between texts.'
                },
                {
        'question': 'Which distance metric is most commonly used to compare word embeddings?',
        'options': [
            'Euclidean distance',
            'Manhattan distance',
            'Cosine similarity',
            'Jaccard similarity'
        ],
        'correct': 2,
        'explanation': 'Cosine similarity is commonly used because it measures the angle between two vectors, effectively capturing similarity in direction regardless of magnitude.'
    },
    {
        'question': 'Why do word embeddings enable transfer learning in NLP tasks?',
        'options': [
            'They are task-specific',
            'They are trained on numerical data',
            'They encode general semantic properties learned from large corpora',
            'They only work for classification tasks'
        ],
        'correct': 2,
        'explanation': 'Pre-trained embeddings encode general semantic properties, enabling transfer learning to new NLP tasks with limited labeled data.'
    },
    {
        'question': 'How does FastText handle out-of-vocabulary (OOV) words?',
        'options': [
            'It ignores them during inference',
            'It generates embeddings by averaging character n-gram vectors',
            'It replaces them with a special token',
            'It deletes them from the text'
        ],
        'correct': 1,
        'explanation': 'FastText represents words as bags of character n-grams, allowing it to compute embeddings for OOV words by averaging the embeddings of their constituent n-grams.'
    },
    {
        'question': 'What happens to the representation of polysemous words in traditional word embeddings?',
        'options': [
            'Each meaning gets its own unique vector',
            'All meanings share a single static vector',
            'Meanings are merged randomly',
            'Meanings are separated by language'
        ],
        'correct': 1,
        'explanation': 'Traditional word embeddings assign a single static vector to each word, regardless of its context or multiple meanings.'
    },
    {
        'question': 'Which of the following relationships can be captured by word embeddings?',
        'options': [
            'Syntactic relationships only',
            'Semantic relationships only',
            'Both syntactic and semantic relationships',
            'No relationships at all'
        ],
        'correct': 2,
        'explanation': 'Word embeddings can capture both syntactic (e.g., plural forms) and semantic (e.g., analogies) relationships between words.'
    },
    {
        'question': 'What is subword information in the context of word embeddings?',
        'options': [
            'Storing phrases instead of single words',
            'Using character n-grams to represent words',
            'Encoding only stopwords',
            'Utilizing sentence-level features'
        ],
        'correct': 1,
        'explanation': 'Subword information refers to representing words with character n-grams, allowing better handling of rare and out-of-vocabulary words.'
    },
    {
        'question': 'Why is dimensionality reduction sometimes applied to word embeddings?',
        'options': [
            'To make them less interpretable',
            'To visualize embeddings and reduce computational complexity',
            'To increase their storage space',
            'To remove semantic information'
        ],
        'correct': 1,
        'explanation': 'Dimensionality reduction techniques like PCA or t-SNE are used to visualize embeddings and make computations more efficient by reducing vector size.'
    },
    {
        'question': 'Which technique is commonly used for visualizing high-dimensional word embeddings?',
        'options': [
            'Fourier Transform',
            'Principal Component Analysis (PCA)',
            'Histogram equalization',
            'Gradient descent'
        ],
        'correct': 1,
        'explanation': 'Principal Component Analysis (PCA) is often used to project high-dimensional embeddings into 2D or 3D for visualization.'
    },
    {
        'question': 'What is the effect of increasing the dimensionality of word embeddings?',
        'options': [
            'It always improves performance',
            'It may capture more complex relationships, but can lead to overfitting or increased computational cost',
            'It makes embeddings sparse',
            'It reduces the size of the embedding file'
        ],
        'correct': 1,
        'explanation': 'Higher dimensional embeddings can capture richer relationships but may also result in overfitting or higher computational demand.'
    },
    {
        'question': 'How are pre-trained word embeddings typically integrated into a deep learning model?',
        'options': [
            'As a fixed layer that is never updated',
            'By training them from scratch during every run',
            'By loading them into the model\'s embedding layer, optionally allowing fine-tuning',
            'By concatenating them with audio features'
        ],
        'correct': 2,
        'explanation': 'Pre-trained embeddings are often loaded into the embedding layer of a model; they can be kept fixed or fine-tuned during training.'
    }
            ],
            'chunking': [
                {
                    'question': 'What is the primary purpose of chunking in NLP?',
                    'options': [
                        'To remove punctuation from text',
                        'To group words into meaningful phrases or constituents',
                        'To convert text to lowercase',
                        'To count the number of words'
                    ],
                    'correct': 1,
                    'explanation': 'Chunking groups words into meaningful phrases or constituents, such as noun phrases and verb phrases, to identify syntactic structures.'
                },
                {
                    'question': 'Which of the following is a common type of chunk in NLP?',
                    'options': [
                        'Paragraph chunk',
                        'Noun phrase chunk',
                        'Sentence chunk',
                        'Document chunk'
                    ],
                    'correct': 1,
                    'explanation': 'Noun phrase chunk is a common type of chunk in NLP that groups words around a noun and its modifiers.'
                },
                {
                    'question': 'What is the difference between chunking and parsing?',
                    'options': [
                        'Chunking is faster than parsing',
                        'Chunking identifies shallow structures, parsing identifies deep grammatical structures',
                        'Chunking works only with English, parsing works with all languages',
                        'There is no difference'
                    ],
                    'correct': 1,
                    'explanation': 'Chunking identifies shallow syntactic structures (like noun phrases), while parsing identifies deeper grammatical structures with full syntactic trees.'
                },
                {
                    'question': 'Which tag is commonly used to indicate the beginning of a chunk?',
                    'options': ['B', 'I', 'O', 'E'],
                    'correct': 0,
                    'explanation': 'The B tag (Beginning) is commonly used to indicate the beginning of a chunk in chunking tasks.'
                },
                {
                    'question': 'Which tag is commonly used to indicate words inside a chunk?',
                    'options': ['B', 'I', 'O', 'E'],
                    'correct': 1,
                    'explanation': 'The I tag (Inside) is commonly used to indicate words that are inside a chunk but not at the beginning.'
                },
                {
                    'question': 'What does the IOB tagging scheme stand for?',
                    'options': [
                        'Inside, Outside, Beginning',
                        'Input, Output, Buffer',
                        'Initial, Ongoing, Boundary',
                        'Internal, Outer, Base'
                    ],
                    'correct': 0,
                    'explanation': 'IOB tagging stands for Inside, Outside, Beginning, a common scheme for marking chunk boundaries.'
                },
                {
                    'question': 'What is the main advantage of using chunking over full parsing?',
                    'options': [
                        'It is more accurate',
                        'It is computationally less expensive and faster',
                        'It works with more languages',
                        'It requires more training data'
                    ],
                    'correct': 1,
                    'explanation': 'Chunking is computationally less expensive and faster than full parsing while still providing useful syntactic information.'
                },
                {
                    'question': 'Which of the following is a typical application of chunking?',
                    'options': [
                        'Image recognition',
                        'Information extraction',
                        'Audio processing',
                        'Database management'
                    ],
                    'correct': 1,
                    'explanation': 'Information extraction is a typical application of chunking, where identifying noun phrases and other chunks helps extract relevant information from text.'
                },
                {
                    'question': 'What is a noun phrase chunk typically centered around?',
                    'options': [
                        'A verb',
                        'A noun',
                        'An adjective',
                        'An adverb'
                    ],
                    'correct': 1,
                    'explanation': 'A noun phrase chunk is typically centered around a noun, which serves as the head of the phrase.'
                },
                {
                    'question': 'Which POS tags are commonly associated with the beginning of noun phrase chunks?',
                    'options': [
                        'VB, VBD, VBG',
                        'DT, JJ, NN',
                        'RB, RBR, RBS',
                        'IN, CC, UH'
                    ],
                    'correct': 1,
                    'explanation': 'Determiners (DT), adjectives (JJ), and nouns (NN) are commonly associated with the beginning of noun phrase chunks.'
                },
                {
                    'question': 'What is the CoNLL-2000 shared task related to?',
                    'options': [
                        'Text summarization',
                        'Named entity recognition',
                        'Chunking',
                        'Sentiment analysis'
                    ],
                    'correct': 2,
                    'explanation': 'The CoNLL-2000 shared task was focused on chunking, providing a standard benchmark for evaluating chunking systems.'
                },
                {
                    'question': 'Which machine learning approach is commonly used for chunking?',
                    'options': [
                        'Linear regression',
                        'Sequence labeling models like CRF or LSTM',
                        'K-means clustering',
                        'Decision trees for regression'
                    ],
                    'correct': 1,
                    'explanation': 'Sequence labeling models like Conditional Random Fields (CRF) or LSTM networks are commonly used for chunking as they can model dependencies between adjacent labels.'
                },
                {
                    'question': 'What is the main challenge in chunking?',
                    'options': [
                        'It requires too much memory',
                        'Determining the correct boundaries of chunks',
                        'It works only with English text',
                        'It is too slow for real-time applications'
                    ],
                    'correct': 1,
                    'explanation': 'The main challenge in chunking is determining the correct boundaries of chunks, especially when dealing with complex syntactic structures.'
                },
                {
                    'question': 'What is the purpose of using a grammar in rule-based chunking?',
                    'options': [
                        'To make the system slower',
                        'To define patterns for identifying chunks',
                        'To translate text between languages',
                        'To remove punctuation from text'
                    ],
                    'correct': 1,
                    'explanation': 'In rule-based chunking, a grammar defines patterns for identifying chunks, such as noun phrases or verb phrases, based on POS tags.'
                },
                {
                    'question': 'Which of the following is an example of a chunking pattern for noun phrases?',
                    'options': [
                        'NP: {<DT>?<JJ>*<NN>}',
                        'VP: {<VB><NP>}',
                        'PP: {<IN><NP>}',
                        'All of the above'
                    ],
                    'correct': 0,
                    'explanation': 'NP: {<DT>?<JJ>*<NN>} is a common chunking pattern for noun phrases, where DT is an optional determiner, JJ is zero or more adjectives, and NN is a noun.'
                },
                {
                    'question': 'What is the role of the "O" tag in chunking?',
                    'options': [
                        'It indicates the outside of a chunk',
                        'It indicates the beginning of a chunk',
                        'It indicates words inside a chunk',
                        'It indicates the end of a chunk'
                    ],
                    'correct': 0,
                    'explanation': 'The "O" tag in chunking indicates words that are outside of any chunk, meaning they don\'t belong to any specific phrase or constituent.'
                },
                {
                    'question': 'Which of the following is a benefit of chunking in information extraction?',
                    'options': [
                        'It increases the size of the dataset',
                        'It helps identify relevant phrases for extraction',
                        'It removes all punctuation',
                        'It converts text to uppercase'
                    ],
                    'correct': 1,
                    'explanation': 'Chunking helps identify relevant phrases for extraction, such as noun phrases that might contain important entities or concepts.'
                },
                {
                    'question': 'What is the difference between chunking and named entity recognition?',
                    'options': [
                        'Chunking identifies general phrases, NER identifies specific entities',
                        'Chunking is faster than NER',
                        'Chunking works only with English, NER works with all languages',
                        'There is no difference'
                    ],
                    'correct': 0,
                    'explanation': 'Chunking identifies general syntactic phrases like noun phrases, while Named Entity Recognition specifically identifies and classifies named entities like persons, organizations, and locations.'
                },
                {
                    'question': 'Which of the following is a common evaluation metric for chunking?',
                    'options': [
                        'Accuracy only',
                        'Precision, recall, and F1-score',
                        'Processing speed only',
                        'Vocabulary size'
                    ],
                    'correct': 1,
                    'explanation': 'Chunking systems are typically evaluated using precision, recall, and F1-score, which measure the accuracy of chunk identification and boundary detection.'
                },
                {
                    'question': 'What is the main advantage of using machine learning for chunking over rule-based approaches?',
                    'options': [
                        'It is always faster',
                        'It can adapt to different domains and languages more easily',
                        'It requires less training data',
                        'It is simpler to implement'
                    ],
                    'correct': 1,
                    'explanation': 'Machine learning approaches for chunking can adapt to different domains and languages more easily than rule-based approaches, which need to be manually crafted for each domain.'
                },
                {
        'question': 'Which of the following best describes parsing in NLP?',
        'options': [
            'Identifying named entities in text',
            'Assigning part-of-speech tags to each word',
            'Analyzing the complete syntactic structure of a sentence',
            'Removing stopwords from a document'
        ],
        'correct': 2,
        'explanation': 'Parsing analyzes the complete syntactic structure of a sentence, producing a parse tree that shows the relationships between words and phrases.'
    },
    {
        'question': 'Which type of parsing builds a tree structure starting from the root node and working down to the leaves?',
        'options': [
            'Bottom-up parsing',
            'Top-down parsing',
            'Random parsing',
            'Chunk-based parsing'
        ],
        'correct': 1,
        'explanation': 'Top-down parsing constructs a tree from the root node (sentence) down to the leaf nodes (words).'
    },
    {
        'question': 'What is a dependency parse tree?',
        'options': [
            'A tree that shows the hierarchical structure of a sentence',
            'A tree that shows grammatical dependencies between individual words',
            'A tree that contains only noun phrases',
            'A tree that separates sentences into paragraphs'
        ],
        'correct': 1,
        'explanation': 'Dependency parse trees show the grammatical dependencies between words, highlighting which words modify or are dependent on others.'
    },
    {
        'question': 'In the IOB tagging scheme, what does the "O" label indicate?',
        'options': [
            'The start of a new chunk',
            'A word outside any chunk',
            'The last word in a chunk',
            'An error in the tagging'
        ],
        'correct': 1,
        'explanation': 'The "O" label indicates that a word is outside any chunk and does not belong to any constituent phrase.'
    },
    {
        'question': 'Which of the following is a common evaluation dataset for constituency parsing?',
        'options': [
            'ImageNet',
            'Penn Treebank',
            'CoNLL-2017',
            'LibriSpeech'
        ],
        'correct': 1,
        'explanation': 'The Penn Treebank is a widely used evaluation dataset for constituency parsing tasks.'
    },
    {
        'question': 'What is a chunk grammar in NLP?',
        'options': [
            'A set of rules for splitting text into sentences',
            'A set of regular expressions for identifying phrase patterns in POS-tagged text',
            'A system for translating text',
            'A method for converting text to speech'
        ],
        'correct': 1,
        'explanation': 'Chunk grammar consists of pattern-based rules, often with regular expressions, to define how phrases can be identified from POS sequences.'
    },
    {
        'question': 'Which symbol is often used to denote a noun phrase in chunking grammar?',
        'options': [
            'NP',
            'VP',
            'PP',
            'ADJP'
        ],
        'correct': 0,
        'explanation': 'NP stands for noun phrase and is commonly used in chunking grammars to denote noun phrase patterns.'
    },
    {
        'question': 'What is the role of part-of-speech (POS) tagging in chunking?',
        'options': [
            'To identify punctuation',
            'To provide word-level grammatical information for chunking patterns',
            'To convert text to lower case',
            'To count the number of words'
        ],
        'correct': 1,
        'explanation': 'POS tags supply the grammatical information needed to apply chunking patterns and identify phrase boundaries.'
    },
    {
        'question': 'Which of the following is a widely used tool for chunking and parsing in Python?',
        'options': [
            'NumPy',
            'NLTK',
            'Pandas',
            'Seaborn'
        ],
        'correct': 1,
        'explanation': 'NLTK (Natural Language Toolkit) provides modules for both chunking and parsing in Python.'
    },
    {
        'question': 'What do parse trees reveal about a sentence?',
        'options': [
            'Word frequencies',
            'Phrase structure and syntactic relationships',
            'Sentiment scores',
            'Named entities'
        ],
        'correct': 1,
        'explanation': 'Parse trees display the hierarchical phrase structure of a sentence and reveal grammatical and syntactic relationships.'
    },
    {
        'question': 'Which parsing technique is efficient for finding the best parse tree using probabilistic grammars?',
        'options': [
            'Depth-first search',
            'CYK algorithm',
            'Earley parser',
            'Dijkstra’s algorithm'
        ],
        'correct': 2,
        'explanation': 'The Earley parser is efficient for probabilistic context-free grammars and can handle ambiguity in parsing.'
    },
    {
        'question': 'Chunking outputs are sometimes called what?',
        'options': [
            'Parse forests',
            'Shallow parses',
            'Semantic graphs',
            'Deep trees'
        ],
        'correct': 1,
        'explanation': 'Chunking is often referred to as shallow parsing because it identifies only basic phrase structures, not full syntactic trees.'
    },
    {
        'question': 'Which task typically precedes chunking in an NLP pipeline?',
        'options': [
            'Tokenization and POS tagging',
            'Coreference resolution',
            'Machine translation',
            'Word sense disambiguation'
        ],
        'correct': 0,
        'explanation': 'Tokenization and POS tagging are usually completed before chunking since chunks rely on POS-tagged text.'
    },
    {
        'question': 'What is one limitation of rule-based chunking systems?',
        'options': [
            'They can’t be used on English text',
            'They are difficult to adapt to new domains or languages',
            'They are always more accurate than machine learning approaches',
            'They require no human expertise'
        ],
        'correct': 1,
        'explanation': 'Rule-based chunkers require linguistically tailored rules, making them hard to adapt for new domains or languages.'
    },
    {
        'question': 'What is a transition-based parser?',
        'options': [
            'A parser that uses transition rules to incrementally build a parse structure',
            'A parser that only uses chunking patterns',
            'A parser that skips transitions',
            'A parser that translates text'
        ],
        'correct': 0,
        'explanation': 'A transition-based parser incrementally predicts transitions (actions) to assemble a parse structure for a sentence.'
    }
            ]
        }
        
        return questions.get(module_name, [])
    
    def calculate_quiz_score(self, module_name, answers):
        """Calculate quiz score based on answers"""
        questions = self.get_quiz_questions(module_name)
        if not questions:
            return 0
        
        correct = 0
        for i, answer in enumerate(answers):
            if i < len(questions) and answer == questions[i]['correct']:
                correct += 1
        
        return int((correct / len(questions)) * 100)
