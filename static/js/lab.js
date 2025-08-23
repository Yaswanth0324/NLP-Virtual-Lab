// Lab-specific JavaScript functionality
class LabManager {
    constructor(moduleName) {
        this.moduleName = moduleName;
        this.currentResults = null;
        this.initializeElements();
        this.setupEventListeners();
        this.setupModule();
    }
    
    initializeElements() {
        this.inputText = document.getElementById('inputText');
        this.operation = document.getElementById('operation');
        this.processBtn = document.getElementById('processBtn');
        this.clearBtn = document.getElementById('clearBtn');
        this.results = document.getElementById('results');
        this.visualization = document.getElementById('visualization');
        this.processingSpinner = document.getElementById('processingSpinner');
        this.explanationContent = document.getElementById('explanationContent');
    }
    
    setupEventListeners() {
        this.processBtn.addEventListener('click', () => this.processText());
        this.clearBtn.addEventListener('click', () => this.clearResults());
        
        // Example text buttons
        document.querySelectorAll('.example-btn').forEach(btn => {
            btn.addEventListener('click', (e) => {
                this.inputText.value = e.target.dataset.text;
                this.inputText.focus();
            });
        });
        
        // Auto-process on operation change
        this.operation.addEventListener('change', () => {
            if (this.inputText.value.trim()) {
                this.processText();
            }
        });
    }
    
    setupModule() {
        this.setupOperationOptions();
        this.setupExplanation();
        this.setupTranslationControlsIfNeeded();
        this.setupSummarizationControlsIfNeeded();
        this.setupTextGenerationControlsIfNeeded();
    }
    
    setupOperationOptions() {
        const operations = this.getOperationsForModule();
        this.operation.innerHTML = '';
        
        operations.forEach(op => {
            const option = document.createElement('option');
            option.value = op.value;
            option.textContent = op.label;
            this.operation.appendChild(option);
        });
    }
    
    getOperationsForModule() {
        const operationMap = {
            'text_preprocessing': [
                { value: 'preprocess', label: 'Complete Preprocessing Pipeline' },
                { value: 'tokenize', label: 'Tokenization' },
                { value: 'stem', label: 'Stemming' },
                { value: 'lemmatize', label: 'Lemmatization' }
            ],
            'pos_tagging': [
                { value: 'pos_tag', label: 'Part-of-Speech Tagging' }
            ],
            'ngram_modeling': [
                { value: 'ngrams', label: 'N-gram Generation' }
            ],
            'named_entity_recognition': [
                { value: 'ner', label: 'Named Entity Recognition' }
            ],
            'sentiment_analysis': [
                { value: 'sentiment', label: 'Sentiment Analysis' }
            ],
            'chunking': [
                { value: 'chunk', label: 'Noun Phrase Chunking' }
            ],
            'text_classification': [
                { value: 'sentiment', label: 'Sentiment Classification' }
            ],
            'word_embeddings': [
                { value: 'tokenize', label: 'Text Tokenization' }
            ],
            'machine_translation': [
                { value: 'translate', label: 'Translate' }
            ],
            'text_summarization': [
                { value: 'summarize', label: 'Summarize Text' }
            ],
            'text_generation': [
                { value: 'generate', label: 'Generate Text' }
            ]
        };
        
        return operationMap[this.moduleName] || [
            { value: 'tokenize', label: 'Tokenization' }
        ];
    }
    
    setupExplanation() {
        const explanations = {
            'text_preprocessing': `
                <h6>Text Preprocessing</h6>
                <p>Text preprocessing is the crucial first step in NLP that cleans and prepares raw text for analysis. It involves several key steps:</p>
                <ul>
                    <li><strong>Tokenization:</strong> Breaking text into individual words or tokens</li>
                    <li><strong>Lowercasing:</strong> Converting all text to lowercase for consistency</li>
                    <li><strong>Punctuation Removal:</strong> Removing punctuation marks that don't add meaning</li>
                    <li><strong>Stopword Removal:</strong> Eliminating common words like "the", "is", "and"</li>
                    <li><strong>Stemming:</strong> Reducing words to their root form (e.g., "running" → "run")</li>
                    <li><strong>Lemmatization:</strong> Finding the dictionary form of words using grammar rules</li>
                </ul>
                <p>Each step serves a specific purpose in preparing text for downstream NLP tasks.</p>
            `,
            'pos_tagging': `
                <h6>Part-of-Speech Tagging</h6>
                <p>POS tagging assigns grammatical categories to each word in a sentence. Common tags include:</p>
                <ul>
                    <li><strong>NN:</strong> Noun, singular (e.g., "cat", "house")</li>
                    <li><strong>NNS:</strong> Noun, plural (e.g., "cats", "houses")</li>
                    <li><strong>VB:</strong> Verb, base form (e.g., "run", "eat")</li>
                    <li><strong>VBD:</strong> Verb, past tense (e.g., "ran", "ate")</li>
                    <li><strong>JJ:</strong> Adjective (e.g., "big", "red")</li>
                    <li><strong>RB:</strong> Adverb (e.g., "quickly", "very")</li>
                    <li><strong>DT:</strong> Determiner (e.g., "the", "a")</li>
                </ul>
                <p>POS tagging is essential for understanding sentence structure and meaning.</p>
            `,
            'ngram_modeling': `
                <h6>N-gram Modeling</h6>
                <p>N-grams are contiguous sequences of n items from a text. They help capture context and word relationships:</p>
                <ul>
                    <li><strong>Unigrams (1-gram):</strong> Individual words</li>
                    <li><strong>Bigrams (2-gram):</strong> Two consecutive words</li>
                    <li><strong>Trigrams (3-gram):</strong> Three consecutive words</li>
                </ul>
                <p>N-grams are useful for:</p>
                <ul>
                    <li>Language modeling and text prediction</li>
                    <li>Autocomplete systems</li>
                    <li>Spell checking</li>
                    <li>Text generation</li>
                </ul>
            `,
            'named_entity_recognition': `
                <h6>Named Entity Recognition (NER)</h6>
                <p>NER identifies and classifies named entities in text into predefined categories:</p>
                <ul>
                    <li><strong>PERSON:</strong> People's names (e.g., "John Smith")</li>
                    <li><strong>ORGANIZATION:</strong> Companies, agencies (e.g., "Apple Inc.")</li>
                    <li><strong>GPE:</strong> Geopolitical entities (e.g., "New York", "USA")</li>
                    <li><strong>DATE:</strong> Dates and times (e.g., "January 1, 2023")</li>
                    <li><strong>MONEY:</strong> Monetary values (e.g., "$100")</li>
                    <li><strong>PERCENT:</strong> Percentages (e.g., "50%")</li>
                </ul>
                <p>NER is crucial for information extraction, question answering, and knowledge graphs.</p>
            `,
            'sentiment_analysis': `
                <h6>Sentiment Analysis</h6>
                <p>Sentiment analysis determines the emotional tone of text. It typically classifies text as:</p>
                <ul>
                    <li><strong>Positive:</strong> Expresses positive emotions or opinions</li>
                    <li><strong>Negative:</strong> Expresses negative emotions or opinions</li>
                    <li><strong>Neutral:</strong> Objective or balanced tone</li>
                </ul>
                <p>VADER (Valence Aware Dictionary and sEntiment Reasoner) provides scores for:</p>
                <ul>
                    <li><strong>Positive:</strong> Degree of positive sentiment</li>
                    <li><strong>Negative:</strong> Degree of negative sentiment</li>
                    <li><strong>Neutral:</strong> Degree of neutral sentiment</li>
                    <li><strong>Compound:</strong> Overall sentiment score (-1 to 1)</li>
                </ul>
            `,
            'chunking': `
                <h6>Chunking and Parsing</h6>
                <p>Chunking groups words into meaningful phrases, typically noun phrases (NP) or verb phrases (VP):</p>
                <ul>
                    <li><strong>Noun Phrases:</strong> Groups of words functioning as a noun</li>
                    <li><strong>Verb Phrases:</strong> Groups of words functioning as a verb</li>
                </ul>
                <p>Chunking is useful for:</p>
                <ul>
                    <li>Information extraction</li>
                    <li>Question answering</li>
                    <li>Semantic analysis</li>
                    <li>Understanding sentence structure</li>
                </ul>
            `,
            'text_classification': `
                <h6>Text Classification</h6>
                <p>Text classification automatically categorizes text into predefined groups. It's one of the most fundamental NLP tasks with many applications:</p>
                <ul>
                    <li><strong>Spam Detection:</strong> Classifying emails as "Spam" or "Not Spam".</li>
                    <li><strong>Topic Modeling:</strong> Assigning topics like "Sports", "Politics", or "Technology" to news articles.</li>
                    <li><strong>Sentiment Analysis:</strong> A type of classification that determines if a text is "Positive", "Negative", or "Neutral".</li>
                </ul>
                <p>Models are trained on labeled data to learn the patterns associated with each category.</p>
            `,
            'word_embeddings': `
                <h6>Word Embeddings</h6>
                <p>Word embeddings are numerical vector representations of words that capture their semantic meaning and relationships. Instead of treating words as isolated strings, embeddings place them in a multi-dimensional space where similar words are closer together.</p>
                <ul>
                    <li><strong>Key Idea:</strong> Words that appear in similar contexts have similar meanings.</li>
                    <li><strong>Famous Example:</strong> Vector("King") - Vector("Man") + Vector("Woman") results in a vector very close to Vector("Queen").</li>
                    <li><strong>Popular Models:</strong> Word2Vec, GloVe, and FastText are common algorithms for creating word embeddings.</li>
                </ul>
                <p>Embeddings are crucial for deep learning models as they allow neural networks to work with the meaning of words, not just their surface form.</p>
            `,
            'machine_translation': `
                <h6>Machine Translation (MT)</h6>
                <p>Machine Translation is the technology that automatically converts text from a <strong>source language</strong> to a <strong>target language</strong>. Think of it like a smart assistant that understands the full <em>meaning and context</em> of a sentence, not just the individual words, and then expresses that same idea in a new language.</p>
                <ul>
                    <li><strong>Rule-Based (RBMT):</strong> The earliest approach, using manually written grammar rules and dictionaries. It often produced robotic-sounding translations because it lacked contextual understanding.</li>
                    <li><strong>Statistical (SMT):</strong> This method learned from massive amounts of human-translated text by finding the most statistically probable phrase-for-phrase translations. It was more fluent but still struggled with complex grammar.</li>
                    <li><strong>Neural (NMT):</strong> The modern, brain-inspired approach. An 'encoder' network reads the entire source sentence to capture its meaning, and a 'decoder' network generates a new, highly accurate and natural-sounding sentence in the target language. This is the technology behind services like Google Translate today.</li>
                </ul>
                <p>While challenges like idioms and cultural nuances still exist, NMT is a cornerstone of modern AI, breaking down language barriers and making global communication and information access possible. 🌍</p>
`,
            'text_summarization': `
                <h6>Text Summarization</h6>
                <p>Text summarization condenses long passages into concise summaries while preserving the key ideas. Modern approaches use transformer models (e.g., BART, T5) trained to generate high-quality abstractive summaries.</p>
                <ul>
                    <li><strong>Abstractive:</strong> Generates new sentences to paraphrase the source.</li>
                    <li><strong>Extractive:</strong> Selects important sentences from the source text.</li>
                    <li><strong>Length Controls:</strong> Use minimum and maximum length to guide output size.</li>
                </ul>
                <p>Use this lab to explore how summary length affects the information retained.</p>
            `,
            'text_generation': `
                <h6>Text Generation</h6>
                <p>Text generation creates new text by predicting one token at a time given a <em>prompt</em>. The model looks at the words you provide and continues with the most likely next words, sampling from a probability distribution.</p>
                <ul>
                    <li><strong>How it works:</strong> The model scores many possible next tokens and samples one based on settings like <em>top-k</em> (keep the k best options) and <em>top-p</em> (keep the smallest set of tokens whose probabilities add up to p).</li>
                    <li><strong>Controlling output:</strong> You can influence length (maximum new tokens), randomness (temperature/top-k/top-p), and how many alternatives to return (number of sequences).</li>
                    <li><strong>Prompting tips:</strong> State the task clearly, include brief constraints (style, length, format), and provide a small example if needed. Clear prompts = better generations.</li>
                    <li><strong>Limitations:</strong> Models can produce repetitive, biased, or incorrect text (“hallucinations”). Always review outputs, especially for factual tasks.</li>
                </ul>
                <p>Try different prompts and parameters to see how the style, creativity, and coherence change.</p>
            `
        };
        
        this.explanationContent.innerHTML = explanations[this.moduleName] || 
            '<p>Learn about this NLP concept through interactive examples and practice.</p>';
    }
    
    async processText() {
        const text = this.inputText.value.trim();
        const operation = this.operation.value;
        
        if (!text) {
            showAlert('Please enter some text to process.', 'warning');
            return;
        }
        
        this.showSpinner();
        this.processBtn.disabled = true;
        
        try {
            let payload = { text, operation };
            let endpoint = '/api/process';

            if (operation === 'translate') {
                const srcInput = document.getElementById('srcLang');
                const destInput = document.getElementById('destLang');
                if (srcInput && destInput) {
                    payload.src_lang = (srcInput.value || 'auto').trim() || 'auto';
                    payload.dest_lang = (destInput.value || 'en').trim() || 'en';
                }
            } else if (operation === 'summarize' || this.moduleName === 'text_summarization') {
                const minLenInput = document.getElementById('minLen');
                const maxLenInput = document.getElementById('maxLen');
                const min_length = minLenInput ? parseInt(minLenInput.value, 10) || 30 : 30;
                const max_length = maxLenInput ? parseInt(maxLenInput.value, 10) || 130 : 130;
                endpoint = '/summarize';
                payload = { text, min_length, max_length };
            } else if (operation === 'generate' || this.moduleName === 'text_generation') {
                const genMaxLenInput = document.getElementById('genMaxLen');
                const genNumSeqInput = document.getElementById('genNumSequences');
                const max_length = genMaxLenInput ? parseInt(genMaxLenInput.value, 10) || 50 : 50;
                const num_return_sequences = genNumSeqInput ? parseInt(genNumSeqInput.value, 10) || 1 : 1;
                endpoint = '/generate';
                payload = { prompt: text, max_length, num_return_sequences };
            }

            const response = await makeAPIRequest(endpoint, {
                method: 'POST',
                body: JSON.stringify(payload)
            });
            
            this.currentResults = response;
            this.displayResults(response);
            this.displayVisualization(response);
            
        } catch (error) {
            console.error('Processing error:', error);
            const msg = (error && error.message) ? `Failed to process text. ${error.message}` : 'Failed to process text. Please try again.';
            this.displayError(msg);
        } finally {
            this.hideSpinner();
            this.processBtn.disabled = false;
        }
    }
    
    displayResults(results) {
        if (results.error) {
            this.displayError(results.error);
            return;
        }
        
        const operation = this.operation.value;
        let html = '';
        
        switch (operation) {
            case 'preprocess':
                html = this.formatPreprocessingResults(results);
                break;
            case 'tokenize':
                html = this.formatTokenizationResults(results);
                break;
            case 'pos_tag':
                html = this.formatPOSResults(results);
                break;
            case 'ngrams':
                html = this.formatNgramResults(results);
                break;
            case 'ner':
                html = this.formatNERResults(results);
                break;
            case 'sentiment':
                html = this.formatSentimentResults(results);
                break;
            case 'stem':
                html = this.formatStemmingResults(results);
                break;
            case 'lemmatize':
                html = this.formatLemmatizationResults(results);
                break;
            case 'chunk':
                html = this.formatChunkingResults(results);
                break;
            case 'translate':
                html = this.formatTranslationResults(results);
                break;
            case 'summarize':
                html = this.formatSummarizationResults(results);
                break;
            case 'generate':
                html = this.formatTextGenerationResults(results);
                break;
            default:
                html = '<div class="alert alert-info">Results will appear here.</div>';
        }
        
        this.results.innerHTML = html;
        this.results.classList.add('fade-in');
    }
    
    formatPreprocessingResults(results) {
        if (!results.steps) return '<div class="alert alert-danger">No preprocessing steps found.</div>';
        
        let html = '<h6>Preprocessing Steps</h6>';
        html += '<div class="preprocessing-steps">';
        
        results.steps.forEach(step => {
            html += `
                <div class="step-item mb-3">
                    <h6 class="text-primary">${step.name}</h6>
                    <div class="step-content p-3 bg-light rounded">
                        <code>${step.text}</code>
                    </div>
                </div>
            `;
        });
        
        html += '</div>';
        return html;
    }
    
    formatTokenizationResults(results) {
        let html = '<h6>Tokenization Results</h6>';
        html += `
            <div class="row">
                <div class="col-md-6">
                    <div class="result-item">
                        <h6>Words (${results.word_count})</h6>
                        <div class="tokens-container">
                            ${results.words.map(word => `<span class="badge bg-primary me-1 mb-1">${word}</span>`).join('')}
                        </div>
                    </div>
                </div>
                <div class="col-md-6">
                    <div class="result-item">
                        <h6>Sentences (${results.sentence_count})</h6>
                        <ol class="sentences-list">
                            ${results.sentences.map(sentence => `<li>${sentence}</li>`).join('')}
                        </ol>
                    </div>
                </div>
            </div>
        `;
        return html;
    }
    
    formatPOSResults(results) {
        let html = '<h6>Part-of-Speech Tags</h6>';
        html += '<div class="pos-tags-container mb-3">';
        
        results.tagged.forEach(([word, pos]) => {
            const posClass = this.getPOSClass(pos);
            html += `<span class="pos-tag ${posClass}" title="${pos}">${word}</span>`;
        });
        
        html += '</div>';
        
        html += '<h6>POS Distribution</h6>';
        html += '<div class="pos-distribution">';
        
        Object.entries(results.pos_groups).forEach(([pos, words]) => {
            html += `
                <div class="pos-group mb-2">
                    <strong>${pos}:</strong> ${words.join(', ')}
                </div>
            `;
        });
        
        html += '</div>';
        return html;
    }
    
    formatNgramResults(results) {
        let html = '<h6>N-gram Analysis</h6>';
        
        ['unigrams', 'bigrams', 'trigrams'].forEach(type => {
            if (results[type]) {
                html += `
                    <div class="ngram-section mb-3">
                        <h6 class="text-primary">${type.charAt(0).toUpperCase() + type.slice(1)}</h6>
                        <div class="ngram-frequencies">
                            ${Object.entries(results[type].frequencies).map(([ngram, freq]) => 
                                `<span class="badge bg-secondary me-1 mb-1">${ngram} (${freq})</span>`
                            ).join('')}
                        </div>
                    </div>
                `;
            }
        });
        
        return html;
    }
    
    formatNERResults(results) {
        let html = '<h6>Named Entities</h6>';
        
        if (results.entities.length === 0) {
            html += '<div class="alert alert-info">No named entities found in the text.</div>';
            return html;
        }
        
        html += '<div class="entities-container mb-3">';
        results.entities.forEach(entity => {
            const entityClass = this.getEntityClass(entity.label);
            html += `<span class="entity-tag ${entityClass}" title="${entity.label}">${entity.entity}</span>`;
        });
        html += '</div>';
        
        html += '<h6>Entity Groups</h6>';
        Object.entries(results.entity_groups).forEach(([label, entities]) => {
            html += `
                <div class="entity-group mb-2">
                    <strong>${label}:</strong> ${entities.join(', ')}
                </div>
            `;
        });
        
        return html;
    }
    
    formatSentimentResults(results) {
        const sentimentClass = results.overall_sentiment.toLowerCase();
        let html = `
            <h6>Sentiment Analysis</h6>
            <div class="sentiment-overview mb-3">
                <h4 class="sentiment-${sentimentClass}">${results.overall_sentiment}</h4>
                <p>Confidence: ${(results.confidence * 100).toFixed(1)}%</p>
            </div>
            <div class="sentiment-breakdown">
                <h6>Detailed Scores</h6>
                <div class="row">
                    <div class="col-md-4">
                        <div class="sentiment-score">
                            <span class="sentiment-positive">Positive</span>
                            <strong>${(results.breakdown.positive * 100).toFixed(1)}%</strong>
                        </div>
                    </div>
                    <div class="col-md-4">
                        <div class="sentiment-score">
                            <span class="sentiment-negative">Negative</span>
                            <strong>${(results.breakdown.negative * 100).toFixed(1)}%</strong>
                        </div>
                    </div>
                    <div class="col-md-4">
                        <div class="sentiment-score">
                            <span class="sentiment-neutral">Neutral</span>
                            <strong>${(results.breakdown.neutral * 100).toFixed(1)}%</strong>
                        </div>
                    </div>
                </div>
            </div>
        `;
        return html;
    }
    
    formatStemmingResults(results) {
        let html = '<h6>Stemming Results</h6>';
        html += '<div class="stemming-pairs">';
        
        results.stemmed_pairs.forEach(([original, stemmed]) => {
            html += `
                <div class="word-pair mb-2">
                    <span class="original-word">${original}</span>
                    <span class="mx-2">→</span>
                    <span class="stemmed-word text-primary">${stemmed}</span>
                </div>
            `;
        });
        
        html += '</div>';
        return html;
    }
    
    formatLemmatizationResults(results) {
        let html = '<h6>Lemmatization Results</h6>';
        html += '<div class="lemmatization-pairs">';
        
        results.lemmatized_pairs.forEach(([original, lemmatized]) => {
            html += `
                <div class="word-pair mb-2">
                    <span class="original-word">${original}</span>
                    <span class="mx-2">→</span>
                    <span class="lemmatized-word text-primary">${lemmatized}</span>
                </div>
            `;
        });
        
        html += '</div>';
        return html;
    }
    
    formatChunkingResults(results) {
        let html = '<h6>Chunking Results</h6>';
        
        if (results.noun_phrases && results.noun_phrases.length > 0) {
            html += '<div class="noun-phrases mb-3">';
            html += '<h6>Noun Phrases</h6>';
            results.noun_phrases.forEach(phrase => {
                html += `<span class="badge bg-success me-1 mb-1">${phrase}</span>`;
            });
            html += '</div>';
        }
        
        // Parse tree is now displayed only in the visualization section
        html += '<div class="alert alert-info">Parse tree visualization available in the Visualization section.</div>';
        
        return html;
    }
    
    getPOSClass(pos) {
        if (pos.startsWith('NN')) return 'noun';
        if (pos.startsWith('VB')) return 'verb';
        if (pos.startsWith('JJ')) return 'adjective';
        if (pos.startsWith('RB')) return 'adverb';
        return 'other';
    }
    
    getEntityClass(label) {
        switch (label) {
            case 'PERSON': return 'person';
            case 'ORGANIZATION': case 'ORG': return 'organization';
            case 'GPE': case 'LOCATION': return 'location';
            default: return 'other';
        }
    }

    setupTranslationControlsIfNeeded() {
        if (this.moduleName !== 'machine_translation') return;
        const operationSelect = document.getElementById('operationSelect');
        if (!operationSelect) return;

        // Create inputs for source and target language codes
        const wrapper = document.createElement('div');
        wrapper.className = 'row g-2 mb-3';
        wrapper.innerHTML = `
            <div class="col-sm-6">
                <label for="srcLang" class="form-label">Source language code</label>
                <input id="srcLang" type="text" class="form-control" placeholder="auto" value="auto" />
                <div class="form-text">Leave as 'auto' to detect automatically.</div>
            </div>
            <div class="col-sm-6">
                <label for="destLang" class="form-label">Target language code</label>
                <input id="destLang" type="text" class="form-control" placeholder="en" value="en" />
                <div class="form-text">Examples: en, hi, fr, es, de, ta, te</div>
            </div>
        `;
        operationSelect.parentElement.insertBefore(wrapper, operationSelect.nextSibling);
    }

    setupSummarizationControlsIfNeeded() {
        if (this.moduleName !== 'text_summarization') return;
        const operationSelect = document.getElementById('operationSelect');
        if (!operationSelect) return;

        // Create inputs for min/max summary length
        const wrapper = document.createElement('div');
        wrapper.className = 'row g-2 mb-3';
        wrapper.innerHTML = `
            <div class="col-sm-6">
                <label for="minLen" class="form-label">Minimum summary length</label>
                <input id="minLen" type="number" class="form-control" value="30" min="10" max="300" />
            </div>
            <div class="col-sm-6">
                <label for="maxLen" class="form-label">Maximum summary length</label>
                <input id="maxLen" type="number" class="form-control" value="130" min="20" max="500" />
            </div>
        `;
        operationSelect.parentElement.insertBefore(wrapper, operationSelect.nextSibling);
    }

    setupTextGenerationControlsIfNeeded() {
        if (this.moduleName !== 'text_generation') return;
        const operationSelect = document.getElementById('operationSelect');
        if (!operationSelect) return;

        const wrapper = document.createElement('div');
        wrapper.className = 'row g-2 mb-3';
        wrapper.innerHTML = `
            <div class="col-sm-6">
                <label for="genMaxLen" class="form-label">Maximum length</label>
                <input id="genMaxLen" type="number" class="form-control" value="50" min="10" max="200" />
            </div>
            <div class="col-sm-6">
                <label for="genNumSequences" class="form-label">Number of sequences</label>
                <input id="genNumSequences" type="number" class="form-control" value="1" min="1" max="5" />
            </div>
        `;
        operationSelect.parentElement.insertBefore(wrapper, operationSelect.nextSibling);
    }

    formatTranslationResults(results) {
        if (results.error) {
            return `<div class="alert alert-danger">${results.error}</div>`;
        }
        const detected = results.detected_source || 'auto';
        const target = results.target_language || 'en';
        const original = results.original_text || '';
        const translated = results.translated_text || '';
        return `
            <h6>Translation</h6>
            <div class="mb-2"><strong>Detected source:</strong> ${detected}</div>
            <div class="mb-2"><strong>Target:</strong> ${target}</div>
            <div class="mb-2"><strong>Original:</strong><br/><div class="p-2 bg-light rounded"><code>${original}</code></div></div>
            <div class="mb-2"><strong>Translated:</strong><br/><div class="p-2 bg-light rounded"><code>${translated}</code></div></div>
        `;
    }

    formatSummarizationResults(results) {
        if (results.error) {
            return `<div class="alert alert-danger">${results.error}</div>`;
        }
        const summary = results.summary || '';
        const original = this.inputText.value || '';
        const compression = original && summary ? ((summary.length / original.length) * 100).toFixed(1) : null;
        return `
            <h6>Summary</h6>
            <div class="p-3 bg-light rounded mb-3"><code>${summary}</code></div>
            ${compression ? `<div class="text-muted">Compression: ${compression}% of original length</div>` : ''}
        `;
    }

    formatTextGenerationResults(results) {
        if (results.error) {
            return `<div class="alert alert-danger">${results.error}</div>`;
        }
        const outputs = results.generated || [];
        if (!Array.isArray(outputs) || outputs.length === 0) {
            return '<div class="alert alert-info">No text generated.</div>';
        }
        const items = outputs.map((t, idx) => `
            <div class="mb-3 p-2 bg-light rounded">
                <div class="small text-muted">Output ${idx + 1}</div>
                <code>${t}</code>
            </div>
        `).join('');
        return `
            <h6>Generated Text</h6>
            ${items}
        `;
    }
    
    displayVisualization(results) {
        const operation = this.operation.value;
        
        // Handle chunking parse tree visualization
        if (operation === 'chunk') {
            if (results.chunk_tree) {
                this.createParseTree(results.chunk_tree);
                return;
            } else {
                this.visualization.innerHTML = '<div class="alert alert-info">No parse tree available.</div>';
                return;
            }
        }
        
        if (!results.visualization) {
            this.visualization.innerHTML = '<div class="alert alert-secondary">No visualization available for this operation.</div>';
            return;
        }
        
        const visType = results.visualization.type;
        const visData = results.visualization.data;
        
        switch (visType) {
            case 'pos_chart':
                this.createPOSChart(visData);
                break;
            case 'ngram_chart':
                this.createNgramChart(visData);
                break;
            case 'entity_chart':
                this.createEntityChart(visData);
                break;
            case 'sentiment_chart':
                this.createSentimentChart(visData);
                break;
            default:
                this.visualization.innerHTML = '<div class="alert alert-info">Visualization not available for this operation.</div>';
        }
    }
    
    createPOSChart(data) {
        const canvas = document.createElement('canvas');
        canvas.id = 'posChart';
        canvas.width = 400;
        canvas.height = 300;
        
        this.visualization.innerHTML = '';
        this.visualization.appendChild(canvas);
        
        const ctx = canvas.getContext('2d');
        const labels = Object.keys(data);
        const values = Object.values(data).map(arr => arr.length);
        
        new Chart(ctx, {
            type: 'bar',
            data: {
                labels: labels,
                datasets: [{
                    label: 'Count',
                    data: values,
                    backgroundColor: 'rgba(13, 110, 253, 0.8)',
                    borderColor: 'rgba(13, 110, 253, 1)',
                    borderWidth: 1
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    y: {
                        beginAtZero: true
                    }
                }
            }
        });
    }
    
    createNgramChart(data) {
        const canvas = document.createElement('canvas');
        canvas.id = 'ngramChart';
        canvas.width = 400;
        canvas.height = 300;
        
        this.visualization.innerHTML = '';
        this.visualization.appendChild(canvas);
        
        const ctx = canvas.getContext('2d');
        
        // Use bigrams for visualization
        const bigramData = data.bigrams || {};
        const labels = Object.keys(bigramData);
        const values = Object.values(bigramData);
        
        new Chart(ctx, {
            type: 'doughnut',
            data: {
                labels: labels,
                datasets: [{
                    data: values,
                    backgroundColor: [
                        'rgba(255, 99, 132, 0.8)',
                        'rgba(54, 162, 235, 0.8)',
                        'rgba(255, 205, 86, 0.8)',
                        'rgba(75, 192, 192, 0.8)',
                        'rgba(153, 102, 255, 0.8)'
                    ]
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    title: {
                        display: true,
                        text: 'Top Bigrams'
                    }
                }
            }
        });
    }
    
    createEntityChart(data) {
        const canvas = document.createElement('canvas');
        canvas.id = 'entityChart';
        canvas.width = 400;
        canvas.height = 300;
        
        this.visualization.innerHTML = '';
        this.visualization.appendChild(canvas);
        
        const ctx = canvas.getContext('2d');
        const labels = Object.keys(data);
        const values = Object.values(data).map(arr => arr.length);
        
        new Chart(ctx, {
            type: 'pie',
            data: {
                labels: labels,
                datasets: [{
                    data: values,
                    backgroundColor: [
                        'rgba(255, 99, 132, 0.8)',
                        'rgba(54, 162, 235, 0.8)',
                        'rgba(255, 205, 86, 0.8)',
                        'rgba(75, 192, 192, 0.8)',
                        'rgba(153, 102, 255, 0.8)'
                    ]
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    title: {
                        display: true,
                        text: 'Entity Distribution'
                    }
                }
            }
        });
    }
    
    createSentimentChart(data) {
        const canvas = document.createElement('canvas');
        canvas.id = 'sentimentChart';
        canvas.width = 400;
        canvas.height = 300;
        
        this.visualization.innerHTML = '';
        this.visualization.appendChild(canvas);
        
        const ctx = canvas.getContext('2d');
        
        new Chart(ctx, {
            type: 'bar',
            data: {
                labels: ['Positive', 'Negative', 'Neutral'],
                datasets: [{
                    label: 'Sentiment Score',
                    data: [data.Positive, data.Negative, data.Neutral],
                    backgroundColor: [
                        'rgba(25, 135, 84, 0.8)',
                        'rgba(220, 53, 69, 0.8)',
                        'rgba(108, 117, 125, 0.8)'
                    ],
                    borderColor: [
                        'rgba(25, 135, 84, 1)',
                        'rgba(220, 53, 69, 1)',
                        'rgba(108, 117, 125, 1)'
                    ],
                    borderWidth: 1
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    y: {
                        beginAtZero: true,
                        max: 1
                    }
                }
            }
        });
    }
    
    createParseTree(chunkTree) {
        this.visualization.innerHTML = '';
        
        if (!chunkTree) {
            this.visualization.innerHTML = '<div class="alert alert-info">No parse tree available.</div>';
            return;
        }
        
        // Create a simple visual representation of the parse tree
        const treeContainer = document.createElement('div');
        treeContainer.className = 'parse-tree-container';
        treeContainer.style.cssText = `
            font-family: monospace;
            background: #f8f9fa;
            color: #212529;
            border: 1px solid #dee2e6;
            border-radius: 0.375rem;
            padding: 1rem;
            overflow-x: auto;
            white-space: pre-wrap;
            font-size: 0.9rem;
        `;
        
        let treeHTML = '<div class="parse-tree">';
        treeHTML += '<h6>Parse Tree Structure</h6>';
        treeHTML += '<div style="margin-top: 1rem;">';
        treeHTML += chunkTree.replace(/\n/g, '<br>').replace(/\s/g, '&nbsp;');
        treeHTML += '</div>';
        treeHTML += '</div>';
        
        treeContainer.innerHTML = treeHTML;
        this.visualization.appendChild(treeContainer);
    }
    
    displayError(message) {
        this.results.innerHTML = `
            <div class="alert alert-danger">
                <i data-feather="alert-circle"></i>
                ${message}
            </div>
        `;
        updateIcons();
    }
    
    clearResults() {
        this.inputText.value = '';
        this.results.innerHTML = '<div class="alert alert-info"><i data-feather="info"></i> Enter some text and click "Process Text" to see the results here.</div>';
        this.visualization.innerHTML = '<div class="alert alert-secondary"><i data-feather="eye"></i> Visual representation will appear here after processing.</div>';
        updateIcons();
    }
    
    showSpinner() {
        this.processingSpinner.style.display = 'block';
        this.results.style.display = 'none';
    }
    
    hideSpinner() {
        this.processingSpinner.style.display = 'none';
        this.results.style.display = 'block';
    }
}

// Initialize lab when page loads
function initializeLab(moduleName) {
    window.labManager = new LabManager(moduleName);
}

// Export for global use
window.initializeLab = initializeLab;
