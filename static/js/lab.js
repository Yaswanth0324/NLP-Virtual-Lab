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
                
                // For question answering module, also populate questions
                if (this.moduleName === 'question_answering' && e.target.dataset.questions) {
                    this.populateQuestions(e.target.dataset.questions);
                }
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
        this.setupTopicModellingExamplesIfNeeded();
        this.setupQuestionAnsweringControlsIfNeeded();
        this.setupSpeechProcessingControlsIfNeeded();
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
                { value: 'chunk', label: 'Noun Phrase Chunking' },
                { value: 'chunk_vp', label: 'Verb Phrase Chunking' },
                { value: 'cfg_parse', label: 'CFG Parse' },
                { value: 'cnf', label: 'Convert to CNF' }
            ],
            'text_classification': [
                { value: 'text_classify', label: 'Text Classification' }
            ],
            'word_embeddings': [
                { value: 'word_embeddings', label: 'Word Embeddings' },
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
            ],
            'topic_modelling': [
                { value: 'topic_modelling', label: 'Topic Modelling (LDA)' }
            ],
            'question_answering': [
                { value: 'question_answer', label: 'Question Answering' }
            ],
            'speech_processing': [
                { value: 'text_to_speech', label: 'Text-to-Speech (TTS)' },
                { value: 'speech_to_text', label: 'Speech-to-Text (STT)' }
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
            `,
            'topic_modelling': `
                <h6>Understanding Topic Modelling (LDA)</h6>
                <p>Topic modelling automatically finds groups of related words (topics) that frequently appear together across documents. LDA assumes each document is a mixture of topics, and each topic is a mixture of words.</p>
                <h6 class="mt-2">How to use this lab</h6>
                <ol>
                  <li>Paste multiple short documents, one per line (recommended), or a longer paragraph.</li>
                  <li>Select <strong>Topic Modelling (LDA)</strong> and click <strong>Process Text</strong>.</li>
                  <li>View the discovered topics and their top words. Use the buttons to switch the topic shown in the chart.</li>
                </ol>
                <h6 class="mt-2">How to read the output</h6>
                <ul>
                  <li><strong>Topics:</strong> Each topic is shown as a list of top words (the strongest signals for that theme).</li>
                  <li><strong>Chart:</strong> Horizontal bars show the most important words for the selected topic.</li>
                  <li><strong>Document mixture:</strong> Internally, each document can contain multiple topics in different proportions.</li>
                </ul>
                <h6 class="mt-2">Tips</h6>
                <ul>
                  <li>Provide at least 5–8 short documents (one per line) for clearer topics.</li>
                  <li>Keep each line focused (e.g., headlines, short paragraphs).</li>
                  <li>Avoid only stopwords; include meaningful content words.</li>
                </ul>
                <h6 class="mt-2">Limitations</h6>
                <ul>
                  <li>Topics are approximate; similar words may appear in multiple topics.</li>
                  <li>Very small inputs can produce unstable topics—try adding more lines.</li>
                </ul>
            `,
            'question_answering': `
                <h6>Question Answering</h6>
                <p>Provide a <strong>context</strong> (in the main text box) and a <strong>question</strong> (in the input below). The model finds the most likely answer span inside the context.</p>
                <ul>
                    <li><strong>Context:</strong> A paragraph or two containing the answer.</li>
                    <li><strong>Question:</strong> A clear, specific question that can be answered from the context.</li>
                    <li><strong>Output:</strong> Extracted answer text and a confidence score.</li>
                </ul>
                <p class="text-muted">Tip: Use factual passages (e.g., Wikipedia-like text) for best results.</p>
            `,
            'speech_processing': `
                <h6>Understanding Speech Processing</h6>
                <p>Speech Processing connects human voice with text by using two inverse operations:</p>
                <ul>
                    <li><strong>Speech-to-Text (STT):</strong> Converts spoken audio from your microphone into written text in real time.</li>
                    <li><strong>Text-to-Speech (TTS):</strong> Converts the text you type into natural-sounding speech using a chosen voice.</li>
                </ul>
                <h6 class="mt-3">How it works in your browser</h6>
                <ul>
                    <li><strong>STT:</strong> Uses the Web Speech API (SpeechRecognition) to stream audio from your mic, detect words, and produce <em>interim</em> (live) and <em>final</em> results. Interim text can change as recognition improves; final text is stable.</li>
                    <li><strong>TTS:</strong> Uses the Speech Synthesis API to pick a system-installed voice, then reads your text aloud with adjustable <em>rate</em>, <em>pitch</em>, and <em>volume</em>.</li>
                </ul>
                <h6 class="mt-3">When to use which</h6>
                <ul>
                    <li><strong>Use STT</strong> to dictate notes, transcribe short explanations, or capture quick ideas without typing.</li>
                    <li><strong>Use TTS</strong> to listen to passages, check pronunciation and pacing, or make content more accessible.</li>
                </ul>
                <h6 class="mt-3">Tips for best results</h6>
                <ul>
                    <li>Use a quiet environment and speak clearly and steadily for STT.</li>
                    <li>Choose the recognition <strong>language</strong> that matches how you will speak (e.g., en-US, hi-IN).</li>
                    <li>On TTS, experiment with <strong>voice</strong>, <strong>rate</strong>, and <strong>pitch</strong> to match your preference.</li>
                </ul>
                <h6 class="mt-3">Browser support and security</h6>
                <ul>
                    <li>STT works best in Chromium-based browsers (Chrome/Edge). Support varies in others.</li>
                    <li>Microphone requires a <strong>secure context</strong> (HTTPS) or <strong>localhost</strong> (127.0.0.1).</li>
                    <li>Allow mic permission when prompted and select the correct input device.</li>
                </ul>
                <p class="text-muted">This lab runs entirely in the browser—no audio is sent to your server by default.</p>
            `
        };
        
        this.explanationContent.innerHTML = explanations[this.moduleName] || 
            '<p>Learn about this NLP concept through interactive examples and practice.</p>';
    }
    
    async processText() {
        const text = this.inputText.value.trim();
        const operation = this.operation.value;
        
        // Allow empty text when using Speech-to-Text in Speech Processing lab
        if (!text && !(this.moduleName === 'speech_processing' && operation === 'speech_to_text')) {
            showAlert('Please enter some text to process.', 'warning');
            return;
        }

        // Handle client-side operations for Speech Processing
        if (this.moduleName === 'speech_processing') {
            if (operation === 'text_to_speech') {
                this.speechSpeak(text);
                this.results.innerHTML = '<div class="alert alert-info">Speaking the provided text using the selected voice.</div>';
                updateIcons();
                return;
            }
            if (operation === 'speech_to_text') {
                this.startSpeechRecognition();
                this.results.innerHTML = '<div class="alert alert-info">Recording started. Speak into your microphone. Click Stop to finish and insert the text.</div>';
                updateIcons();
                return;
            }
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
            } else if (operation === 'question_answer' || this.moduleName === 'question_answering') {
                const qInput = document.getElementById('qaQuestion');
                const question = qInput ? qInput.value.trim() : '';
                const context = text;
                if (!question) {
                    this.hideSpinner();
                    showAlert('Please enter a question for Question Answering.', 'warning');
                    this.processBtn.disabled = false;
                    return;
                }
                endpoint = '/qa';
                payload = { question, context };
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
            case 'text_classify':
                html = this.formatClassificationResults(results);
                break;
            case 'stem':
                html = this.formatStemmingResults(results);
                break;
            case 'lemmatize':
                html = this.formatLemmatizationResults(results);
                break;
            case 'word_embeddings':
                html = this.formatEmbeddingsResults(results);
                break;
            case 'chunk':
                html = this.formatChunkingResults(results);
                break;
            case 'chunk_vp':
                html = this.formatChunkingResults(results);
                break;
            case 'cfg_parse':
                html = this.formatParseResults(results);
                break;
            case 'cnf':
                html = this.formatParseResults(results);
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
            case 'topic_modelling':
                html = this.formatTopicModellingResults(results);
                break;
            case 'question_answer':
                html = this.formatQAResults(results);
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

    formatClassificationResults(results) {
        if (results.error) {
            return `<div class="alert alert-danger">${results.error}</div>`;
        }
        const predicted = results.predicted_label || 'Unknown';
        const confidence = results.confidence || 0;
        const probs = results.probabilities || {};
        const keywords = results.keywords_found || {};
        // Build a list of top classes by probability
        const entries = Object.entries(probs).sort((a,b) => b[1]-a[1]);
        const listItems = entries.map(([label, p]) => `<li>${label}: ${(p*100).toFixed(1)}%</li>`).join('');
        // Keywords for the predicted class (if any)
        const keyList = (keywords[predicted] || []).map(k => `<span class="badge bg-secondary me-1 mb-1">${k}</span>`).join('');
        return `
            <h6>Text Classification</h6>
            <div class="mb-3">
                <div><strong>Predicted Class:</strong> ${predicted}</div>
                <div><strong>Confidence:</strong> ${(confidence*100).toFixed(1)}%</div>
            </div>
            <div class="mb-3">
                <h6>Class Probabilities</h6>
                <ul>${listItems}</ul>
            </div>
            ${keyList ? `<div class="mb-2"><h6>Signals found for "${predicted}"</h6>${keyList}</div>` : ''}
        `;
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

        if (results.verb_phrases && results.verb_phrases.length > 0) {
            html += '<div class="verb-phrases mb-3">';
            html += '<h6>Verb Phrases</h6>';
            results.verb_phrases.forEach(phrase => {
                html += `<span class="badge bg-primary me-1 mb-1">${phrase}</span>`;
            });
            html += '</div>';
        }
        
        if (results.ascii_tree) {
            html += '<div class="mb-2"><strong>ASCII Tree</strong></div>';
            html += `<pre class="bg-light p-2 rounded" style="white-space: pre; overflow:auto;"><code>${this._esc(results.ascii_tree)}</code></pre>`;
        }
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

    setupQuestionAnsweringControlsIfNeeded() {
        if (this.moduleName !== 'question_answering') return;
        const operationSelect = document.getElementById('operationSelect');
        if (!operationSelect) return;

        // Question input (context is the main text box)
        const wrapper = document.createElement('div');
        wrapper.className = 'mb-3';
        wrapper.innerHTML = `
            <label for="qaQuestion" class="form-label">Question</label>
            <input id="qaQuestion" type="text" class="form-control" placeholder="e.g., What is the capital of France?" />
            <div class="form-text">Context should be in the main text box above.</div>
        `;
        operationSelect.parentElement.insertBefore(wrapper, operationSelect.nextSibling);

        // Add questions section
        const questionsWrapper = document.createElement('div');
        questionsWrapper.className = 'mb-3';
        questionsWrapper.innerHTML = `
            <label class="form-label">Sample Questions</label>
            <div id="sampleQuestions" class="sample-questions">
                <div class="text-muted">Click on an example text above to see related questions here.</div>
            </div>
        `;
        wrapper.parentElement.insertBefore(questionsWrapper, wrapper.nextSibling);
    }

    // --- Speech Processing (TTS/STT) Controls ---
    setupSpeechProcessingControlsIfNeeded() {
        if (this.moduleName !== 'speech_processing') return;
        const operationSelect = document.getElementById('operationSelect');
        if (!operationSelect) return;

        const container = document.createElement('div');
        container.id = 'speechControls';
        container.className = 'mt-3';
        container.innerHTML = `
            <div class="card">
                <div class="card-body">
                    <div id="ttsControls" class="mb-3" style="display:none;">
                        <div class="row g-3 mb-3">
                            <div class="col-md-6">
                                <label class="form-label">Voice</label>
                                <select id="ttsVoiceSelect" class="form-select"></select>
                            </div>
                            <div class="col-md-6">
                                <label class="form-label">Voice filter (lang or name)</label>
                                <input id="ttsVoiceFilter" class="form-control" placeholder="e.g., en, hi, te" />
                            </div>
                        </div>
                        <div class="row g-3 mb-3">
                            <div class="col-md-4">
                                <label class="form-label">Rate</label>
                                <input id="ttsRate" class="form-range" type="range" min="0.5" max="2" step="0.1" value="1" />
                                <div class="small text-muted">Current: <span id="ttsRateVal">1.0</span></div>
                            </div>
                            <div class="col-md-4">
                                <label class="form-label">Pitch</label>
                                <input id="ttsPitch" class="form-range" type="range" min="0" max="2" step="0.1" value="1" />
                                <div class="small text-muted">Current: <span id="ttsPitchVal">1.0</span></div>
                            </div>
                            <div class="col-md-4">
                                <label class="form-label">Volume</label>
                                <input id="ttsVolume" class="form-range" type="range" min="0" max="1" step="0.05" value="1" />
                                <div class="small text-muted">Current: <span id="ttsVolumeVal">1.00</span></div>
                            </div>
                        </div>
                        <div class="d-flex gap-2">
                            <button id="btnTTSSpeak" type="button" class="btn btn-primary"><i data-feather="play"></i> Speak</button>
                            <button id="btnTTSStop" type="button" class="btn btn-outline-danger"><i data-feather="square"></i> Stop</button>
                            <button id="btnTTSPreview" type="button" class="btn btn-outline-secondary"><i data-feather="headphones"></i> Preview Voice</button>
                        </div>
                        <div id="ttsSupport" class="mt-2 small"></div>
                    </div>

                    <div id="sttControls" style="display:none;">
                        <div class="row g-3 mb-3">
                            <div class="col-md-6">
                                <label class="form-label">Recognition language</label>
                                <select id="sttLangSelect" class="form-select">
                                    <option value="en-US">English (US) - en-US</option>
                                    <option value="en-GB">English (UK) - en-GB</option>
                                    <option value="hi-IN">Hindi (India) - hi-IN</option>
                                    <option value="te-IN">Telugu (India) - te-IN</option>
                                    <option value="ta-IN">Tamil (India) - ta-IN</option>
                                    <option value="bn-IN">Bengali (India) - bn-IN</option>
                                    <option value="mr-IN">Marathi (India) - mr-IN</option>
                                    <option value="gu-IN">Gujarati (India) - gu-IN</option>
                                    <option value="kn-IN">Kannada (India) - kn-IN</option>
                                    <option value="ml-IN">Malayalam (India) - ml-IN</option>
                                    <option value="pa-IN">Punjabi (India) - pa-IN</option>
                                </select>
                            </div>
                            <div class="col-md-3">
                                <div class="form-check mt-4">
                                    <input id="sttContinuousToggle" class="form-check-input" type="checkbox" checked />
                                    <label class="form-check-label" for="sttContinuousToggle">Continuous</label>
                                </div>
                            </div>
                            <div class="col-md-3">
                                <div class="form-check mt-4">
                                    <input id="sttInterimToggle" class="form-check-input" type="checkbox" checked />
                                    <label class="form-check-label" for="sttInterimToggle">Interim</label>
                                </div>
                            </div>
                        </div>
                        <div class="d-flex gap-2 align-items-center mb-2">
                            <button id="btnSTTStart" type="button" class="btn btn-success"><i data-feather="mic"></i> Start Recording</button>
                            <button id="btnSTTStop" type="button" class="btn btn-outline-danger" disabled><i data-feather="square"></i> Stop</button>
                            <button id="btnSTTClear" type="button" class="btn btn-outline-secondary"><i data-feather="trash-2"></i> Clear</button>
                            <span id="sttStatusBadge" class="badge bg-secondary ms-auto">Idle</span>
                        </div>
                        <div class="small text-muted mb-1">Transcript (also copied into the main input when stopped)</div>
                        <div id="sttTranscript" class="form-control" style="min-height: 100px;"></div>
                        <div id="sttSupport" class="mt-2 small"></div>
                    </div>
                </div>
            </div>
        `;

        operationSelect.parentElement.insertBefore(container, operationSelect.nextSibling);
        updateIcons();

        const updateVisibility = () => {
            const op = this.operation.value;
            const showTTS = (op === 'text_to_speech');
            const showSTT = (op === 'speech_to_text');
            document.getElementById('ttsControls').style.display = showTTS ? '' : 'none';
            document.getElementById('sttControls').style.display = showSTT ? '' : 'none';

            // Hide the main input and main process/clear controls when STT is selected
            const inputWrapper = this.inputText ? this.inputText.parentElement : null;
            const controlsWrapper = this.processBtn ? this.processBtn.parentElement : null;
            if (inputWrapper) inputWrapper.style.display = showSTT ? 'none' : '';
            if (controlsWrapper) controlsWrapper.style.display = showSTT ? 'none' : '';

            // Hide only the main Process button when TTS is selected (keep Clear visible)
            if (this.processBtn) this.processBtn.style.display = showTTS ? 'none' : '';

            // Hide the Example Texts card when STT is selected
            const examples = document.querySelector('.example-texts');
            if (examples) {
                const examplesCard = examples.closest('.card') || examples;
                examplesCard.style.display = showSTT ? 'none' : '';
            }
            // Hide the transcript area under the microphone when STT is selected (output will show in Results)
            const transcriptDiv = document.getElementById('sttTranscript');
            if (transcriptDiv) {
                transcriptDiv.style.display = showSTT ? 'none' : '';
                const label = transcriptDiv.previousElementSibling;
                if (label && label.classList && label.classList.contains('small')) {
                    label.style.display = showSTT ? 'none' : '';
                }
            }

            updateIcons();
        };
        this.operation.addEventListener('change', updateVisibility);
        updateVisibility();

        // Initialize TTS and STT subsystems
        this._initTTS();
        this._initSTT();
    }

    _initTTS() {
        this.ttsVoices = [];
        const support = ('speechSynthesis' in window) && ('SpeechSynthesisUtterance' in window);
        const supportDiv = document.getElementById('ttsSupport');
        if (supportDiv) supportDiv.innerHTML = support ? '<span class="badge bg-success">TTS supported</span>' : '<span class="badge bg-danger">TTS not supported in this browser</span>';
        if (!support) return;

        const populate = () => {
            this.ttsVoices = window.speechSynthesis.getVoices().slice();
            const select = document.getElementById('ttsVoiceSelect');
            if (!select) return;
            const filter = (document.getElementById('ttsVoiceFilter')?.value || '').toLowerCase();
            select.innerHTML = '';
            const filtered = this.ttsVoices.filter(v => !filter || v.lang.toLowerCase().includes(filter) || (v.name || '').toLowerCase().includes(filter));
            filtered.forEach(v => {
                const opt = document.createElement('option');
                opt.value = v.name;
                opt.textContent = `${v.name} — ${v.lang}${v.default ? ' (default)' : ''}`;
                select.appendChild(opt);
            });
            if (select.options.length === 0) {
                const opt = document.createElement('option');
                opt.value = '';
                opt.textContent = 'No voices available';
                select.appendChild(opt);
            }
        };

        populate();
        if (window.speechSynthesis.onvoiceschanged !== undefined) {
            window.speechSynthesis.onvoiceschanged = populate;
        }

        const rate = document.getElementById('ttsRate');
        const pitch = document.getElementById('ttsPitch');
        const volume = document.getElementById('ttsVolume');
        const rateVal = document.getElementById('ttsRateVal');
        const pitchVal = document.getElementById('ttsPitchVal');
        const volumeVal = document.getElementById('ttsVolumeVal');
        if (rate) rate.addEventListener('input', () => rateVal.textContent = Number(rate.value).toFixed(1));
        if (pitch) pitch.addEventListener('input', () => pitchVal.textContent = Number(pitch.value).toFixed(1));
        if (volume) volume.addEventListener('input', () => volumeVal.textContent = Number(volume.value).toFixed(2));

        const btnSpeak = document.getElementById('btnTTSSpeak');
        const btnStop = document.getElementById('btnTTSStop');
        const btnPreview = document.getElementById('btnTTSPreview');
        if (btnSpeak) btnSpeak.addEventListener('click', () => this.speechSpeak(this.inputText.value));
        if (btnStop) btnStop.addEventListener('click', () => this.speechStopSpeak());
        if (btnPreview) btnPreview.addEventListener('click', () => this.speechSpeak('This is a short voice preview.'));
    }

    speechSpeak(text) {
        const support = ('speechSynthesis' in window) && ('SpeechSynthesisUtterance' in window);
        if (!support) {
            showAlert('Text-to-Speech is not supported in this browser.', 'danger');
            return;
        }
        const trimmed = (text || '').trim();
        if (!trimmed) {
            showAlert('Enter some text to speak.', 'warning');
            return;
        }
        window.speechSynthesis.cancel();
        const utt = new SpeechSynthesisUtterance(trimmed);
        const select = document.getElementById('ttsVoiceSelect');
        const rate = document.getElementById('ttsRate');
        const pitch = document.getElementById('ttsPitch');
        const volume = document.getElementById('ttsVolume');
        const voices = window.speechSynthesis.getVoices();
        const chosen = Array.from((select || {}).options || []).find(o => o.selected);
        if (chosen) {
            const v = voices.find(v => v.name === chosen.value);
            if (v) utt.voice = v;
        }
        utt.rate = Number(rate?.value || 1);
        utt.pitch = Number(pitch?.value || 1);
        utt.volume = Number(volume?.value || 1);
        window.speechSynthesis.speak(utt);
    }

    speechStopSpeak() {
        if ('speechSynthesis' in window) window.speechSynthesis.cancel();
    }

    _initSTT() {
        const SR = window.SpeechRecognition || window.webkitSpeechRecognition;
        const supportDiv = document.getElementById('sttSupport');
        const supported = !!SR;
        if (supportDiv) supportDiv.innerHTML = supported ? '<span class="badge bg-success">STT supported</span>' : '<span class="badge bg-danger">STT not supported in this browser</span>';
        if (!supported) return;

        this.recognition = new SR();
        this.recognizing = false;
        this.recognition.continuous = true;
        this.recognition.interimResults = true;
        this.recognition.lang = (document.getElementById('sttLangSelect')?.value) || 'en-US';

        this.recognition.onstart = () => {
            this.recognizing = true;
            const b = document.getElementById('sttStatusBadge');
            if (b) b.textContent = 'Listening';
            const start = document.getElementById('btnSTTStart');
            const stop = document.getElementById('btnSTTStop');
            if (start) start.disabled = true;
            if (stop) stop.disabled = false;
        };
        this.recognition.onerror = (e) => {
            const b = document.getElementById('sttStatusBadge');
            if (b) b.textContent = 'Error';
            const msg = this._sttExplainError(e && e.error);
            const sup = document.getElementById('sttSupport');
            if (sup) {
                sup.innerHTML = `<span class="badge bg-danger">STT error</span> ${msg}`;
            }
            console.error('STT error:', e);
        };
        this.recognition.onend = () => {
            this.recognizing = false;
            const b = document.getElementById('sttStatusBadge');
            if (b) b.textContent = 'Idle';
            const start = document.getElementById('btnSTTStart');
            const stop = document.getElementById('btnSTTStop');
            if (start) start.disabled = false;
            if (stop) stop.disabled = true;
        };

        this._sttInterim = '';
        this._sttFinal = '';
        this.recognition.onresult = (event) => {
            this._sttInterim = '';
            for (let i = event.resultIndex; i < event.results.length; ++i) {
                const res = event.results[i];
                if (res.isFinal) this._sttFinal += res[0].transcript;
                else this._sttInterim += res[0].transcript;
            }
            // Show live recognized text in the Results section (not under microphone)
            const recognizedHtml = `${this._esc(this._sttFinal)} ${this._sttInterim ? '<span class="text-muted">' + this._esc(this._sttInterim) + '</span>' : ''}`.trim();
            this.results.innerHTML = `<h6>Speech-to-Text (Live)</h6><div class="p-2 bg-light rounded">${recognizedHtml || '<span class="text-muted">Listening...</span>'}</div>`;
            updateIcons();
        };

        const langSel = document.getElementById('sttLangSelect');
        const cont = document.getElementById('sttContinuousToggle');
        const inter = document.getElementById('sttInterimToggle');
        if (langSel) langSel.addEventListener('change', () => { if (this.recognition) this.recognition.lang = langSel.value; });
        if (cont) cont.addEventListener('change', () => { if (this.recognition) this.recognition.continuous = !!cont.checked; });
        if (inter) inter.addEventListener('change', () => { if (this.recognition) this.recognition.interimResults = !!inter.checked; });

        const startBtn = document.getElementById('btnSTTStart');
        const stopBtn = document.getElementById('btnSTTStop');
        const clearBtn = document.getElementById('btnSTTClear');
        if (startBtn) startBtn.addEventListener('click', () => this.startSpeechRecognition());
        if (stopBtn) stopBtn.addEventListener('click', () => this.stopSpeechRecognition());
        if (clearBtn) clearBtn.addEventListener('click', () => this.clearSpeechTranscript());
    }

    startSpeechRecognition() {
        const SR = window.SpeechRecognition || window.webkitSpeechRecognition;
        if (!SR || !this.recognition) {
            showAlert('Speech recognition is not supported in this browser.', 'danger');
            return;
        }
        // Security context check (required by browsers for mic access)
        const isLocalSecure = (window.isSecureContext === true) || ['localhost', '127.0.0.1'].includes(location.hostname);
        if (!isLocalSecure) {
            showAlert('Microphone access requires a secure context (HTTPS) or localhost. Open the site via https:// or http://127.0.0.1.', 'warning');
        }
        const start = () => {
            // Reset previous transcript buffers and UI before new recording
            this._sttFinal = '';
            this._sttInterim = '';
            const div = document.getElementById('sttTranscript');
            if (div) div.innerHTML = '';
            // Inform user in the Results section
            this.results.innerHTML = '<div class="alert alert-info">Recording started. Speak into your microphone. Click Stop to finalize.</div>';
            try { this.recognition.start(); } catch (_) { /* already started */ }
        };
        const md = navigator.mediaDevices;
        if (md && typeof md.getUserMedia === 'function') {
            md.getUserMedia({ audio: true }).then(stream => {
                // Close stream immediately; SpeechRecognition handles capture
                try { (stream.getTracks() || []).forEach(t => t.stop()); } catch (_) {}
                start();
            }).catch(err => {
                const sup = document.getElementById('sttSupport');
                if (sup) sup.innerHTML = `<span class="badge bg-danger">Mic error</span> ${this._sttExplainError('permission')}`;
                showAlert('Microphone permission denied or unavailable.', 'danger');
                console.error('getUserMedia error:', err);
            });
        } else {
            start();
        }
    }

    stopSpeechRecognition() {
        if (this.recognition && this.recognizing) this.recognition.stop();
        const txtDiv = document.getElementById('sttTranscript');
        let finalText = '';
        if (txtDiv) {
            const stripped = txtDiv.innerText || txtDiv.textContent || '';
            finalText = (this._sttFinal || stripped || '').trim();
            this.inputText.value = finalText;
        } else {
            finalText = (this._sttFinal || '').trim();
            this.inputText.value = finalText;
        }
        // Show final recognized text in Results section
        this.results.innerHTML = `<h6>Speech-to-Text (Final)</h6><div class="p-2 bg-light rounded"><code>${this._esc(finalText)}</code></div>`;
        updateIcons();
    }

    clearSpeechTranscript() {
        const div = document.getElementById('sttTranscript');
        if (div) div.innerHTML = '';
        this.inputText.value = '';
        // Reset recognition buffers so previous text does not reappear
        this._sttFinal = '';
        this._sttInterim = '';
        // Reflect cleared state in Results section
        this.results.innerHTML = '<div class="alert alert-secondary">Transcript cleared. Click Start Recording to begin a new session.</div>';
        updateIcons();
    }

    _esc(s) { return (s || '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;','\'':'&#39;'}[c])); }

    _sttExplainError(code) {
        switch (code) {
            case 'not-allowed':
            case 'service-not-allowed':
            case 'permission':
                return 'Microphone permission denied or blocked. Allow mic access in your browser site settings and reload.';
            case 'no-speech':
                return 'No speech detected. Check your microphone and try again.';
            case 'audio-capture':
                return 'No microphone found or it is in use by another app. Check your audio device.';
            case 'aborted':
                return 'Recording was aborted (user action or a new start()).';
            case 'network':
                return 'Network error with speech service. Check your connection and try again.';
            case 'bad-grammar':
            case 'language-not-supported':
                return 'Recognition language/grammar not supported by this browser.';
            default:
                return 'An error occurred. Use a Chromium-based browser over HTTPS or localhost and ensure microphone permission is granted.';
        }
    }

    populateQuestions(questionsJson) {
        try {
            const questions = JSON.parse(questionsJson);
            const questionsContainer = document.getElementById('sampleQuestions');
            
            if (!questionsContainer || !Array.isArray(questions)) return;
            
            questionsContainer.innerHTML = '';
            
            questions.forEach((question, index) => {
                const questionBtn = document.createElement('button');
                questionBtn.className = 'btn btn-sm btn-outline-secondary me-2 mb-2 question-btn';
                questionBtn.textContent = question;
                questionBtn.addEventListener('click', (e) => {
                    e.preventDefault();
                    const qaInput = document.getElementById('qaQuestion');
                    if (qaInput) {
                        qaInput.value = question;
                        qaInput.focus();
                    }
                });
                questionsContainer.appendChild(questionBtn);
            });
        } catch (error) {
            console.error('Error parsing questions:', error);
        }
    }

    setupTopicModellingExamplesIfNeeded() {
        if (this.moduleName !== 'topic_modelling') return;
        const container = document.querySelector('.example-texts');
        if (!container) return;

        const examples = [
            {
                label: 'Tech News',
                text: [
                    'AI startup releases open-source model for image generation.',
                    'Cloud providers compete on GPU availability and pricing.',
                    'Regulators release guidance on responsible AI deployments.',
                    'Researchers publish paper on efficient transformers.',
                    'Edge devices run quantized models with low power.',
                    'Startups race to build foundation model tooling.',
                    'Big tech announces partnerships with chip makers.',
                    'Universities launch AI ethics programs.'
                ].join('\n')
            },
            {
                label: 'Health & Fitness',
                text: [
                    'Regular exercise improves cardiovascular health.',
                    'High fiber diet aids digestion and satiety.',
                    'Sleep quality affects concentration and mood.',
                    'Hydration supports metabolism and endurance.',
                    'Strength training increases bone density.',
                    'Yoga reduces stress and improves flexibility.',
                    'Walking after meals regulates blood sugar.',
                    'Sedentary lifestyle correlates with weight gain.'
                ].join('\n')
            },
            {
                label: 'Politics & Policy',
                text: [
                    'Parliament debates a new data privacy bill.',
                    'Opposition raises concerns about surveillance.',
                    'Committee proposes transparency requirements.',
                    'Court rules on constitutionality of the act.',
                    'Public consultations gather citizen feedback.',
                    'Minister announces digital rights charter.',
                    'Election commission updates campaign finance rules.',
                    'Local councils pilot open data portals.'
                ].join('\n')
            },
            {
                label: 'Sports Articles',
                text: [
                    'The team secured a narrow win in the final.',
                    'Star striker scored two goals in stoppage time.',
                    'Coach emphasized defense during practice.',
                    'Injuries forced a change in the lineup.',
                    'Fans celebrated the championship parade.',
                    'Midfielder transferred to a rival club.',
                    'Referee decisions sparked controversy.',
                    'Training camp focused on endurance drills.'
                ].join('\n')
            },
            {
                label: 'Business & Startups',
                text: [
                    'The startup closed a Series A funding round.',
                    'SaaS revenue grew due to enterprise deals.',
                    'Customer churn decreased after onboarding revamp.',
                    'Founders expanded into the European market.',
                    'Pricing experiments improved gross margins.',
                    'Partnerships drove new distribution channels.',
                    'Layoffs followed a strategic pivot.',
                    'Board approved a share buyback plan.'
                ].join('\n')
            },
            {
                label: 'Science & Environment',
                text: [
                    'Researchers measured rising ocean temperatures.',
                    'A new battery chemistry improved energy density.',
                    'Astronomers detected a distant exoplanet.',
                    'Ecologists tracked biodiversity in rainforests.',
                    'Physicists tested a quantum error-correction method.',
                    'Glacier retreat accelerated during heat waves.',
                    'Chemists developed biodegradable plastics.',
                    'Engineers built a low-cost air-quality sensor.'
                ].join('\n')
            }
        ];

        // Replace existing examples with topic modelling examples
        container.innerHTML = '';
        examples.forEach(ex => {
            const btn = document.createElement('button');
            btn.className = 'btn btn-sm btn-outline-info me-2 mb-2 example-btn';
            btn.setAttribute('data-text', ex.text);
            btn.textContent = ex.label;
            btn.addEventListener('click', () => {
                this.inputText.value = ex.text;
                this.inputText.focus();
            });
            container.appendChild(btn);
        });
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

    formatParseResults(results) {
        let html = '<h6>Parsing Results</h6>';
        if (results.tokens) {
            html += `<div class="mb-2"><strong>Tokens:</strong> ${results.tokens.join(' ')}</div>`;
        }
        if (Array.isArray(results.parse_trees) && results.parse_trees.length > 0) {
            html += '<div class="mb-2"><strong>Top parse(s):</strong></div>';
            results.parse_trees.forEach((t, i) => {
                html += `<div class="p-2 bg-light rounded mb-2"><div class="small text-muted">Parse ${i+1}</div><pre class="mb-0"><code>${this._esc(t)}</code></pre></div>`;
            });
        }
        if (results.original_tree) {
            html += '<div class="mb-2"><strong>Original tree:</strong></div>';
            html += `<div class="p-2 bg-light rounded mb-2"><pre class="mb-0"><code>${this._esc(results.original_tree)}</code></pre></div>`;
        }
        if (results.cnf_tree) {
            html += '<div class="mb-2"><strong>CNF tree:</strong></div>';
            html += `<div class="p-2 bg-light rounded mb-2"><pre class="mb-0"><code>${this._esc(results.cnf_tree)}</code></pre></div>`;
        }
        if (results.ascii_tree) {
            html += '<div class="mb-2"><strong>ASCII Tree</strong></div>';
            html += `<pre class="bg-light p-2 rounded" style="white-space: pre; overflow:auto;"><code>${this._esc(results.ascii_tree)}</code></pre>`;
        }
        html += '<div class="alert alert-info">Parse tree visualization available in the Visualization section.</div>';
        return html;
    }

    createParseTree(treeStr, asciiStr) {
        // Prefer ASCII pretty-printed tree if provided
        const ascii = (asciiStr && typeof asciiStr === 'string' && asciiStr.trim().length > 0) ? asciiStr : treeStr;
        this.visualization.innerHTML = `<pre class="bg-light p-2 rounded" style="white-space: pre; overflow:auto;"><code>${this._esc(ascii)}</code></pre>`;
    }
    
    displayVisualization(results) {
        const operation = this.operation.value;
        
        // Handle parse tree visualization for chunking & parsing ops
        if (['chunk','chunk_vp','cfg_parse','cnf'].includes(operation)) {
            if (results.chunk_tree || results.ascii_tree) {
                if (typeof this.createParseTree === 'function') {
                    this.createParseTree(results.chunk_tree || '', results.ascii_tree || '');
                } else {
                    const ascii = results.ascii_tree || results.chunk_tree || '';
                    this.visualization.innerHTML = '<pre class="bg-light p-2 rounded" style="white-space: pre; overflow:auto;"><code>' + this._esc(ascii) + '</code></pre>';
                }
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
            case 'embedding_projection':
                this.createEmbeddingProjection(visData);
                break;
            case 'classification_chart':
                this.createClassificationChart(visData);
                break;
            case 'topic_words':
                this.createTopicWordsChart(visData);
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

    createClassificationChart(data) {
        const canvas = document.createElement('canvas');
        canvas.id = 'classificationChart';
        canvas.width = 400;
        canvas.height = 300;

        this.visualization.innerHTML = '';
        this.visualization.appendChild(canvas);

        const ctx = canvas.getContext('2d');
        // Sort classes by probability and keep top 6
        const entries = Object.entries(data || {}).sort((a,b) => b[1]-a[1]).slice(0, 6);
        const labels = entries.map(([k]) => k);
        const values = entries.map(([,v]) => v);

        new Chart(ctx, {
            type: 'bar',
            data: {
                labels,
                datasets: [{
                    label: 'Probability',
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
                        beginAtZero: true,
                        max: 1
                    }
                }
            }
        });
    }

    createTopicWordsChart(topics) {
        // topics: [{topic, words, distribution}]
        this.visualization.innerHTML = '';

        const wrapper = document.createElement('div');
        wrapper.style.width = '100%';
        this.visualization.appendChild(wrapper);

        if (!Array.isArray(topics) || topics.length === 0) {
            wrapper.innerHTML = '<div class="alert alert-secondary">No topics found.</div>';
            return;
        }

        // Topic selector
        const tabs = document.createElement('div');
        tabs.className = 'mb-2';
        wrapper.appendChild(tabs);

        // Fixed-height chart container to prevent runaway resizing
        const chartHolder = document.createElement('div');
        chartHolder.className = 'chart-container';
        chartHolder.style.width = '100%';
        chartHolder.style.height = '320px';
        chartHolder.style.position = 'relative';
        wrapper.appendChild(chartHolder);
        const canvas = document.createElement('canvas');
        chartHolder.appendChild(canvas);

        let chart = null;
        const render = (topic) => {
            const words = Array.isArray(topic.words) ? topic.words.slice() : [];
            const weights = Array.isArray(topic.weights) ? topic.weights.slice() : null;

            // Build pairs of {word, value}. If weights exist, use them; otherwise use descending rank
            let pairs = words.map((w, i) => ({ w, v: weights ? Number(weights[i]) || 0 : (words.length - i) }));
            // If weights provided, sort descending by weight to ensure most important first
            if (weights) {
                pairs.sort((a, b) => b.v - a.v);
            }
            // Limit to first 12 items for readability
            const maxItems = Math.min(pairs.length, 12);
            pairs = pairs.slice(0, maxItems);

            // For horizontal bar, first label renders at top row and ensure finite values
            const labels = pairs.map(p => p.w);
            const values = pairs.map(p => {
                const v = Number(p.v);
                return Number.isFinite(v) ? v : 0;
            });

            const ctx = canvas.getContext('2d');
            if (chart) chart.destroy();
            chart = new Chart(ctx, {
                type: 'bar',
                data: {
                    labels,
                    datasets: [{
                        label: `Topic ${topic.topic} top words (weight)`,
                        data: values,
                        backgroundColor: 'rgba(13, 110, 253, 0.8)'
                    }]
                },
                options: {
                    indexAxis: 'y',
                    responsive: true,
                    maintainAspectRatio: false,
                    animation: false,
                    plugins: { legend: { display: false } },
                    scales: {
                        x: { beginAtZero: true }
                    },
                    layout: { padding: { top: 4, right: 8, bottom: 4, left: 8 } }
                }
            });
        };

        topics.forEach((t, idx) => {
            const btn = document.createElement('button');
            btn.className = `btn btn-sm ${idx === 0 ? 'btn-primary' : 'btn-outline-primary'} me-2`;
            btn.textContent = `Topic ${t.topic}`;
            btn.addEventListener('click', () => {
                tabs.querySelectorAll('button').forEach((b, i) => {
                    b.className = `btn btn-sm ${i === idx ? 'btn-primary' : 'btn-outline-primary'} me-2`;
                });
                render(t);
            });
            tabs.appendChild(btn);
        });

        render(topics[0]);
    }

    formatTopicModellingResults(results) {
        if (results.error) {
            return `<div class="alert alert-danger">${results.error}</div>`;
        }
        const num = results.num_topics || 0;
        const topics = Array.isArray(results.topics) ? results.topics : [];
        const list = topics.map(t => {
            const words = (t.words || []).map(w => `<span class="badge bg-secondary me-1 mb-1">${w}</span>`).join('');
            return `<div class="mb-3"><strong>Topic ${t.topic}:</strong><div class="mt-1">${words}</div></div>`;
        }).join('');
        return `
            <h6>Topic Modelling</h6>
            <div class="mb-2"><strong>Topics:</strong> ${num}</div>
            ${list || '<div class="text-muted">No topics extracted.</div>'}
            <div class="text-muted">Use the Visualization section to switch between topics.</div>
        `;
    }

    formatQAResults(results) {
        if (results.error) {
            return `<div class="alert alert-danger">${results.error}</div>`;
        }
        const answer = results.answer || '';
        const score = typeof results.score === 'number' ? (results.score * 100).toFixed(2) : '—';
        if (!answer) {
            return '<div class="alert alert-info">No answer found in the given context.</div>';
        }
        return `
            <h6>Answer</h6>
            <div class="p-2 bg-light rounded mb-2"><code>${answer}</code></div>
            <div class="text-muted">Confidence: ${score}%</div>
        `;
    }
    
    formatEmbeddingsResults(results) {
        if (results.error) {
            return `<div class="alert alert-danger">${results.error}</div>`;
        }
        const model = results.model || 'unknown';
        const dim = results.dimension || 0;
        const emb = results.embeddings || {};
        const words = Object.keys(emb);
        const list = words.map(w => {
            const v = emb[w];
            return `<li><code>${w}</code> — ${Array.isArray(v) ? `vector[${v.length}]` : '<span class="text-muted">out-of-vocabulary</span>'}</li>`;
        }).join('');
        return `
            <h6>Word Embeddings</h6>
            <div class="mb-2"><strong>Model:</strong> ${model}</div>
            <div class="mb-3"><strong>Dimension:</strong> ${dim}</div>
            <div class="mb-2"><strong>Words:</strong></div>
            <ul>${list}</ul>
            <div class="text-muted">A 2D projection (PCA) is shown in the Visualization section if enough words are in vocabulary.</div>
        `;
    }

    createEmbeddingProjection(data) {
        // data: { word: {x, y, oov?}, ... }
        this.visualization.innerHTML = '';

        // Create container that stretches to available width inside flex parent
        const container = document.createElement('div');
        container.style.height = '320px';
        container.style.width = '100%';
        container.style.position = 'relative';
        container.style.background = '#1e1e1e';
        container.style.border = '1px solid #2b2b2b';
        container.style.borderRadius = '6px';
        container.style.padding = '8px';
        this.visualization.appendChild(container);

        const keys = Object.keys(data || {});
        if (keys.length === 0) {
            container.innerHTML = '<div class="alert alert-secondary">No projection available.</div>';
            return;
        }

        // Colors for in-vocab and OOV
        const colorIV = 'rgba(13, 110, 253, 0.9)'; // blue
        const colorOOV = 'rgba(255, 159, 64, 0.95)'; // orange

        // Tooltip element
        const tooltip = document.createElement('div');
        tooltip.style.cssText = 'position:absolute; pointer-events:none; background:rgba(0,0,0,0.8); color:#fff; padding:4px 6px; border-radius:4px; font-size:12px; display:none; z-index:5;';
        container.appendChild(tooltip);

        // Legend
        const legend = document.createElement('div');
        legend.style.cssText = 'position:absolute; top:8px; left:8px; display:flex; gap:10px; align-items:center; background:rgba(0,0,0,0.4); padding:4px 6px; border-radius:4px; color:#eee; font-size:12px;';
        legend.innerHTML = `
            <span style="display:inline-flex; align-items:center; gap:6px;"><span style="width:10px; height:10px; border-radius:50%; background:${colorIV}; display:inline-block;"></span>In-vocab</span>
            <span style="display:inline-flex; align-items:center; gap:6px;"><span style="width:10px; height:10px; border-radius:50%; background:${colorOOV}; display:inline-block;"></span>OOV</span>
        `;
        container.appendChild(legend);

        const xs = keys.map(k => data[k].x);
        const ys = keys.map(k => data[k].y);
        const minX = Math.min(...xs), maxX = Math.max(...xs);
        const minY = Math.min(...ys), maxY = Math.max(...ys);

        const pad = 24;

        // Create SVG and ensure it fills the container
        const svgNS = 'http://www.w3.org/2000/svg';
        const svg = document.createElementNS(svgNS, 'svg');
        svg.setAttribute('width', '100%');
        svg.setAttribute('height', '100%');
        svg.style.display = 'block';
        container.appendChild(svg);

        // Compute dimensions after SVG is in the DOM
        const bounds = container.getBoundingClientRect();
        const innerWidth = Math.max(0, bounds.width - pad * 2) || Math.max(0, this.visualization.clientWidth - pad * 2) || 600;
        const innerHeight = Math.max(0, (container.clientHeight || 300) - pad * 2);

        // viewBox for crisp scaling, include padding
        svg.setAttribute('viewBox', `0 0 ${innerWidth + pad * 2} ${innerHeight + pad * 2}`);
        svg.setAttribute('preserveAspectRatio', 'xMidYMid meet');

        // Axes (optional light grid)
        const gridColor = 'rgba(255,255,255,0.08)';
        const drawLine = (x1,y1,x2,y2) => {
            const line = document.createElementNS(svgNS, 'line');
            line.setAttribute('x1', x1); line.setAttribute('y1', y1);
            line.setAttribute('x2', x2); line.setAttribute('y2', y2);
            line.setAttribute('stroke', gridColor);
            line.setAttribute('stroke-width', '1');
            svg.appendChild(line);
        };
        // Border
        drawLine(pad, pad, innerWidth + pad, pad);
        drawLine(pad, innerHeight + pad, innerWidth + pad, innerHeight + pad);
        drawLine(pad, pad, pad, innerHeight + pad);
        drawLine(innerWidth + pad, pad, innerWidth + pad, innerHeight + pad);

        const scaleX = v => pad + ((v - minX) / (maxX - minX || 1)) * innerWidth;
        const scaleY = v => pad + (1 - ((v - minY) / (maxY - minY || 1))) * innerHeight; // invert Y

        // Draw points and labels with hover tooltips
        keys.forEach(k => {
            const point = data[k] || {x: 0, y: 0};
            const cx = scaleX(point.x);
            const cy = scaleY(point.y);
            const oov = !!point.oov;
            const fill = oov ? colorOOV : colorIV;

            const circle = document.createElementNS(svgNS, 'circle');
            circle.setAttribute('cx', cx);
            circle.setAttribute('cy', cy);
            circle.setAttribute('r', 4.5);
            circle.setAttribute('fill', fill);
            circle.style.cursor = 'default';
            const title = document.createElementNS(svgNS, 'title');
            title.textContent = `${k} ${oov ? '(OOV)' : ''}`;
            circle.appendChild(title);
            svg.appendChild(circle);

            const label = document.createElementNS(svgNS, 'text');
            label.setAttribute('x', cx + 6);
            label.setAttribute('y', cy - 6);
            label.setAttribute('fill', '#ddd');
            label.setAttribute('font-size', '12');
            label.textContent = k;
            svg.appendChild(label);

            // Tooltip interactions
            const showTip = (evt) => {
                tooltip.style.display = 'block';
                tooltip.innerHTML = `<strong>${k}</strong> ${oov ? '<span style="color:#ffb26b">(OOV)</span>' : ''}`;
                const rect = container.getBoundingClientRect();
                const x = (evt.clientX - rect.left) + 10;
                const y = (evt.clientY - rect.top) + 10;
                tooltip.style.left = `${x}px`;
                tooltip.style.top = `${y}px`;
            };
            const hideTip = () => { tooltip.style.display = 'none'; };

            circle.addEventListener('mouseenter', showTip);
            circle.addEventListener('mousemove', showTip);
            circle.addEventListener('mouseleave', hideTip);
            label.addEventListener('mouseenter', showTip);
            label.addEventListener('mousemove', showTip);
            label.addEventListener('mouseleave', hideTip);
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
