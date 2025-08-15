// Quiz functionality for NLP Virtual Lab
class QuizManager {
    constructor(moduleName) {
        this.moduleName = moduleName;
        this.questions = [];
        this.currentQuestion = 0;
        this.userAnswers = [];
        this.score = 0;
        this.initializeElements();
        this.setupEventListeners();
    }
    
    initializeElements() {
        this.quizStart = document.getElementById('quizStart');
        this.quizQuestions = document.getElementById('quizQuestions');
        this.quizResults = document.getElementById('quizResults');
        this.quizLoading = document.getElementById('quizLoading');
        this.startQuizBtn = document.getElementById('startQuizBtn');
        this.questionCount = document.getElementById('questionCount');
        this.questionContainer = document.getElementById('questionContainer');
        this.currentQuestionNum = document.getElementById('currentQuestionNum');
        this.totalQuestions = document.getElementById('totalQuestions');
        this.progressBar = document.getElementById('progressBar');
        this.prevBtn = document.getElementById('prevBtn');
        this.nextBtn = document.getElementById('nextBtn');
        this.submitBtn = document.getElementById('submitBtn');
        this.finalScore = document.getElementById('finalScore');
        this.resultTitle = document.getElementById('resultTitle');
        this.resultMessage = document.getElementById('resultMessage');
        this.resultIcon = document.getElementById('resultIcon');
        this.retakeQuizBtn = document.getElementById('retakeQuizBtn');
    }
    
    setupEventListeners() {
        this.startQuizBtn.addEventListener('click', () => this.startQuiz());
        this.prevBtn.addEventListener('click', () => this.previousQuestion());
        this.nextBtn.addEventListener('click', () => this.nextQuestion());
        this.submitBtn.addEventListener('click', () => this.submitQuiz());
        this.retakeQuizBtn.addEventListener('click', () => this.retakeQuiz());
    }
    
    // Load questions on initialization to show count
    async initializeQuiz() {
        await this.getQuestionCount();
    }
    
    async loadQuestions() {
        this.showLoading();
        
        try {
            const response = await makeAPIRequest(`/api/quiz/${this.moduleName}`);
            this.questions = response;
            this.userAnswers = new Array(this.questions.length).fill(null);
            this.questionCount.textContent = this.questions.length;
            this.totalQuestions.textContent = this.questions.length;
            this.hideLoading();
        } catch (error) {
            console.error('Failed to load quiz questions:', error);
            this.displayError('Failed to load quiz questions. Please try again.');
        }
    }
    
    // Method to get question count without UI changes
    async getQuestionCount() {
        try {
            const response = await makeAPIRequest(`/api/quiz/${this.moduleName}`);
            this.questions = response;
            this.questionCount.textContent = this.questions.length;
            this.totalQuestions.textContent = this.questions.length;
            return this.questions.length;
        } catch (error) {
            console.error('Failed to load quiz questions:', error);
            return 0;
        }
    }
    
    async startQuiz() {
        await this.loadQuestions();
        
        if (this.questions.length === 0) {
            showAlert('No questions available for this module.', 'warning');
            return;
        }
        
        this.currentQuestion = 0;
        this.showQuestions();
        this.displayQuestion();
    }
    
    displayQuestion() {
        const question = this.questions[this.currentQuestion];
        
        let html = `
            <div class="question-content">
                <h5 class="question-text">${question.question}</h5>
                <div class="options-container mt-3">
        `;
        
        question.options.forEach((option, index) => {
            const isSelected = this.userAnswers[this.currentQuestion] === index;
            html += `
                <button class="question-option ${isSelected ? 'selected' : ''}" 
                        data-option="${index}">
                    ${option}
                </button>
            `;
        });
        
        html += `
                </div>
            </div>
        `;
        
        this.questionContainer.innerHTML = html;
        
        // Add event listeners to options
        this.questionContainer.querySelectorAll('.question-option').forEach(btn => {
            btn.addEventListener('click', (e) => {
                const option = parseInt(e.target.dataset.option);
                this.selectAnswer(option);
            });
        });
        
        this.updateProgress();
        this.updateNavigation();
    }
    
    selectAnswer(optionIndex) {
        this.userAnswers[this.currentQuestion] = optionIndex;
        
        // Update visual selection
        this.questionContainer.querySelectorAll('.question-option').forEach((btn, index) => {
            btn.classList.toggle('selected', index === optionIndex);
        });
        
        this.updateNavigation();
    }
    
    updateProgress() {
        this.currentQuestionNum.textContent = this.currentQuestion + 1;
        const progress = ((this.currentQuestion + 1) / this.questions.length) * 100;
        this.progressBar.style.width = `${progress}%`;
    }
    
    updateNavigation() {
        this.prevBtn.disabled = this.currentQuestion === 0;
        
        const isLastQuestion = this.currentQuestion === this.questions.length - 1;
        const hasAnswered = this.userAnswers[this.currentQuestion] !== null;
        
        if (isLastQuestion) {
            this.nextBtn.style.display = 'none';
            this.submitBtn.style.display = 'inline-block';
            this.submitBtn.disabled = !hasAnswered;
        } else {
            this.nextBtn.style.display = 'inline-block';
            this.submitBtn.style.display = 'none';
            this.nextBtn.disabled = !hasAnswered;
        }
    }
    
    previousQuestion() {
        if (this.currentQuestion > 0) {
            this.currentQuestion--;
            this.displayQuestion();
        }
    }
    
    nextQuestion() {
        if (this.currentQuestion < this.questions.length - 1) {
            this.currentQuestion++;
            this.displayQuestion();
        }
    }
    
    async submitQuiz() {
        // Check if all questions are answered
        const unanswered = this.userAnswers.some(answer => answer === null);
        if (unanswered) {
            if (typeof showAlert === 'function') {
                showAlert('Please answer all questions before submitting.', 'warning');
            } else {
                alert('Please answer all questions before submitting.');
            }
            return;
        }
        
        try {
            // Calculate score locally
            let correct = 0;
            for (let i = 0; i < this.questions.length; i++) {
                if (this.userAnswers[i] === this.questions[i].correct_answer) {
                    correct++;
                }
            }
            
            this.score = Math.round((correct / this.questions.length) * 100);
            this.showResults();
            
            // Update progress tracking
            if (typeof progressTracker !== 'undefined') {
                progressTracker.updateModuleProgress(this.moduleName, {
                    quizCompleted: true,
                    score: this.score,
                    attempts: (progressTracker.getModuleProgress(this.moduleName).attempts || 0) + 1
                });
            }
            
        } catch (error) {
            console.error('Failed to submit quiz:', error);
            if (typeof showAlert === 'function') {
                showAlert('Failed to submit quiz. Please try again.', 'danger');
            } else {
                alert('Failed to submit quiz. Please try again.');
            }
        }
    }
    
    showResults() {
        this.finalScore.textContent = `${this.score}%`;
        
        // Set result message and icon based on score
        if (this.score >= 80) {
            this.resultTitle.textContent = 'Excellent!';
            this.resultMessage.innerHTML = '<strong>Great job!</strong> You have a solid understanding of this topic.';
            this.resultMessage.className = 'alert alert-success';
            this.resultIcon.innerHTML = '<i data-feather="award" class="quiz-icon text-success"></i>';
        } else if (this.score >= 60) {
            this.resultTitle.textContent = 'Good Work!';
            this.resultMessage.innerHTML = '<strong>Well done!</strong> You have a good grasp of the concepts.';
            this.resultMessage.className = 'alert alert-info';
            this.resultIcon.innerHTML = '<i data-feather="thumbs-up" class="quiz-icon text-info"></i>';
        } else {
            this.resultTitle.textContent = 'Keep Learning!';
            this.resultMessage.innerHTML = '<strong>Don\'t worry!</strong> Review the material and try again.';
            this.resultMessage.className = 'alert alert-warning';
            this.resultIcon.innerHTML = '<i data-feather="book" class="quiz-icon text-warning"></i>';
        }
        
        this.showResultsScreen();
        updateIcons();
    }
    
    retakeQuiz() {
        this.currentQuestion = 0;
        this.userAnswers = new Array(this.questions.length).fill(null);
        this.score = 0;
        this.showQuestions();
        this.displayQuestion();
    }
    
    showStartScreen() {
        this.quizStart.style.display = 'block';
        this.quizQuestions.style.display = 'none';
        this.quizResults.style.display = 'none';
        this.quizLoading.style.display = 'none';
    }
    
    showQuestions() {
        this.quizStart.style.display = 'none';
        this.quizQuestions.style.display = 'block';
        this.quizResults.style.display = 'none';
        this.quizLoading.style.display = 'none';
    }
    
    showResultsScreen() {
        this.quizStart.style.display = 'none';
        this.quizQuestions.style.display = 'none';
        this.quizResults.style.display = 'block';
        this.quizLoading.style.display = 'none';
    }
    
    showLoading() {
        this.quizStart.style.display = 'none';
        this.quizQuestions.style.display = 'none';
        this.quizResults.style.display = 'none';
        this.quizLoading.style.display = 'block';
    }
    
    hideLoading() {
        this.quizLoading.style.display = 'none';
    }
    
    displayError(message) {
        this.quizLoading.innerHTML = `
            <div class="card-body text-center">
                <div class="alert alert-danger">
                    <i data-feather="alert-circle"></i>
                    ${message}
                </div>
                <button class="btn btn-primary" onclick="location.reload()">
                    <i data-feather="refresh-cw"></i>
                    Retry
                </button>
            </div>
        `;
        updateIcons();
    }
}

// Initialize quiz when page loads
function initializeQuiz(moduleName) {
    window.quizManager = new QuizManager(moduleName);
    return window.quizManager;
}

// Export for global use
window.initializeQuiz = initializeQuiz;

// Also make QuizManager available globally
window.QuizManager = QuizManager;
