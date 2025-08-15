// Main JavaScript file for NLP Virtual Lab
document.addEventListener('DOMContentLoaded', function() {
    // Initialize Feather icons
    feather.replace();
    
    // Add smooth scrolling for anchor links
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', function (e) {
            e.preventDefault();
            const target = document.querySelector(this.getAttribute('href'));
            if (target) {
                target.scrollIntoView({
                    behavior: 'smooth',
                    block: 'start'
                });
            }
        });
    });
    
    // Add animation to cards on scroll
    const observerOptions = {
        root: null,
        rootMargin: '0px',
        threshold: 0.1
    };
    
    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.classList.add('fade-in');
            }
        });
    }, observerOptions);
    
    // Observe all cards
    document.querySelectorAll('.card').forEach(card => {
        observer.observe(card);
    });
    
    // Add hover effects to buttons
    document.querySelectorAll('.btn').forEach(btn => {
        btn.addEventListener('mouseenter', function() {
            this.style.transform = 'translateY(-1px)';
        });
        
        btn.addEventListener('mouseleave', function() {
            this.style.transform = 'translateY(0)';
        });
    });
    
    // Initialize tooltips if Bootstrap is available
    if (typeof bootstrap !== 'undefined') {
        var tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
        tooltipTriggerList.map(function (tooltipTriggerEl) {
            return new bootstrap.Tooltip(tooltipTriggerEl);
        });
    }
});

// Utility functions
function showAlert(message, type = 'info') {
    const alertDiv = document.createElement('div');
    alertDiv.className = `alert alert-${type} alert-dismissible fade show`;
    alertDiv.innerHTML = `
        ${message}
        <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
    `;
    
    // Insert at the top of the main content
    const main = document.querySelector('main');
    main.insertBefore(alertDiv, main.firstChild);
    
    // Auto-dismiss after 5 seconds
    setTimeout(() => {
        if (alertDiv.parentNode) {
            alertDiv.remove();
        }
    }, 5000);
}

function formatTime(seconds) {
    const minutes = Math.floor(seconds / 60);
    const remainingSeconds = seconds % 60;
    return `${minutes}:${remainingSeconds.toString().padStart(2, '0')}`;
}

function copyToClipboard(text) {
    navigator.clipboard.writeText(text).then(function() {
        showAlert('Copied to clipboard!', 'success');
    }, function() {
        showAlert('Failed to copy to clipboard', 'error');
    });
}

// API helper functions
async function makeAPIRequest(url, options = {}) {
    try {
        const response = await fetch(url, {
            headers: {
                'Content-Type': 'application/json',
                ...options.headers
            },
            ...options
        });
        
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        return await response.json();
    } catch (error) {
        console.error('API request failed:', error);
        throw error;
    }
}

// Make functions globally available
window.makeAPIRequest = makeAPIRequest;

// Progress tracking
class ProgressTracker {
    constructor() {
        this.progress = this.loadProgress();
    }
    
    loadProgress() {
        const saved = localStorage.getItem('nlp_lab_progress');
        return saved ? JSON.parse(saved) : {};
    }
    
    saveProgress() {
        localStorage.setItem('nlp_lab_progress', JSON.stringify(this.progress));
    }
    
    updateModuleProgress(moduleName, data) {
        this.progress[moduleName] = {
            ...this.progress[moduleName],
            ...data,
            lastUpdated: new Date().toISOString()
        };
        this.saveProgress();
    }
    
    getModuleProgress(moduleName) {
        return this.progress[moduleName] || {
            completed: false,
            score: 0,
            attempts: 0
        };
    }
    
    getOverallProgress() {
        const modules = Object.keys(this.progress);
        const completed = modules.filter(m => this.progress[m].completed).length;
        const total = modules.length;
        return total > 0 ? (completed / total) * 100 : 0;
    }
}

// Global progress tracker instance
window.progressTracker = new ProgressTracker();

// Theme management
class ThemeManager {
    constructor() {
        this.theme = this.getStoredTheme() || 'dark';
        this.applyTheme();
    }
    
    getStoredTheme() {
        return localStorage.getItem('nlp_lab_theme');
    }
    
    setStoredTheme(theme) {
        localStorage.setItem('nlp_lab_theme', theme);
    }
    
    applyTheme() {
        document.documentElement.setAttribute('data-bs-theme', this.theme);
    }
    
    toggleTheme() {
        this.theme = this.theme === 'dark' ? 'light' : 'dark';
        this.setStoredTheme(this.theme);
        this.applyTheme();
    }
}

// Global theme manager instance
window.themeManager = new ThemeManager();

// Error handling
window.addEventListener('error', function(event) {
    console.error('Global error:', event.error);
    showAlert('An unexpected error occurred. Please try again.', 'danger');
});

window.addEventListener('unhandledrejection', function(event) {
    console.error('Unhandled promise rejection:', event.reason);
    showAlert('An error occurred while processing your request.', 'danger');
});

// Performance monitoring
window.addEventListener('load', function() {
    const loadTime = performance.now();
    console.log(`Page loaded in ${loadTime.toFixed(2)}ms`);
});

// Accessibility improvements
document.addEventListener('keydown', function(event) {
    // Add keyboard navigation support
    if (event.key === 'Escape') {
        // Close any open modals or dropdowns
        const openModals = document.querySelectorAll('.modal.show');
        openModals.forEach(modal => {
            if (modal.classList.contains('show')) {
                const bsModal = bootstrap.Modal.getInstance(modal);
                if (bsModal) {
                    bsModal.hide();
                }
            }
        });
    }
});

// Update icons when new content is loaded
function updateIcons() {
    feather.replace();
}

// Export utility functions for use in other modules
window.nlpLabUtils = {
    showAlert,
    formatTime,
    copyToClipboard,
    makeAPIRequest,
    updateIcons
};
