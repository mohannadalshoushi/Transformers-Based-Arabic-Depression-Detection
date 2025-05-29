/**
 * Arabic Depression Detection - Main JavaScript File
 * Handles the ChatGPT-like interface and model interactions
 */

class DepressionDetectionApp {
    constructor() {
        this.selectedModel = '';
        this.isProcessing = false;
        
        // DOM elements
        this.modelSelect = document.getElementById('modelSelect');
        this.textInputSection = document.getElementById('textInputSection');
        this.textInput = document.getElementById('textInput');
        this.sendBtn = document.getElementById('sendBtn');
        this.sendSection = document.getElementById('sendSection');
        this.messagesArea = document.getElementById('messagesArea');
        this.charCount = document.getElementById('charCount');
        
        this.initializeEventListeners();
    }
    
    initializeEventListeners() {
        // Model selection change
        this.modelSelect.addEventListener('change', (e) => {
            this.handleModelSelection(e.target.value);
        });
        
        // Text input events
        this.textInput.addEventListener('input', (e) => {
            this.handleTextInput(e.target.value);
        });
        
        this.textInput.addEventListener('keypress', (e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                this.handleSubmit();
            }
        });
        
        // Send button click
        this.sendBtn.addEventListener('click', () => {
            this.handleSubmit();
        });
    }
    
    handleModelSelection(selectedModel) {
        this.selectedModel = selectedModel;
        
        if (selectedModel) {
            // Show text input and send button
            this.textInputSection.style.display = 'block';
            this.sendSection.style.display = 'block';
            this.textInput.disabled = false;
            this.textInput.focus();
        } else {
            // Hide text input and send button
            this.textInputSection.style.display = 'none';
            this.sendSection.style.display = 'none';
            this.textInput.disabled = true;
            this.textInput.value = '';
            this.charCount.textContent = '0';
        }
        this.updateSendButton();
    }
    
    handleTextInput(text) {
        const length = text.length;
        this.charCount.textContent = length;
        
        // Update character count color
        if (length > 900) {
            this.charCount.style.color = 'hsl(var(--danger-color))';
        } else if (length > 700) {
            this.charCount.style.color = 'hsl(var(--warning-color))';
        } else {
            this.charCount.style.color = 'hsl(var(--text-muted))';
        }
        
        this.updateSendButton();
    }
    
    updateSendButton() {
        const hasText = this.textInput.value.trim().length >= 3;
        const hasModel = this.selectedModel !== '';
        
        this.sendBtn.disabled = !hasText || !hasModel || this.isProcessing;
    }
    
    autoResizeTextarea() {
        this.textInput.style.height = 'auto';
        this.textInput.style.height = Math.min(this.textInput.scrollHeight, 200) + 'px';
    }
    
    async handleSubmit() {
        if (this.isProcessing) return;
        
        const text = this.textInput.value.trim();
        if (!text || !this.selectedModel) return;
        
        // Validate text length
        if (text.length < 3) {
            this.showError('النص قصير جداً. يرجى إدخال نص أطول');
            return;
        }
        
        if (text.length > 1000) {
            this.showError('النص طويل جداً. يرجى إدخال نص أقصر من 1000 حرف');
            return;
        }
        
        // Clear previous results
        this.clearResults();
        
        // Clear input
        this.textInput.value = '';
        this.handleTextInput('');
        
        // Show loading state
        this.setLoadingState(true);
        
        try {
            // Send request to backend
            const response = await fetch('/predict', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    text: text,
                    model: this.selectedModel
                })
            });
            
            const data = await response.json();
            
            if (data.error) {
                this.showError(data.message);
            } else {
                this.displayResult(data.result);
            }
            
        } catch (error) {
            console.error('Error:', error);
            this.showError('حدث خطأ في الاتصال بالخادم. يرجى المحاولة مرة أخرى');
        } finally {
            this.setLoadingState(false);
        }
    }
    
    setLoadingState(loading) {
        this.isProcessing = loading;
        this.sendBtn.classList.toggle('loading', loading);
        this.updateSendButton();
        
        if (loading) {
            const loadingDiv = document.createElement('div');
            loadingDiv.className = 'loading-message mb-3';
            loadingDiv.innerHTML = `
                <div class="alert alert-info">
                    <i class="fas fa-spinner fa-spin me-2"></i>
                    جاري تحليل النص...
                </div>
            `;
            this.messagesArea.appendChild(loadingDiv);
            this.scrollToBottom();
        } else {
            this.removeLoadingMessage();
        }
    }
    
    clearResults() {
        // Remove all result messages, keep only welcome message
        const results = this.messagesArea.querySelectorAll('.user-message, .bot-message, .loading-message');
        results.forEach(result => result.remove());
    }
    
    addBotMessage(content, className = '') {
        const messageDiv = document.createElement('div');
        messageDiv.className = `message-bubble bot-message ${className}`;
        messageDiv.innerHTML = `
            <div class="message-content">
                <div class="d-flex align-items-center mb-2">
                    <i class="fas fa-robot me-2 text-primary"></i>
                    <strong>النظام</strong>
                </div>
                <div>${content}</div>
            </div>
        `;
        
        this.messagesArea.appendChild(messageDiv);
        this.scrollToBottom();
        
        return messageDiv;
    }
    
    removeLoadingMessage() {
        const loadingMessage = this.messagesArea.querySelector('.loading-message');
        if (loadingMessage) {
            loadingMessage.remove();
        }
    }
    
    displayResult(result) {
        const isDepression = result.depression_detected;
        const confidence = result.confidence ? Math.round(result.confidence * 100) : null;
        
        const alertClass = isDepression ? 'alert-danger' : 'alert-success';
        const resultIcon = isDepression ? 'fas fa-exclamation-triangle' : 'fas fa-check-circle';
        
        const messageDiv = document.createElement('div');
        messageDiv.className = 'bot-message mb-3';
        messageDiv.innerHTML = `
            <div class="alert ${alertClass}">
                <h6 class="alert-heading">
                    <i class="${resultIcon} me-2"></i>
                    نتيجة التحليل
                </h6>
                <p class="mb-1"><strong>${result.result_text}</strong></p>
                <hr>
                <p class="mb-0 small">النموذج: ${result.model_name.toUpperCase()}</p>
            </div>
        `;
        
        this.messagesArea.appendChild(messageDiv);
        this.scrollToBottom();
    }
    
    showError(message) {
        const content = `
            <div class="error-message">
                <i class="fas fa-exclamation-circle me-2"></i>
                ${message}
            </div>
        `;
        
        this.addBotMessage(content);
    }
    
    getModelDisplayName(modelName) {
        const modelNames = {
            'svm': 'SVM - نموذج دعم الآلة المتجه',
            'stacking': 'Stacking - النموذج المجمع',
            'lstm': 'LSTM - الشبكة العصبية',
            'arabicbert': 'ArabicBERT - نموذج المحولات'
        };
        
        return modelNames[modelName] || modelName;
    }
    
    escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }
    
    scrollToBottom() {
        setTimeout(() => {
            this.messagesArea.scrollTop = this.messagesArea.scrollHeight;
        }, 100);
    }
}

// Initialize the application when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    new DepressionDetectionApp();
});

// Service Worker Registration (for PWA capabilities)
if ('serviceWorker' in navigator) {
    window.addEventListener('load', () => {
        navigator.serviceWorker.register('/static/sw.js')
            .then((registration) => {
                console.log('SW registered: ', registration);
            })
            .catch((registrationError) => {
                console.log('SW registration failed: ', registrationError);
            });
    });
}

// Handle online/offline status
window.addEventListener('online', () => {
    console.log('Back online');
});

window.addEventListener('offline', () => {
    console.log('Gone offline');
});
