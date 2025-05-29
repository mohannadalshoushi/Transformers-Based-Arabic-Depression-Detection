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
        this.textInputGroup = document.getElementById('textInputGroup');
        this.textInput = document.getElementById('textInput');
        this.sendBtn = document.getElementById('sendBtn');
        this.messagesArea = document.getElementById('messagesArea');
        this.characterCount = document.getElementById('characterCount');
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
        
        // Auto-resize textarea
        this.textInput.addEventListener('input', () => {
            this.autoResizeTextarea();
        });
    }
    
    handleModelSelection(selectedModel) {
        this.selectedModel = selectedModel;
        
        if (selectedModel) {
            // Show text input area
            this.textInputGroup.style.display = 'flex';
            this.characterCount.style.display = 'block';
            this.textInput.disabled = false;
            this.textInput.focus();
            
            // Add model selection message
            this.addBotMessage(`تم اختيار نموذج: <strong>${this.getModelDisplayName(selectedModel)}</strong><br>يمكنك الآن كتابة النص للتحليل.`);
        } else {
            // Hide text input area
            this.textInputGroup.style.display = 'none';
            this.characterCount.style.display = 'none';
            this.textInput.disabled = true;
            this.textInput.value = '';
            this.updateSendButton();
        }
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
        
        // Add user message
        this.addUserMessage(text);
        
        // Clear input
        this.textInput.value = '';
        this.handleTextInput('');
        this.autoResizeTextarea();
        
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
            this.addBotMessage('جاري تحليل النص...', 'loading');
        } else {
            this.removeLoadingMessage();
        }
    }
    
    addUserMessage(text) {
        const messageDiv = document.createElement('div');
        messageDiv.className = 'message-bubble user-message';
        messageDiv.innerHTML = `
            <div class="message-content">
                <div class="d-flex align-items-center mb-2">
                    <i class="fas fa-user me-2"></i>
                    <strong>أنت</strong>
                    <span class="model-badge ${this.selectedModel} ms-auto">${this.selectedModel.toUpperCase()}</span>
                </div>
                <div>${this.escapeHtml(text)}</div>
            </div>
        `;
        
        this.messagesArea.appendChild(messageDiv);
        this.scrollToBottom();
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
        const loadingMessage = this.messagesArea.querySelector('.loading');
        if (loadingMessage) {
            loadingMessage.remove();
        }
    }
    
    displayResult(result) {
        const isDepression = result.depression_detected;
        const confidence = result.confidence ? Math.round(result.confidence * 100) : null;
        
        const resultClass = isDepression ? 'depression-detected' : 'no-depression';
        const resultIcon = isDepression ? 'fas fa-exclamation-triangle text-danger' : 'fas fa-check-circle text-success';
        const resultColor = isDepression ? 'danger' : 'success';
        
        let confidenceText = '';
        if (confidence) {
            confidenceText = `<div class="mt-2 small text-muted">
                <i class="fas fa-chart-line me-1"></i>
                مستوى الثقة: ${confidence}%
            </div>`;
        }
        
        const content = `
            <div class="result-bubble ${resultClass}">
                <div class="result-title">
                    <i class="${resultIcon}"></i>
                    <span>${result.result_text}</span>
                    <span class="model-badge ${result.model_name} ms-auto">${result.model_name.toUpperCase()}</span>
                </div>
                <div class="small text-muted">
                    النموذج المستخدم: ${this.getModelDisplayName(result.model_name)}
                </div>
                ${confidenceText}
            </div>
        `;
        
        this.addBotMessage(content);
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
