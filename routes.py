import logging
from flask import render_template, request, jsonify
from app import app
from ml_models import DepressionDetector
from text_preprocessor import ArabicTextPreprocessor

# Initialize the ML models and preprocessor
detector = DepressionDetector()
preprocessor = ArabicTextPreprocessor()

@app.route('/')
def index():
    """Main page with the ChatGPT-like interface"""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Handle prediction requests"""
    try:
        # Get data from request
        data = request.get_json()
        text = data.get('text', '').strip()
        model_name = data.get('model', '')
        
        # Validate input
        if not text:
            return jsonify({
                'error': True,
                'message': 'يرجى إدخال النص المراد تحليله'
            }), 400
        
        if not model_name:
            return jsonify({
                'error': True,
                'message': 'يرجى اختيار النموذج أولاً'
            }), 400
        
        # Validate model name
        valid_models = ['svm', 'stacking', 'lstm', 'arabicbert']
        if model_name not in valid_models:
            return jsonify({
                'error': True,
                'message': 'النموذج المختار غير صحيح'
            }), 400
        
        # Validate text length (minimum and maximum)
        if len(text) < 3:
            return jsonify({
                'error': True,
                'message': 'النص قصير جداً. يرجى إدخال نص أطول'
            }), 400
        
        if len(text) > 1000:
            return jsonify({
                'error': True,
                'message': 'النص طويل جداً. يرجى إدخال نص أقصر من 1000 حرف'
            }), 400
        
        # Preprocess the text
        processed_text = preprocessor.preprocess(text)
        
        if not processed_text or len(processed_text.strip()) == 0:
            return jsonify({
                'error': True,
                'message': 'النص لا يحتوي على محتوى قابل للتحليل بعد المعالجة'
            }), 400
        
        # Make prediction
        result = detector.predict(processed_text, model_name)
        
        # Log the prediction for debugging
        app.logger.info(f"Prediction made - Model: {model_name}, Text: {text[:50]}..., Result: {result.get_result_text()}")
        
        return jsonify({
            'error': False,
            'result': result.to_dict()
        })
    
    except Exception as e:
        app.logger.error(f"Error in prediction: {str(e)}")
        return jsonify({
            'error': True,
            'message': 'حدث خطأ في النظام. يرجى المحاولة مرة أخرى'
        }), 500

@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'models_loaded': detector.get_loaded_models()
    })

@app.errorhandler(404)
def not_found(error):
    return render_template('index.html'), 404

@app.errorhandler(500)
def internal_error(error):
    app.logger.error(f"Internal server error: {error}")
    return jsonify({
        'error': True,
        'message': 'حدث خطأ في الخادم'
    }), 500
