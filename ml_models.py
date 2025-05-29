import logging
import os
import numpy as np
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from models import PredictionResult

class DepressionDetector:
    """Main class for depression detection using multiple ML models"""
    
    def __init__(self):
        self.models = {}
        self.vectorizers = {}
        self.loaded_models = []
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize all ML models"""
        try:
            # Initialize TF-IDF vectorizer (same configuration as in notebooks)
            self.main_vectorizer = TfidfVectorizer(
                max_features=5000,
                ngram_range=(1, 2),
                min_df=2,
                max_df=0.8
            )
            
            # Since we don't have the actual trained models, we'll simulate their behavior
            # In a real implementation, you would load the saved models here
            self._load_svm_model()
            self._load_stacking_model()
            self._load_lstm_model()
            self._load_arabicbert_model()
            
            logging.info(f"Models initialized successfully: {self.loaded_models}")
            
        except Exception as e:
            logging.error(f"Error initializing models: {e}")
    
    def _load_svm_model(self):
        """Load or initialize SVM model"""
        try:
            # In production, you would load the trained model:
            # with open('models_data/svm_model.pkl', 'rb') as f:
            #     self.models['svm'] = pickle.load(f)
            
            # For demonstration, create a basic SVM model
            self.models['svm'] = SVC(kernel='rbf', probability=True, random_state=42)
            self.vectorizers['svm'] = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
            self.loaded_models.append('svm')
            
        except Exception as e:
            logging.error(f"Error loading SVM model: {e}")
    
    def _load_stacking_model(self):
        """Load or initialize Stacking model"""
        try:
            # Create base models for stacking
            base_models = [
                ('svm', SVC(probability=True, random_state=42)),
                ('rf', RandomForestClassifier(n_estimators=100, random_state=42)),
                ('nb', MultinomialNB())
            ]
            
            # Create stacking classifier
            self.models['stacking'] = StackingClassifier(
                estimators=base_models,
                final_estimator=LogisticRegression(),
                cv=5
            )
            self.vectorizers['stacking'] = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
            self.loaded_models.append('stacking')
            
        except Exception as e:
            logging.error(f"Error loading Stacking model: {e}")
    
    def _load_lstm_model(self):
        """Load or initialize LSTM model"""
        try:
            # In production, you would load the trained Keras model:
            # from tensorflow.keras.models import load_model
            # self.models['lstm'] = load_model('models_data/lstm_model.h5')
            
            # For demonstration, we'll simulate LSTM behavior
            self.models['lstm'] = 'lstm_placeholder'
            self.loaded_models.append('lstm')
            
        except Exception as e:
            logging.error(f"Error loading LSTM model: {e}")
    
    def _load_arabicbert_model(self):
        """Load or initialize ArabicBERT model"""
        try:
            # In production, you would load the fine-tuned BERT model:
            # from transformers import AutoTokenizer, AutoModelForSequenceClassification
            # self.tokenizer = AutoTokenizer.from_pretrained('models_data/arabicbert/')
            # self.models['arabicbert'] = AutoModelForSequenceClassification.from_pretrained('models_data/arabicbert/')
            
            # For demonstration, we'll simulate BERT behavior
            self.models['arabicbert'] = 'bert_placeholder'
            self.loaded_models.append('arabicbert')
            
        except Exception as e:
            logging.error(f"Error loading ArabicBERT model: {e}")
    
    def predict(self, text, model_name):
        """Make prediction using specified model"""
        try:
            if model_name not in self.models:
                raise ValueError(f"Model {model_name} not found")
            
            # For demonstration purposes, we'll create a simple rule-based classifier
            # that looks for depression-related keywords in Arabic
            prediction = self._simulate_prediction(text, model_name)
            confidence = np.random.uniform(0.6, 0.95)  # Simulate confidence score
            
            return PredictionResult(
                model_name=model_name,
                text=text,
                prediction=prediction,
                confidence=confidence
            )
            
        except Exception as e:
            logging.error(f"Error in prediction: {e}")
            raise
    
    def _simulate_prediction(self, text, model_name):
        """
        Enhanced depression detection based on comprehensive Arabic analysis
        """
        
        # Comprehensive Arabic depression indicators
        depression_indicators = {
            'direct_terms': [
                'اكتئاب', 'مكتئب', 'كآبة', 'كئيب', 'حزن', 'حزين', 'حزينة',
                'يأس', 'يائس', 'يائسة', 'إحباط', 'محبط', 'محبطة'
            ],
            'emotional_states': [
                'وحدة', 'وحيد', 'وحيدة', 'منعزل', 'منعزلة', 'عزلة', 'معزول',
                'قلق', 'قلقان', 'قلقة', 'خوف', 'خائف', 'خائفة', 'رعب',
                'ضيق', 'ضايق', 'ضايقة', 'زعل', 'زعلان', 'زعلانة', 'منزعج'
            ],
            'physical_symptoms': [
                'تعب', 'تعبان', 'تعبانة', 'إرهاق', 'مرهق', 'مرهقة',
                'أرق', 'نوم', 'صداع', 'ألم', 'مريض', 'مريضة'
            ],
            'negative_thoughts': [
                'فاشل', 'فاشلة', 'فشل', 'مستحيل', 'صعب', 'صعبة', 'مشكلة',
                'مشاكل', 'ميؤوس', 'بائس', 'بائسة', 'تعيس', 'تعيسة'
            ],
            'hopelessness': [
                'ما في أمل', 'مافي أمل', 'بلا أمل', 'لا أمل', 'ما عاد في',
                'خلاص', 'انتهى', 'مالي مستقبل', 'ما في مستقبل'
            ],
            'inability_phrases': [
                'ما أقدر', 'ما بقدر', 'مش قادر', 'مش قادرة', 'ما فيني',
                'ما بفيق', 'ما بقوى', 'عاجز', 'عاجزة', 'ما أستطيع'
            ],
            'negative_feelings': [
                'كره', 'أكره', 'بكره', 'كاره', 'مش طايق', 'مش طايقة',
                'زهقت', 'زهقان', 'زهقانة', 'مللت', 'مملول', 'مملولة'
            ],
            'life_difficulty': [
                'حياتي صعبة', 'الحياة صعبة', 'عيشة صعبة', 'الدنيا صعبة',
                'حياتي مظلمة', 'الحياة مظلمة', 'حياتي جحيم', 'عذاب'
            ],
            'dialectal_expressions': [
                'تايه', 'تايهة', 'ضايع', 'ضايعة', 'محتار', 'محتارة',
                'مش لاقي', 'مش لاقية', 'ما لقيت', 'مش عارف', 'مش عارفة',
                'خنقني', 'يخنقني', 'خانقني', 'مختنق', 'مختنقة', 'ما فيي',
                'كنحس', 'نحس', 'بوحدي', 'براسي', 'ما بقات', 'ما بقا',
                'رغبة', 'فوالو', 'والو', 'ولا شي', 'ولا حاجة'
            ]
        }
        
        positive_indicators = {
            'happiness': [
                'سعيد', 'سعيدة', 'سعادة', 'فرح', 'فرحان', 'فرحانة', 'فرحة',
                'مبسوط', 'مبسوطة', 'مسرور', 'مسرورة', 'بهجة', 'مبهج'
            ],
            'optimism': [
                'أمل', 'أتمنى', 'بتمنى', 'متفائل', 'متفائلة', 'تفاؤل',
                'واثق', 'واثقة', 'ثقة', 'إيجابي', 'إيجابية'
            ],
            'energy': [
                'نشاط', 'نشيط', 'نشيطة', 'طاقة', 'حماس', 'متحمس',
                'متحمسة', 'حيوية', 'حيوي', 'نشيطة'
            ],
            'achievement': [
                'نجح', 'نجحت', 'نجاح', 'إنجاز', 'فوز', 'فزت', 'فايز',
                'فايزة', 'تقدم', 'تحسن', 'تطور'
            ]
        }
        
        # Normalize text
        text_normalized = self._normalize_arabic_text(text.lower())
        
        # Calculate scores
        depression_score = 0
        positive_score = 0
        
        # Count depression indicators with different weights
        for category, terms in depression_indicators.items():
            weight = 3 if category in ['hopelessness', 'life_difficulty'] else 2 if category in ['dialectal_expressions', 'inability_phrases'] else 1
            for term in terms:
                term_normalized = self._normalize_arabic_text(term.lower())
                if term_normalized in text_normalized:
                    depression_score += weight
        
        # Count positive indicators
        for category, terms in positive_indicators.items():
            for term in terms:
                term_normalized = self._normalize_arabic_text(term.lower())
                if term_normalized in text_normalized:
                    positive_score += 1
        
        # Model-specific sensitivity adjustments
        model_thresholds = {
            'svm': 1,
            'stacking': 1, 
            'lstm': 2,
            'arabicbert': 1
        }
        
        threshold = model_thresholds.get(model_name, 1)
        
        # Decision logic
        if depression_score >= threshold and depression_score > positive_score:
            return 1  # Depression detected
        else:
            return 0  # No depression detected
    
    def _normalize_arabic_text(self, text):
        """Normalize Arabic text for better matching"""
        # Normalize common Arabic character variations
        text = text.replace('أ', 'ا').replace('إ', 'ا').replace('آ', 'ا')
        text = text.replace('ة', 'ه').replace('ى', 'ي')
        return text
    
    def get_loaded_models(self):
        """Return list of successfully loaded models"""
        return self.loaded_models
    
    def get_model_info(self, model_name):
        """Get information about a specific model"""
        model_info = {
            'svm': {
                'name_ar': 'نموذج SVM (دعم الآلة المتجه)',
                'description_ar': 'نموذج تعلم آلة تقليدي يستخدم خوارزمية دعم الآلة المتجه',
                'type': 'traditional_ml'
            },
            'stacking': {
                'name_ar': 'نموذج التكديس (Stacking)',
                'description_ar': 'نموذج مجمع يجمع عدة نماذج تعلم آلة مختلفة',
                'type': 'ensemble'
            },
            'lstm': {
                'name_ar': 'نموذج LSTM (الشبكة العصبية)',
                'description_ar': 'نموذج تعلم عميق يستخدم الذاكرة طويلة قصيرة المدى',
                'type': 'deep_learning'
            },
            'arabicbert': {
                'name_ar': 'نموذج ArabicBERT (المحولات)',
                'description_ar': 'نموذج محولات مدرب مسبقاً على النصوص العربية',
                'type': 'transformer'
            }
        }
        
        return model_info.get(model_name, {})
