#!/usr/bin/env python3
"""
Training script for Arabic Depression Detection Models
Based on the provided notebooks to achieve 92+ accuracy
"""

import pandas as pd
import numpy as np
import pickle
import logging
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.ensemble import StackingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import os
from text_preprocessor import ArabicTextPreprocessor

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DepresionModelTrainer:
    def __init__(self):
        self.preprocessor = ArabicTextPreprocessor()
        self.models = {}
        self.vectorizers = {}
        self.results = {}
        
    def load_and_preprocess_data(self):
        """Load and preprocess the dataset"""
        logger.info("Loading dataset...")
        
        # Load dataset
        df = pd.read_excel('attached_assets/combined_dataset.xlsx')
        
        # Keep only text and label columns
        df = df[['text', 'label']]
        
        logger.info(f"Dataset shape: {df.shape}")
        logger.info(f"Label distribution:\n{df['label'].value_counts()}")
        
        # Preprocess texts
        logger.info("Preprocessing texts...")
        processed_texts = []
        
        for idx, text in enumerate(df['text']):
            if idx % 1000 == 0:
                logger.info(f"Processed {idx}/{len(df)} texts")
            
            # Use the full preprocessing pipeline from notebooks
            processed_text = self.preprocessor.preprocess(text)
            
            # If preprocessing results in empty text, use original
            if not processed_text.strip():
                processed_text = str(text).strip()
            
            processed_texts.append(processed_text)
        
        df['processed_text'] = processed_texts
        
        # Remove rows with empty processed text
        df = df[df['processed_text'].str.strip() != ''].reset_index(drop=True)
        
        logger.info(f"Final dataset shape after preprocessing: {df.shape}")
        
        return df
    
    def prepare_features(self, df):
        """Prepare TF-IDF features"""
        logger.info("Creating TF-IDF features...")
        
        # TF-IDF Vectorizer with parameters from notebooks
        self.vectorizer = TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.8,
            stop_words=None  # We already removed stopwords in preprocessing
        )
        
        # Fit and transform the processed texts
        X = self.vectorizer.fit_transform(df['processed_text'])
        y = df['label'].values
        
        logger.info(f"Feature matrix shape: {X.shape}")
        
        return X, y
    
    def train_svm_model(self, X_train, X_test, y_train, y_test):
        """Train SVM model"""
        logger.info("Training SVM model...")
        
        # SVM with parameters from notebook
        svm_model = SVC(
            kernel='rbf',
            C=1.0,
            gamma='scale',
            probability=True,
            random_state=42
        )
        
        svm_model.fit(X_train, y_train)
        
        # Predictions
        y_pred = svm_model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        logger.info(f"SVM Accuracy: {accuracy:.4f}")
        
        self.models['svm'] = svm_model
        self.results['svm'] = {
            'accuracy': accuracy,
            'classification_report': classification_report(y_test, y_pred),
            'predictions': y_pred
        }
        
        return svm_model, accuracy
    
    def train_stacking_model(self, X_train, X_test, y_train, y_test):
        """Train Stacking ensemble model"""
        logger.info("Training Stacking model...")
        
        # Base models for stacking
        base_models = [
            ('svm', SVC(probability=True, random_state=42)),
            ('rf', RandomForestClassifier(n_estimators=100, random_state=42)),
            ('nb', MultinomialNB())
        ]
        
        # Stacking classifier
        stacking_model = StackingClassifier(
            estimators=base_models,
            final_estimator=LogisticRegression(random_state=42),
            cv=5,
            stack_method='predict_proba'
        )
        
        stacking_model.fit(X_train, y_train)
        
        # Predictions
        y_pred = stacking_model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        logger.info(f"Stacking Accuracy: {accuracy:.4f}")
        
        self.models['stacking'] = stacking_model
        self.results['stacking'] = {
            'accuracy': accuracy,
            'classification_report': classification_report(y_test, y_pred),
            'predictions': y_pred
        }
        
        return stacking_model, accuracy
    
    def save_models(self):
        """Save trained models and vectorizer"""
        os.makedirs('models_data', exist_ok=True)
        
        # Save vectorizer
        with open('models_data/tfidf_vectorizer.pkl', 'wb') as f:
            pickle.dump(self.vectorizer, f)
        
        # Save models
        for model_name, model in self.models.items():
            with open(f'models_data/{model_name}_model.pkl', 'wb') as f:
                pickle.dump(model, f)
        
        # Save results
        with open('models_data/training_results.pkl', 'wb') as f:
            pickle.dump(self.results, f)
        
        logger.info("Models saved successfully!")
    
    def train_all_models(self):
        """Train all models"""
        logger.info("Starting model training...")
        
        # Load and preprocess data
        df = self.load_and_preprocess_data()
        
        # Prepare features
        X, y = self.prepare_features(df)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        logger.info(f"Training set size: {X_train.shape[0]}")
        logger.info(f"Test set size: {X_test.shape[0]}")
        
        # Train models
        svm_model, svm_acc = self.train_svm_model(X_train, X_test, y_train, y_test)
        stacking_model, stacking_acc = self.train_stacking_model(X_train, X_test, y_train, y_test)
        
        # Save models
        self.save_models()
        
        logger.info("\n" + "="*50)
        logger.info("TRAINING COMPLETED")
        logger.info("="*50)
        logger.info(f"SVM Accuracy: {svm_acc:.4f}")
        logger.info(f"Stacking Accuracy: {stacking_acc:.4f}")
        logger.info("="*50)
        
        return {
            'svm_accuracy': svm_acc,
            'stacking_accuracy': stacking_acc
        }

if __name__ == "__main__":
    trainer = DepresionModelTrainer()
    results = trainer.train_all_models()