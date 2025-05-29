"""
Data models for the Arabic Depression Detection application.
This file contains any data structures needed for the application.
"""

class PredictionResult:
    """Class to represent prediction results from ML models"""
    
    def __init__(self, model_name, text, prediction, confidence=None):
        self.model_name = model_name
        self.text = text
        self.prediction = prediction  # 0 or 1
        self.confidence = confidence
        
    def is_depression_detected(self):
        """Returns True if depression is detected (prediction == 1)"""
        return self.prediction == 1
    
    def get_result_text(self):
        """Returns Arabic text describing the result"""
        if self.is_depression_detected():
            return "تم اكتشاف الاكتئاب"
        else:
            return "لم يتم اكتشاف الاكتئاب"
    
    def to_dict(self):
        """Convert to dictionary for JSON response"""
        return {
            'model_name': self.model_name,
            'text': self.text,
            'prediction': self.prediction,
            'confidence': self.confidence,
            'result_text': self.get_result_text(),
            'depression_detected': self.is_depression_detected()
        }
