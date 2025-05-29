"""
Sample data and model configurations for the Arabic Depression Detection system.
This file contains sample Arabic texts for testing and model configuration data.
"""

# Sample Arabic texts for testing (from various sources, ensuring ethical use)
SAMPLE_TEXTS = {
    "depression_samples": [
        "أشعر بالحزن الشديد والاكتئاب منذ فترة طويلة",
        "لا أستطيع النوم جيداً والأرق يسيطر علي",
        "أفقد الاهتمام بكل شيء حولي ولا أجد المتعة في أي نشاط",
        "أشعر بالوحدة والعزلة رغم وجود الناس حولي",
        "الحياة صعبة جداً ولا أجد أي أمل في المستقبل",
        "أعاني من القلق المستمر والخوف من المجهول",
        "لا أستطيع التركيز في العمل أو الدراسة",
        "أشعر بالإرهاق والتعب حتى لو لم أبذل مجهود",
        "كل شيء في حياتي يبدو مظلماً وبلا معنى",
        "أفكر في أشياء سلبية باستمرار ولا أستطيع التوقف"
    ],
    
    "normal_samples": [
        "أشعر بالسعادة والراحة اليوم",
        "استمتعت بوقت رائع مع الأصدقاء",
        "الطقس جميل والشمس مشرقة",
        "أنا محظوظ لوجود عائلة رائعة",
        "العمل يسير بشكل جيد والزملاء متعاونون",
        "أتطلع إلى المستقبل بتفاؤل وأمل",
        "أحب قراءة الكتب ومشاهدة الأفلام الجميلة",
        "الرياضة تساعدني على الشعور بالنشاط والحيوية",
        "أقدر النعم الكثيرة في حياتي",
        "أستمتع بتعلم أشياء جديدة ومفيدة"
    ]
}

# Model configurations and descriptions
MODEL_CONFIGURATIONS = {
    "svm": {
        "name_ar": "نموذج دعم الآلة المتجه (SVM)",
        "name_en": "Support Vector Machine",
        "description_ar": "نموذج تعلم آلة تقليدي يستخدم خوارزمية دعم الآلة المتجه لتصنيف النصوص",
        "description_en": "Traditional machine learning model using Support Vector Machine algorithm",
        "type": "traditional_ml",
        "features": [
            "استخدام TF-IDF لاستخراج المعالم",
            "تصنيف خطي عالي الدقة",
            "سرعة في التنبؤ",
            "قابلية تفسير النتائج"
        ],
        "color": "info",
        "icon": "fas fa-vector-square"
    },
    
    "stacking": {
        "name_ar": "نموذج التكديس (Stacking)",
        "name_en": "Stacking Ensemble",
        "description_ar": "نموذج مجمع يجمع بين عدة نماذج تعلم آلة مختلفة لتحسين الدقة",
        "description_en": "Ensemble model combining multiple machine learning algorithms",
        "type": "ensemble",
        "features": [
            "جمع قوة عدة نماذج",
            "دقة عالية في التصنيف",
            "تقليل الأخطاء",
            "مقاومة الإفراط في التدريب"
        ],
        "color": "success",
        "icon": "fas fa-layer-group"
    },
    
    "lstm": {
        "name_ar": "الشبكة العصبية LSTM",
        "name_en": "Long Short-Term Memory",
        "description_ar": "نموذج تعلم عميق يستخدم الذاكرة طويلة قصيرة المدى لفهم السياق",
        "description_en": "Deep learning model using Long Short-Term Memory networks",
        "type": "deep_learning",
        "features": [
            "فهم السياق والتسلسل",
            "ذاكرة للنصوص الطويلة",
            "تعلم الأنماط المعقدة",
            "معالجة متقدمة للغة"
        ],
        "color": "warning",
        "icon": "fas fa-brain"
    },
    
    "arabicbert": {
        "name_ar": "نموذج المحولات ArabicBERT",
        "name_en": "Arabic BERT Transformer",
        "description_ar": "نموذج محولات مدرب مسبقاً على النصوص العربية باستخدام تقنية BERT",
        "description_en": "Pre-trained transformer model specifically for Arabic text",
        "type": "transformer",
        "features": [
            "فهم عميق للغة العربية",
            "تدريب مسبق على ملايين النصوص",
            "أحدث تقنيات الذكاء الاصطناعي",
            "دقة عالية جداً"
        ],
        "color": "danger",
        "icon": "fas fa-robot"
    }
}

# Preprocessing steps description in Arabic
PREPROCESSING_STEPS = [
    {
        "step": 1,
        "name_ar": "تنظيف أساسي",
        "description_ar": "تحويل النص إلى نص مُنظف أساسي وإزالة المسافات الزائدة"
    },
    {
        "step": 2,
        "name_ar": "إزالة الروابط",
        "description_ar": "إزالة جميع الروابط والـ URLs من النص"
    },
    {
        "step": 3,
        "name_ar": "إزالة الرموز التعبيرية",
        "description_ar": "إزالة الرموز التعبيرية (Emojis) والرموز الخاصة"
    },
    {
        "step": 4,
        "name_ar": "إزالة الأحرف الأجنبية",
        "description_ar": "إزالة الأحرف الإنجليزية والأرقام والاحتفاظ بالأحرف العربية فقط"
    },
    {
        "step": 5,
        "name_ar": "إزالة الحركات",
        "description_ar": "إزالة الحركات العربية (التشكيل) من النص"
    },
    {
        "step": 6,
        "name_ar": "توحيد الأحرف",
        "description_ar": "توحيد أشكال الأحرف العربية المختلفة (مثل الألف والياء)"
    },
    {
        "step": 7,
        "name_ar": "تنظيم المسافات",
        "description_ar": "إزالة المسافات الزائدة وتنظيم المسافات بين الكلمات"
    },
    {
        "step": 8,
        "name_ar": "إزالة علامات الترقيم",
        "description_ar": "إزالة جميع علامات الترقيم العربية والأجنبية"
    },
    {
        "step": 9,
        "name_ar": "تقطيع الكلمات",
        "description_ar": "تقسيم النص إلى كلمات منفردة (Tokenization)"
    },
    {
        "step": 10,
        "name_ar": "إزالة كلمات الوقف",
        "description_ar": "إزالة الكلمات الشائعة التي لا تحمل معنى مفيد"
    },
    {
        "step": 11,
        "name_ar": "إزالة الكلمات القصيرة",
        "description_ar": "إزالة الكلمات التي تحتوي على أقل من حرفين"
    },
    {
        "step": 12,
        "name_ar": "استخلاص الجذور",
        "description_ar": "تحويل الكلمات إلى جذورها الأساسية (Stemming)"
    }
]

# Arabic depression-related keywords for analysis
DEPRESSION_KEYWORDS = [
    # Direct depression terms
    'اكتئاب', 'كآبة', 'حزن', 'يأس', 'قنوط',
    
    # Emotional states
    'حزين', 'يائس', 'منزعج', 'قلق', 'خائف', 'متوتر', 'مضطرب',
    
    # Physical symptoms
    'تعب', 'إرهاق', 'أرق', 'نوم', 'صداع', 'ألم',
    
    # Social symptoms
    'وحدة', 'عزلة', 'انطواء', 'وحش', 'منعزل',
    
    # Cognitive symptoms
    'تركيز', 'ذاكرة', 'تشتت', 'نسيان', 'تفكير',
    
    # Behavioral symptoms
    'فقدان', 'اهتمام', 'رغبة', 'شهية', 'نشاط', 'طاقة',
    
    # Hopelessness
    'أمل', 'مستقبل', 'معنى', 'هدف', 'قيمة',
    
    # Negative thoughts
    'سلبي', 'مظلم', 'صعب', 'مستحيل', 'فاشل'
]

# Positive keywords for contrast
POSITIVE_KEYWORDS = [
    # Happiness
    'سعيد', 'سعادة', 'فرح', 'فرحان', 'مبسوط', 'مسرور',
    
    # Hope and optimism
    'أمل', 'تفاؤل', 'متفائل', 'واثق', 'إيجابي',
    
    # Energy and activity
    'نشاط', 'طاقة', 'حيوية', 'نشيط', 'محتمس',
    
    # Social connection
    'حب', 'صداقة', 'عائلة', 'أصدقاء', 'تواصل',
    
    # Achievement
    'نجح', 'نجاح', 'إنجاز', 'فوز', 'تقدم', 'تحسن',
    
    # Positive descriptions
    'جميل', 'رائع', 'ممتاز', 'مذهل', 'رائع', 'مفيد',
    
    # Gratitude
    'شكر', 'امتنان', 'تقدير', 'محظوظ', 'ممتن'
]

def get_sample_text(category="mixed", count=1):
    """
    Get sample texts for testing
    
    Args:
        category: 'depression', 'normal', or 'mixed'
        count: number of samples to return
    
    Returns:
        List of sample texts
    """
    if category == "depression":
        return SAMPLE_TEXTS["depression_samples"][:count]
    elif category == "normal":
        return SAMPLE_TEXTS["normal_samples"][:count]
    elif category == "mixed":
        depression_half = count // 2
        normal_half = count - depression_half
        return (SAMPLE_TEXTS["depression_samples"][:depression_half] + 
                SAMPLE_TEXTS["normal_samples"][:normal_half])
    else:
        return []

def get_model_info(model_name=None):
    """
    Get model information
    
    Args:
        model_name: specific model name or None for all models
    
    Returns:
        Model information dictionary
    """
    if model_name:
        return MODEL_CONFIGURATIONS.get(model_name, {})
    else:
        return MODEL_CONFIGURATIONS

def get_preprocessing_info():
    """Get preprocessing steps information"""
    return PREPROCESSING_STEPS

def get_keywords():
    """Get depression and positive keywords"""
    return {
        'depression': DEPRESSION_KEYWORDS,
        'positive': POSITIVE_KEYWORDS
    }
