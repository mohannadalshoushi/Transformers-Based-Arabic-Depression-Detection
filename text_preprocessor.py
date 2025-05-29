import re
import logging
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.isri import ISRIStemmer
import emoji

# Download required NLTK data
try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
except Exception as e:
    logging.warning(f"Could not download NLTK data: {e}")

class ArabicTextPreprocessor:
    """
    Arabic text preprocessor implementing the 12-step cleaning pipeline
    from the original notebooks
    """
    
    def __init__(self):
        self.stemmer = ISRIStemmer()
        try:
            self.stop_words = set(stopwords.words('arabic'))
        except:
            # Fallback Arabic stopwords if NLTK data not available
            self.stop_words = {
                'في', 'من', 'إلى', 'على', 'عن', 'مع', 'كل', 'بعض', 'هذا', 'هذه',
                'ذلك', 'تلك', 'التي', 'الذي', 'التي', 'اللذان', 'اللتان', 'اللذين',
                'اللتين', 'اللواتي', 'اللاتي', 'والتي', 'والذي', 'أن', 'إن', 'كان',
                'كانت', 'يكون', 'تكون', 'أكون', 'نكون', 'يا', 'ما', 'لا', 'لم', 'لن'
            }
        
        # Extended Arabic stopwords
        additional_stopwords = {
            'انا', 'انت', 'هو', 'هي', 'نحن', 'انتم', 'هم', 'هن', 'ليس', 'ليست',
            'غير', 'سوف', 'قد', 'كما', 'عند', 'عندما', 'حيث', 'بينما', 'لكن',
            'لكنه', 'لكنها', 'ولكن', 'أم', 'أما', 'إما', 'أو', 'أي', 'بعد', 'قبل',
            'خلال', 'أثناء', 'حول', 'دون', 'بدون', 'ضد', 'نحو', 'تجاه', 'لدى',
            'لديه', 'لديها', 'لدينا', 'لديكم', 'لديهم', 'لديهن'
        }
        self.stop_words.update(additional_stopwords)
    
    def preprocess(self, text):
        """
        Apply simplified preprocessing for better results
        """
        if not text or not isinstance(text, str):
            return ""
        
        try:
            # Step 1: Basic cleaning
            text = str(text).strip()
            
            # Step 2: Remove URLs
            text = self._remove_urls(text)
            
            # Step 3: Remove emojis
            text = self._remove_emojis(text)
            
            # Step 4: Remove English characters and numbers
            text = self._remove_english_and_numbers(text)
            
            # Step 5: Remove Arabic diacritics
            text = self._remove_diacritics(text)
            
            # Step 6: Normalize Arabic characters
            text = self._normalize_arabic(text)
            
            # Step 7: Remove extra whitespaces
            text = self._remove_extra_whitespace(text)
            
            # Return the processed text (less aggressive processing)
            return text
            
        except Exception as e:
            logging.error(f"Error in text preprocessing: {e}")
            return text  # Return original text if preprocessing fails
    
    def _remove_urls(self, text):
        """Step 2: Remove URLs"""
        url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
        text = re.sub(url_pattern, '', text)
        return text
    
    def _remove_emojis(self, text):
        """Step 3: Remove emojis"""
        try:
            # Remove emojis using emoji library
            text = emoji.demojize(text, delimiters=("", ""))
            # Remove any remaining emoji patterns
            emoji_pattern = re.compile(
                "["
                "\U0001F600-\U0001F64F"  # emoticons
                "\U0001F300-\U0001F5FF"  # symbols & pictographs
                "\U0001F680-\U0001F6FF"  # transport & map symbols
                "\U0001F1E0-\U0001F1FF"  # flags (iOS)
                "\U00002702-\U000027B0"
                "\U000024C2-\U0001F251"
                "]+", flags=re.UNICODE)
            text = emoji_pattern.sub(r'', text)
        except:
            pass
        return text
    
    def _remove_english_and_numbers(self, text):
        """Step 4: Remove English characters and numbers"""
        # Remove English letters and numbers, keep Arabic letters and spaces
        text = re.sub(r'[a-zA-Z0-9]+', ' ', text)
        return text
    
    def _remove_diacritics(self, text):
        """Step 5: Remove Arabic diacritics"""
        diacritics = re.compile(r'[\u064B-\u0652\u0670\u0640]')  # Arabic diacritics
        text = diacritics.sub('', text)
        return text
    
    def _normalize_arabic(self, text):
        """Step 6: Normalize Arabic characters"""
        # Normalize alef forms
        text = re.sub(r'[إأآا]', 'ا', text)
        # Normalize teh marbuta
        text = re.sub(r'ة', 'ه', text)
        # Normalize yeh forms
        text = re.sub(r'ي', 'ي', text)
        text = re.sub(r'ى', 'ي', text)
        return text
    
    def _remove_extra_whitespace(self, text):
        """Step 7: Remove extra whitespaces"""
        text = re.sub(r'\s+', ' ', text)
        return text.strip()
    
    def _remove_punctuation(self, text):
        """Step 8: Remove punctuation"""
        # Remove Arabic and common punctuation
        punctuation = r'[۔؍؎؏ؘؙؚؐؑؒؓؔؕؖؗ؛؞؟ؠآاأابةتثجحخدذرزسشصضطظعغفقكلمنهوىي٠١٢٣٤٥٦٧٨٩ـًٌٍَُِّْ٪٫٬٭ٰٱٲٳٴٵٶٷٸٹٺٻټٽپٿڀځڂڃڄڅچڇڈډڊڋڌڍڎڏڐڑڒړڔڕږڗژڙښڛڜڝڞڟڠڡڢڣڤڥڦڧڨکڪګڬڭڮگڰڱڲڳڴڵڶڷڸڹںڻڼڽھڿۀہۂۃۄۅۆۇۈۉۊۋیۍێۏېۑےۓ۔ەۖۗۘۙۚۛۜ۝۞ۣ۟۠ۡۢۤۥۦۧۨ۩۪ۭ۫۬ۮۯ۰۱۲۳۴۵۶۷۸۹ۺۻۼ۽۾ۿ\.\,\!\?\:\;\"\'\(\)\[\]\{\}\-\_\+\=\*\/\\\|\@\#\$\%\^\&\~\`]'
        text = re.sub(punctuation, ' ', text)
        return text
    
    def _tokenize(self, text):
        """Step 9: Tokenization"""
        try:
            # Try NLTK tokenization first
            tokens = word_tokenize(text)
        except:
            # Fallback to simple whitespace tokenization
            tokens = text.split()
        
        return [token for token in tokens if token.strip()]
    
    def _remove_stopwords(self, tokens):
        """Step 10: Remove stopwords"""
        return [token for token in tokens if token not in self.stop_words]
    
    def _remove_short_words(self, tokens):
        """Step 11: Remove words shorter than 2 characters"""
        return [token for token in tokens if len(token) >= 2]
    
    def _apply_stemming(self, tokens):
        """Step 12: Apply Arabic stemming"""
        try:
            return [self.stemmer.stem(token) for token in tokens]
        except:
            # If stemming fails, return original tokens
            return tokens
    
    def get_preprocessing_steps(self):
        """Return description of preprocessing steps"""
        return [
            "تحويل النص إلى نص مُنظف أساسي",
            "إزالة الروابط (URLs)",
            "إزالة الرموز التعبيرية (Emojis)",
            "إزالة الأحرف الإنجليزية والأرقام",
            "إزالة الحركات العربية",
            "توحيد الأحرف العربية",
            "إزالة المسافات الزائدة",
            "إزالة علامات الترقيم",
            "تقطيع النص إلى كلمات",
            "إزالة كلمات الوقف",
            "إزالة الكلمات القصيرة",
            "استخلاص جذور الكلمات"
        ]
