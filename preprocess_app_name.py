import re
import emoji # pip install emoji
import spacy # pip install spacy; python -m spacy download en_core_web_sm
import nltk   # pip install nltk
from nltk.corpus import stopwords
from unidecode import unidecode # pip install unidecode
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer

# Ensure NLTK stopwords are downloaded once (globally or in your main script)
# try:
#     stopwords.words('english')
# except LookupError:
#     nltk.download('stopwords')

# python -m spacy download en_core_web_sm # Run this in your terminal if not done

def preprocess_app_name_optimized(name_series):
    """
    Given a pandas Series of raw app names, returns a fitted sklearn Pipeline
    that transforms names into TF-IDF feature vectors.
    Optimized based on typical app name characteristics.
    """

    try:
        nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
    except OSError:
        print("Spacy 'en_core_web_sm' model not found. Please run: \npython -m spacy download en_core_web_sm")
        raise

    try:
        nltk_stopwords = set(stopwords.words('english'))
    except LookupError:
        nltk.download('stopwords')
        nltk_stopwords = set(stopwords.words('english'))

    custom_app_stopwords = {
        'app', 'apps', 'free', 'pro', 'lite', 'new', 'hd', 'plus', 'mobile', 'android',
        'editor', 'maker', 'game', 'games', 'theme', 'themes', 'launcher', 'keyboard',
        'wallpaper', 'wallpapers', 'live', 'photo', 'photos', 'video', 'videos', 'music',
        'downloader', 'scanner', 'manager', 'creator', 'player', 'browser', 'calculator',
        'calendar', 'weather', 'chat', 'call', 'calls', 'text', 'sms', 'online', 'offline',
        'guide', 'viewer', 'reader', 'best', 'top', 'tools', 'official', 'beta', 'corp',
        'inc', 'llc', 'ltd', 'com', 'studio', 'master', 'edition', 'for', 'and', 'with',
        'the', 'a', 'an', 'my', 'you', 'your', 'all', 'get', 'to', 'by', 'of', 'in', 'on',
        'it', 'is', 'at', 'me', 'co', 'corp', 'llp', 'ft', 'dj', 'ai', 'ar', 'vr', 'tv',
        'go', 'share', 'learn', 'fast', 'easy', 'simple', 'smart', 'ultimate', 'dark',
        'pink', 'gold', 'red', 'blue', 'green', 'white', 'black', 'effect', 'effects',
        'background', 'backgrounds', 'cool', 'cute', 'emoji', 'gifs', 'selfie', 'camera',
        'real', 'data', 'os', 'test', 'gps', 'vpn', 'tv', 'food', 'recipes', 'diet', 'health',
        'fitness', 'workout', 'dating', 'love', 'single', 'singles', 'social', 'network',
        'shopping', 'shop', 'deals', 'coupons', 'delivery', 'services', 'news', 'jobs',
        'book', 'books', 'world', 'local', 'diy', 'tips', 'stories', 'widget', 'alert',
        'alerts', 'tracker', 'utility', 'status', 'remote', 'control', 'cleaner', 'booster',
        'lock', 'private', 'secure', 'security', 'speed', 'web', 'wifi', 'blocker',
        'number', 'v', 'x', 'z', 'c', 'e', 'k', 'r', 's', 't', 'w', 'b', 'd', 'f', 'g', 'h',
        'j', 'l', 'm', 'n', 'p', 'q', 'u', 'y',
        'promo', 'buy', 'sell', 'show', 'full'
    }

    combined_stopwords = nltk_stopwords.union(custom_app_stopwords)


    def _clean_text(text):
        text = str(text)
        text = text.lower()
        text = emoji.demojize(text, delimiters=(" emoji_", "_emoji "))
        text = unidecode(text)
        text = re.sub(r'[™®©]', '', text)
        text = re.sub(r'&', ' and ', text)
        text = re.sub(r'\+', ' plus ', text)

        #General punctuation and symbol removal. Keep alphanumeric, spaces, and underscores (for emojis)
        # Allows numbers like "2018", "4k", "3d"
        text = re.sub(r'[^a-z0-9\s_]', ' ', text)

        text = re.sub(r'\s+', ' ', text).strip()
        if text in {"#name?", "nan", "na", ""}:
            return ""
        return text

    def _custom_spacy_analyzer(text):
        cleaned_text = _clean_text(text)
        if not cleaned_text:
            return []

        doc = nlp(cleaned_text)
        lemmas = [
            token.lemma_
            for token in doc
            if token.lemma_ not in combined_stopwords and \
               not token.is_punct and \
               not token.is_space and \
               len(token.lemma_.strip()) > 0 and \
               (len(token.lemma_.strip()) > 1 or token.lemma_.isdigit())
        ]
        return lemmas

    # Build and fit the pipeline with TF-IDF vectorizer
    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(
            analyzer=_custom_spacy_analyzer,
            lowercase=False,          # Already handled in _clean_text and analyzer
            token_pattern=None,       # Analyzer handles tokenization
            ngram_range=(1, 2),       # Unigrams and bigrams, very useful for app names
            max_features=5000,        # Limits vocabulary size, tune as needed
            min_df=2
        ))
    ])

    pipeline.fit(name_series.astype(str).fillna(""))
    return pipeline