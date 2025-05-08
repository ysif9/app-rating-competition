import re
import emoji # pip install emoji
import spacy # pip install spacy; python -m spacy download en_core_web_sm
import nltk   # pip install nltk
from nltk.corpus import stopwords
from sklearn.preprocessing import FunctionTransformer
from unidecode import unidecode # pip install unidecode
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer

# Ensure NLTK stopwords are downloaded once (globally or in your main script)
# try:
#     stopwords.words('english')
# except LookupError:
#     nltk.download('stopwords')

# python -m spacy download en_core_web_sm # Run this in your terminal if not done

def preprocess_app_name():
    """
    Given a pandas Series of raw app names, returns a fitted sklearn Pipeline
    that transforms names into TF-IDF feature vectors.
    Optimized based on typical app name characteristics.
    """

    try:
        nlp = spacy.load("en_core_web_sm", disable=["parser", "ner", "tok2vec", "tagger"])
        # make the underlying Thinc vector array writeable
        # nlp.vocab.vectors.data = nlp.vocab.vectors.data.copy()
    except OSError:
        print("Spacy 'en_core_web_sm' model not found. Please run: \npython -m spacy download en_core_web_sm")
        raise

    try:
        nltk_stopwords = set(stopwords.words('english'))
    except LookupError:
        nltk.download('stopwords')
        nltk_stopwords = set(stopwords.words('english'))

    custom_app_stopwords = {
        'app', 'apps',
        'inc', 'llc', 'ltd', 'corp', 'corporation', 'co',
        'android', 'mobile',
        'version', 'edition',
        'my',
        'get', 'your',
        'free', 'pro',
        'llp', 'ft'
    }

    combined_stopwords = nltk_stopwords.union(custom_app_stopwords)


    def _clean_text(text):
        text = str(text)
        text = emoji.demojize(text, delimiters=(" emoji_", "_emoji "))
        text = unidecode(text)
        text = text.lower()
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
    # pipeline = Pipeline([
    #     ("tfidf", TfidfVectorizer(
    #         analyzer=_custom_spacy_analyzer,
    #         lowercase=False,          # Already handled in _clean_text and analyzer
    #         token_pattern=None,       # Analyzer handles tokenization
    #         ngram_range=(1, 2),       # Unigrams and bigrams, very useful for app names
    #         max_features=5000,        # Limits vocabulary size, tune as needed
    #         min_df=1
    #     ))
    # ])
    # return pipeline

    return Pipeline([
        # turn the (n_samples,1) DataFrame into a 1D array/Series
        ("extract_str", FunctionTransformer(lambda X: X.values.ravel(), validate=False)),

        ("tfidf", TfidfVectorizer(
            analyzer="word",  # ← use the built-in word analyzer
            tokenizer=_custom_spacy_analyzer,
            lowercase=False,
            token_pattern=None,
            ngram_range=(1, 2),
            max_features=5000,
            min_df=1
        ))
    ])