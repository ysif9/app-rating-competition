import re
import emoji # install pip
import spacy # install pip
import nltk
from unidecode import unidecode
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer


def preprocess_app_name(name_series):
    """
    Given a pandas Series of raw app names, returns a fitted sklearn Pipeline
    that transforms names into TF-IDF feature vectors with min_df=1.
    """
    # Ensure required resources are available
    # nltk.download('punkt')
    # nltk.download('stopwords')
    # python -m spacy download en_core_web_sm

    # Load spaCy model without unnecessary components
    nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
    stop_words = set(nltk.corpus.stopwords.words('english'))

    # Cleaning function: demojize, normalize, remove unwanted chars
    def _clean(text):
        text = str(text).lower()
        text = emoji.demojize(text, delimiters=(" emoji_", " "))
        text = unidecode(text)
        text = re.sub(r'[™®©]', '', text)
        text = re.sub(r'[^a-z0-9\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        if text in {"#name?", "nan", "na"}:
            return ""
        return text

    # Custom analyzer: clean -> tokenize -> remove stopwords -> lemmatize
    def _spacy_analyzer(text):
        cleaned = _clean(text)
        tokens = nltk.word_tokenize(cleaned)
        tokens = [t for t in tokens if t and t not in stop_words]
        doc = nlp(" ".join(tokens))
        return [tok.lemma_ for tok in doc if tok.lemma_]

    # Build and fit the pipeline with TF-IDF vectorizer
    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(
            analyzer=_spacy_analyzer,
            lowercase=False,
            token_pattern=None,
            ngram_range=(1, 2),
            max_features=5000,
            min_df=1  # include terms appearing in at least 1 doc
        ))
    ])

    # Fit the pipeline on the provided name series
    pipeline.fit(name_series.astype(str).fillna(""))
    return pipeline