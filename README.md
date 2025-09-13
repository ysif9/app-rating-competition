# App Rating Competition

A machine learning project focused on predicting app ratings using advanced feature engineering, ensemble methods, and natural language processing techniques. This project demonstrates professional-grade data science practices for tabular data competition with complex preprocessing pipelines and model stacking strategies.

## Interesting Techniques

### Advanced Feature Engineering
- **Custom Outlier Capping**: [OutlierCapper.py](OutlierCapper.py) implements a scikit-learn transformer using configurable quantiles and IQR-based limits for robust outlier handling
- **Regex Pattern Matching**: Complex [regular expressions](https://developer.mozilla.org/en-US/docs/Web/JavaScript/Guide/Regular_Expressions) for parsing app sizes, version numbers, and monetary values with unit conversions (MB/KB, M/K suffixes)
- **Holiday Feature Engineering**: [preprocessing.py](preprocessing.py) uses the `holidays` library to create temporal features based on US holiday calendars for release date analysis

### Natural Language Processing
- **Advanced Text Preprocessing**: [preprocess_app_name.py](preprocess_app_name.py) combines emoji demojization, Unicode normalization, and custom stopword filtering
- **spaCy Integration**: Lemmatization and linguistic analysis with disabled unnecessary pipeline components for performance optimization
- **TF-IDF Vectorization**: Custom analyzers with n-gram extraction (unigrams and bigrams) specifically tuned for app naming patterns

### Machine Learning Pipelines
- **Scikit-learn Pipeline Architecture**: Modular preprocessing with [FeatureUnion](https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.FeatureUnion.html) and custom transformers following sklearn's transformer interface
- **Power Transformations**: Box-Cox transformations for handling skewed numerical distributions
- **Multi-target Encoding**: Ordinal and one-hot encoding strategies for categorical variables with unknown value handling

### Data Visualization
- **Distribution Analysis**: [utils.py](utils.py) contains matplotlib-based functions for exploratory data analysis with customizable histogram plotting

## Technologies and Libraries

### Machine Learning Frameworks
- **[CatBoost](https://catboost.ai/)**: Gradient boosting framework optimized for categorical features
- **[LightGBM](https://lightgbm.readthedocs.io/)**: High-performance gradient boosting framework
- **[XGBoost](https://xgboost.readthedocs.io/)**: Scalable gradient boosting system
- **[scikit-learn](https://scikit-learn.org/)**: Core machine learning library for preprocessing and model evaluation

### Natural Language Processing
- **[spaCy](https://spacy.io/)**: Industrial-strength NLP with `en_core_web_sm` model for English text processing
- **[NLTK](https://www.nltk.org/)**: Natural language toolkit for stopwords and text processing utilities
- **[emoji](https://pypi.org/project/emoji/)**: Python library for emoji handling and demojization
- **[unidecode](https://pypi.org/project/unidecode/)**: ASCII transliteration for Unicode text normalization

### Hyperparameter Optimization
- **[Optuna](https://optuna.org/)**: Automatic hyperparameter optimization framework for model tuning

### Data Quality and Analysis
- **[Cleanlab](https://cleanlab.ai/)**: Data-centric AI library for identifying label errors and dataset issues
- **[holidays](https://pypi.org/project/holidays/)**: Python library for generating country-specific holiday calendars

### Data Science Stack
- **[pandas](https://pandas.pydata.org/)**: Data manipulation and analysis
- **[NumPy](https://numpy.org/)**: Numerical computing foundation
- **[matplotlib](https://matplotlib.org/)**: Plotting and visualization
- **[seaborn](https://seaborn.pydata.org/)**: Statistical data visualization
- **[SciPy](https://scipy.org/)**: Scientific computing algorithms
- **[statsmodels](https://www.statsmodels.org/)**: Statistical modeling and econometrics

## Project Structure

```
├── data/
│   ├── app-rating-competition/
│   └── processed datasets
├── submissions/
│   └── philo_submissions/
├── report/
└── notebooks and scripts
```

### Directory Descriptions

- **`data/`**: Contains the original competition dataset and processed/cleaned versions of training data used throughout the pipeline
- **`submissions/`**: Model predictions and competition submissions, with `philo_submissions/` containing team member's individual submission attempts
- **`report/`**: Final documentation including [Team19_Report.pdf](report/Team19_Report.pdf) with detailed methodology and results analysis

## Core Files

- **[preprocessing.py](preprocessing.py)**: Comprehensive data preprocessing pipelines with domain-specific transformers for app metadata
- **[preprocess_app_name.py](preprocess_app_name.py)**: Specialized NLP pipeline for app name feature extraction using advanced text processing techniques  
- **[utils.py](utils.py)**: Utility functions for data parsing, visualization, and statistical analysis including outlier detection methods
- **[OutlierCapper.py](OutlierCapper.py)**: Custom scikit-learn transformer implementing configurable outlier capping using statistical thresholds
- **[pyproject.toml](pyproject.toml)**: Project configuration with comprehensive dependency management for reproducible environments