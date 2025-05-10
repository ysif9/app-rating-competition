import pandas as pd
import numpy as np
import re

import holidays
from sklearn.impute import SimpleImputer
from sklearn.pipeline import FeatureUnion, make_pipeline, Pipeline
from sklearn.preprocessing import PowerTransformer, FunctionTransformer, OrdinalEncoder, OneHotEncoder, StandardScaler
from utils import *


# Assume OutlierCapper and utils (parse_number, group_installs_count, group_years, group_holidays, extract_min_base_os)
# are defined elsewhere as per the original context.
# For the purpose of this example, I'll add placeholders for these if they are directly used in modified code.

# Placeholder for OutlierCapper if it were needed in the modified section
class OutlierCapper:
    def __init__(self, lower_quantile=0.10, upper_quantile=0.90, factor=1.5):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X

us_holidays = holidays.US(years=range(2010, 2019))


# ------------- New Size Cleaning Functions (as provided) -----------------
def parse_size_to_mb(size_str: str) -> float:
    if pd.isna(size_str):
        return np.nan

    s = str(size_str).strip()  # Ensure string conversion for safety
    if s.lower() == 'varies with device':
        return np.nan

    m = re.match(r'^([\d\.]+)\s*([MmKk])$', s)
    if not m:
        # Try to interpret as a raw number (assuming it's already in MB if no unit)
        try:
            # This case might not be ideal if raw numbers are in bytes or KB,
            # but following the provided function structure.
            return float(s)
        except ValueError:
            return np.nan

    num, unit_char = m.groups()  # Renamed 'unit' to 'unit_char' to avoid conflict if 'unit' is a global
    num = float(num)
    unit_char = unit_char.upper()

    if unit_char == 'M':
        return num
    elif unit_char == 'K':
        return num / 1024.0  # Convert KB to MB

    return np.nan  # Should not be reached if regex matches 'M' or 'K'


def standardize_sizes(series: pd.Series) -> pd.Series:
    return series.apply(parse_size_to_mb)


'''
# ------------- Finalized Preprocessing -----------------
'''


def category_pipeline():
    return make_pipeline(
        OneHotEncoder(handle_unknown="ignore", sparse_output=True)
    )


def size_pipeline():
    # MODIFIED: Uses new standardize_sizes function
    # "Varies with device" handled by parse_size_to_mb (returns np.nan)
    # Imputation of np.nan happens via SimpleImputer
    return make_pipeline(
        FunctionTransformer(lambda X: standardize_sizes(X.iloc[:, 0]).to_frame(), validate=False),
        SimpleImputer(strategy="median"),  # Imputes NaNs including those from "Varies with device"
        StandardScaler()
    )


def installs_numerical_pipeline():  # Assumed this is for a generic 'Installs' column if different from 'downloads'
    return make_pipeline(
        # Uses parse_number from utils.py, which handles '+', ',', and converts to int or np.nan
        FunctionTransformer(lambda X: X.iloc[:, 0].map(parse_number).to_frame(), validate=False),
        # OutlierCapper(lower_quantile=0.10, upper_quantile=0.90, factor=1.5),
        box_cox_pipeline()
    )


def reviews_numerical_pipeline():
    # MODIFIED: Ensures parse_number from utils.py is used.
    # utils.parse_number returns int or np.nan. This meets "type int" conceptually for reviews.
    # Box-Cox will handle these numeric (potentially float after imputation) values.
    return make_pipeline(
        FunctionTransformer(lambda X: X.iloc[:, 0].map(parse_number).to_frame(), validate=False),
        SimpleImputer(strategy="median"),  # Added to handle potential NaNs before Box-Cox
        # OutlierCapper(lower_quantile=0.10, upper_quantile=0.90, factor=1.5),
        box_cox_pipeline()
    )


def type_pipeline():
    return make_pipeline(
        OneHotEncoder(handle_unknown="ignore", drop="first", sparse_output=False),
        SimpleImputer(strategy="most_frequent")  # Note: Imputing after OHE might be unusual. Usually impute before.
    )


# ------------------------------
def box_cox_pipeline():
    return make_pipeline(
        SimpleImputer(strategy="median"),  # Ensure no NaNs before PowerTransformer
        PowerTransformer(method="box-cox", standardize=True))


def downloads_pipeline():  # Assumed this is for the 'downloads' column from 'cols_to_clean'
    # MODIFIED: Uses utils.parse_number for robust cleaning and int conversion.
    # utils.parse_number handles '+', ',', '$' (as per remove_items) and 'M'/'K', returns int/np.nan.
    def clean_and_convert_downloads(X_series):
        # Use utils.parse_number which is robust and meets criteria
        cleaned_series = X_series.map(parse_number)
        return cleaned_series.to_frame()

    return make_pipeline(
        FunctionTransformer(lambda X: clean_and_convert_downloads(X.iloc[:, 0]), validate=False),
        SimpleImputer(strategy="constant", fill_value=0),  # Fills NaNs from parse_number
        FunctionTransformer(lambda X_df: X_df.astype(int), validate=False),  # Ensure int type after imputation
        log_pipeline(),
    )


def ordinal_category_pipeline():
    return make_pipeline(
        OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    )


def installs_group_pipeline():
    return make_pipeline(
        FunctionTransformer(lambda X: X.iloc[:, 0]
                            .map(group_installs_count)
                            .to_frame(),
                            validate=False),
        category_pipeline()
    )


def installs_pipeline():  # This seems to be a comprehensive pipeline for an 'Installs' column
    return FeatureUnion([
        ("installs_cat", ordinal_category_pipeline()),
        ("installs_num", installs_numerical_pipeline()),  # Uses parse_number for its numeric part
        ("installs_group", installs_group_pipeline())
    ])


def price_pipeline():  # Assumed this is for 'price_if_paid' column from 'cols_to_clean'
    # MODIFIED: Uses specified string replacements and ensures float type.
    # utils.parse_number would convert "1.99" to 1 (int), which is not suitable for price.
    def clean_price_col(X_series):
        s = X_series.astype(str)
        # Applying relevant parts of remove_items = ['+', ',', '$'] for price
        s = s.str.replace('$', '', regex=False)
        s = s.str.replace(',', '', regex=False)
        # '+' is generally not in price strings, so not explicitly removing.
        # Convert to numeric, coercing errors, then ensure float type
        return pd.to_numeric(s, errors='coerce').astype(float)

    return make_pipeline(
        FunctionTransformer(lambda X: clean_price_col(X.iloc[:, 0]).to_frame(), validate=False),
        SimpleImputer(strategy="median"),  # Impute NaNs that may arise from cleaning
        log_pipeline()  # log_pipeline often follows price transformation
    )


def age_rating_pipeline():
    return make_pipeline(
        # FunctionTransformer(combine_everyone_age_rating), # Assuming combine_everyone_age_rating is in utils
        OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    )


def year_group_pipeline():
    return make_pipeline(
        FunctionTransformer(lambda X: pd.DataFrame(X, columns=["last_updated"])["last_updated"]
                            .dt.year
                            .map(group_years)  # Assuming group_years is in utils
                            .to_frame(name="year_group"), ),
        category_pipeline()
    )


def holiday_group_pipeline():
    return make_pipeline(
        FunctionTransformer(lambda df_input: df_input['last_updated']  # df_input instead of df to avoid confusion
                            .map(us_holidays)
                            .fillna('Not Holiday')
                            .to_frame(name="holiday_name"),
                            validate=False),
        FunctionTransformer(group_holidays, validate=False),  # Assuming group_holidays is in utils
        OneHotEncoder(handle_unknown="ignore", sparse_output=True)
    )


def release_date_pipeline():
    return make_pipeline(
        FunctionTransformer(
            lambda df_input: df_input.assign(last_updated=pd.to_datetime(df_input["last_updated"], errors='coerce'))),
        FeatureUnion([
            ("last_updated_year",
             FunctionTransformer(lambda df_input: df_input["last_updated"].dt.year.to_frame(name="last_updated_year"))),
            ("last_updated_month",
             FunctionTransformer(
                 lambda df_input: df_input["last_updated"].dt.month.to_frame(name="last_updated_month"))),
            ("last_updated_day",
             FunctionTransformer(lambda df_input: df_input["last_updated"].dt.day.to_frame(name="last_updated_day"))),
            ("weekday",
             FunctionTransformer(lambda df_input: df_input["last_updated"].dt.weekday.to_frame(name="weekday"))),
        ])
    )


def os_version_pipeline():
    return make_pipeline(
        FunctionTransformer(lambda X: X.iloc[:, 0].map(extract_min_base_os).to_frame(name="min_os")),
        # Assuming extract_min_base_os is in utils
        FunctionTransformer(lambda v: v.astype(float))
    )


def log_pipeline():
    # Added imputer here for robustness before np.log1p, assumes 0 is a safe fill for log(1+x)
    return make_pipeline(
        SimpleImputer(strategy='constant', fill_value=0),
        FunctionTransformer(np.log1p, validate=True)  # np.log1p handles 0 correctly (log1p(0) = 0)
    )


def current_ver_pipeline():
    replaces = [u'\u00AE', u'\u2013', u'\u00C3', u'\u00E3', u'\u00B3', '[', ']', "'"]
    regexes = [r'[-+|/:/;(_)@]', r'\s+', r'[A-Za-z]+']

    def _clean_current_ver(X_input):
        if isinstance(X_input, pd.DataFrame):
            s = X_input.iloc[:, 0].astype(str)
        else:
            s = pd.Series(X_input.reshape(-1), dtype=str)

        for rep in replaces:
            s = s.str.replace(rep, '', regex=False)
        for rx in regexes:
            s = s.str.replace(rx, '0', regex=True)

        s = s.map(lambda x: x.replace('.', ',', 1).replace('.', '').replace(',', '.', 1) if isinstance(x, str) else x)
        s = pd.to_numeric(s, errors='coerce')
        return s.values.reshape(-1, 1)

    return Pipeline([
        ("clean", FunctionTransformer(_clean_current_ver, validate=False)),
        ("impute", SimpleImputer(strategy="median"))
    ])

def app_pipeline():
    return Pipeline([
        ("ordinal", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1))
    ])


def app_tags_pipeline():
    return Pipeline([
        ("ordinal", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1))
    ])