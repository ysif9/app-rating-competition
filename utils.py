import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import re

def plot_numeric_distribution(df, column_name, bins=50, color='skyblue'):
    plt.figure(figsize=(12, 6))
    plt.hist(df[column_name].dropna(), bins=bins, color=color, edgecolor='black')
    plt.title(f'Distribution of {column_name}', fontsize=14, pad=15)
    plt.xlabel(column_name, fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def parse_number(value):
    """
    Convert strings like '$1.5M', '$600K', '$100,000+' into integers.
    Strips '$', ',', and '+' before processing.
    """
    if isinstance(value, str):
        value = value.strip().upper().replace("$", "").replace(",", "").replace("+", "")
        try:
            if value.endswith("M"):
                return int(float(value[:-1]) * 1_000_000)
            elif value.endswith("K"):
                return int(float(value[:-1]) * 1_000)
            elif value.replace('.', '', 1).isdigit():
                return int(float(value))
        except ValueError:
            return np.nan
    elif isinstance(value, (int, float)):
        return int(value)
    return np.nan

def parse_size_array(X):
    # coerce whatever you got (DataFrame or array-like) into a NumPy array
    flat = np.asarray(X).ravel()

    out = []
    for x in flat:
        if pd.isna(x) or "varies" in str(x).lower():
            out.append(np.nan)
            continue

        s = str(x).replace(",", "").replace("+", "").strip()
        m = re.match(r"([\d\.]+)([MK]?)$", s, re.I)
        if not m:
            out.append(np.nan)
            continue

        num, unit = m.groups()
        val = float(num)
        if unit.upper() == "M":
            val *= 1024  # convert MB → KB
        # K or blank stays as KB
        out.append(val)

    return np.array(out).reshape(-1, 1)

def group_installs_count(val):
    very_low = ["1+", "5+", "10+", "50+", "100+"]
    low_mid = ["500+", "1,000+", "5,000+", "10,000+", "50,000+"]
    mid = ["100,000+", "500,000+"]
    high = ["1,000,000+", "5,000,000+", "10,000,000+"]
    top = ["50,000,000+", "100,000,000+", "500,000,000+", "1,000,000,000+"]

    if val in very_low:
        return "Very Low"
    elif val in low_mid:
        return "Low-Mid"
    elif val in mid:
        return "Mid"
    elif val in high:
        return "High"
    elif val in top:
        return "Top Tier"
    else:
        return "Other"

def group_holidays(df):
    high = {
        "Independence Day", "Veterans Day", "Thanksgiving Day",
        "Memorial Day", "Christmas Day (observed)"
    }
    mid = {
        "Not Holiday", "Columbus Day", "Martin Luther King Jr. Day",
        "Veterans Day (observed)", "New Year's Day", "Labor Day"
    }
    low = {
        "Christmas Day", "Washington's Birthday"
    }

    def map_group(holiday):
        if pd.isna(holiday):
            return "Other"
        elif holiday in high:
            return "High Rating Holiday"
        elif holiday in mid:
            return "Mid Rating Holiday"
        elif holiday in low:
            return "Low Rating Holiday"
        else:
            return "Other"

    return df.apply(lambda col: col.map(map_group))

def group_years(year):
    if year <= 2012:
        return 'Old'
    elif 2013 <= year <= 2016:
        return 'Middle'
    else:
        return 'Recent'

def combine_everyone_age_rating(X):
    X = X.copy()
    if not isinstance(X, pd.DataFrame):
        X = pd.DataFrame(X, columns=["age_rating"])

    X["age_rating"] = X["age_rating"].replace({"Everyone 10+": "Everyone"})
    return X

def extract_min_base_os(value):
    value = str(value).upper().strip()

    # Match standard version patterns: '5.0 and up', '5.0', '5.0 - 6.0', etc.
    match = re.search(r'(\d+\.\d+)', value)
    if match:
        return float(match.group(1))

    # Special case: Wear OS like '4.4W and up'
    match_wear = re.search(r'(\d+\.\d+)W', value)
    if match_wear:
        return float(match_wear.group(1))

    return 0.0  # Return 0.0 instead of string if a format doesn't match


def is_wear_os(cleaned_value):
    return 'W' in cleaned_value


def is_version_range(cleaned_value):
    return '-' in cleaned_value

def outlier_thresholds(dataframe, variable):
    quartile1 = dataframe[variable].quantile(0.10)
    quartile3 = dataframe[variable].quantile(0.90)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit

def has_outliers(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    if dataframe[(dataframe[variable] < low_limit) | (dataframe[variable] > up_limit)].any(axis=None):
        print(variable, "yes")
    print(variable, "no")

def replace_with_thresholds(df, numeric_columns):
    for variable in numeric_columns:
        low_limit, up_limit = outlier_thresholds(df, variable)
        df.loc[(df[variable] < low_limit), variable] = low_limit
        df.loc[(df[variable] > up_limit), variable] = up_limit