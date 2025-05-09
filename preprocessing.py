import holidays
from sklearn.impute import SimpleImputer
from sklearn.pipeline import FeatureUnion, make_pipeline, Pipeline
from sklearn.preprocessing import PowerTransformer, FunctionTransformer, OrdinalEncoder, OneHotEncoder, StandardScaler

from OutlierCapper import OutlierCapper
from utils import *

us_holidays = holidays.US(years=range(2010, 2019))

'''
# ------------- Finalized Preprocessing -----------------
'''


def category_pipeline():
    return make_pipeline(
        OneHotEncoder(handle_unknown="ignore", sparse_output=True)
    )


def size_pipeline():
    def clean_size(X):
        s = X.iloc[:, 0] \
             .str.replace("M", "e6") \
             .str.replace("K", "e3") \
             .str.replace("k", "e3")
        # compute median from the truly-numeric entries
        num = pd.to_numeric(s[s != "Varies with device"])
        med = num.median()
        # replace “Varies with device” with that median, then cast
        return pd.to_numeric(s.replace("Varies with device", str(med))).to_frame()
    return make_pipeline(
        FunctionTransformer(clean_size, validate=False),
        SimpleImputer(strategy="median"),
        StandardScaler()
    )



def installs_numerical_pipeline():
    return make_pipeline(
        FunctionTransformer(lambda X: X.iloc[:, 0].map(parse_number).to_frame(), ),
        # OutlierCapper(lower_quantile=0.10, upper_quantile=0.90, factor=1.5),
        box_cox_pipeline()
    )

def reviews_numerical_pipeline():
    return make_pipeline(
        FunctionTransformer(lambda X: X.iloc[:, 0].map(parse_number).to_frame(), ),
        # OutlierCapper(lower_quantile=0.10, upper_quantile=0.90, factor=1.5),
        box_cox_pipeline()
    )

def type_pipeline():
    return make_pipeline(
        OneHotEncoder(handle_unknown="ignore", drop="first", sparse_output=False),
        SimpleImputer(strategy="most_frequent")
    )


# ------------------------------
def box_cox_pipeline():
    return make_pipeline(
        PowerTransformer(method="box-cox", standardize=True))


def downloads_pipeline():
    return make_pipeline(
        FunctionTransformer(lambda X: X.iloc[:, 0].map(parse_number).to_frame(), ),
        SimpleImputer(strategy="constant", fill_value=0, add_indicator=True),
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


def installs_pipeline():
    return FeatureUnion([
        ("installs_cat", ordinal_category_pipeline()),
        ("installs_num", installs_numerical_pipeline()),
        ("installs_group", installs_group_pipeline())
    ])


def price_pipeline():
    return make_pipeline(
        FunctionTransformer(lambda X: X.iloc[:, 0].map(parse_number).to_frame(), ),
        log_pipeline()
    )


def age_rating_pipeline():
    return make_pipeline(
        # FunctionTransformer(combine_everyone_age_rating),
        OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    )


def year_group_pipeline():
    return make_pipeline(
        FunctionTransformer(lambda X: pd.DataFrame(X, columns=["last_updated"])["last_updated"]
                            .dt.year
                            .map(group_years)
                            .to_frame(name="year_group"), ),
        category_pipeline()
    )


def holiday_group_pipeline():
    return make_pipeline(
        FunctionTransformer(lambda df: df['last_updated']
                            .map(us_holidays)
                            .fillna('Not Holiday')
                            .to_frame(name="holiday_name"),
                            validate=False),
        FunctionTransformer(group_holidays, validate=False),
        OneHotEncoder(handle_unknown="ignore", sparse_output=True)  # consistent format
    )


def release_date_pipeline():
    return make_pipeline(
        FunctionTransformer(lambda df: df.assign(last_updated=pd.to_datetime(df["last_updated"]))),
        FeatureUnion([
            ("year", FunctionTransformer(lambda df: df["last_updated"].dt.year.to_frame())),
            # ("year_group", year_group_pipeline()),
            # ("holiday_group", holiday_group_pipeline()),
            # ("is_holiday", FunctionTransformer(
            #     lambda df: pd.DataFrame({
            #         "is_holiday": df['last_updated'].dt.date.isin(us_holidays)
            #     })
            # )),

            ("weekday", FunctionTransformer(lambda df: df["last_updated"].dt.weekday.to_frame(name="weekday"))),
        ])
    )


def os_version_pipeline():
    return make_pipeline(
        FunctionTransformer(lambda X: X.iloc[:, 0].map(extract_min_base_os).to_frame(name="min_os")),
        FunctionTransformer(lambda v: v.astype(float))
    )


def log_pipeline():
    return make_pipeline(FunctionTransformer(np.log1p))


def current_ver_pipeline():
    replaces = [u'\u00AE', u'\u2013', u'\u00C3', u'\u00E3', u'\u00B3', '[', ']', "'"]
    regexes = [r'[-+|/:/;(_)@]', r'\s+', r'[A-Za-z]+']

    def _clean_current_ver(X):
        # turn X into a pandas Series of strings
        if isinstance(X, pd.DataFrame):
            s = X.iloc[:, 0].astype(str)
        else:
            # X might be a numpy array of shape (n_samples,1)
            s = pd.Series(X.reshape(-1), dtype=str)

        # 1) strip out unwanted literal characters
        for rep in replaces:
            s = s.str.replace(rep, '', regex=False)

        # 2) replace any of your regex patterns with "0"
        for rx in regexes:
            s = s.str.replace(rx, '0', regex=True)

        # 3) fix decimal placement once
        s = s.map(lambda x: x.replace('.', ',', 1).replace('.', '').replace(',', '.', 1))

        # 4) convert to float
        return s.astype(float).values.reshape(-1, 1)

    return Pipeline([
        ("clean", FunctionTransformer(_clean_current_ver, validate=False)),
        ("impute", SimpleImputer(strategy="median"))
    ])

def app_pipeline():
    """Label-encode the 'App' column to integer IDs."""
    return Pipeline([
        ("ordinal", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1))
    ])


def app_tags_pipeline():
    """Label-encode the 'Genres' column to integer IDs."""
    return Pipeline([
        ("ordinal", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1))
    ])