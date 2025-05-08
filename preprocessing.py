import holidays
from sklearn.impute import SimpleImputer
from sklearn.pipeline import FeatureUnion, make_pipeline
from sklearn.preprocessing import PowerTransformer, FunctionTransformer, OrdinalEncoder, OneHotEncoder, StandardScaler

from utils import *

us_holidays = holidays.US(years=range(2010, 2019))

'''
# ------------- Finalized Preprocessing -----------------
'''


def app_name_pipeline():
    pass


def category_pipeline():
    return make_pipeline(
        OneHotEncoder(handle_unknown="ignore", sparse_output=True)
    )


def size_pipeline():
    return make_pipeline(
        FunctionTransformer(parse_size_array, validate=False),
        SimpleImputer(strategy="median"),
        StandardScaler(),  # optional
    )


def reviews_numerical_pipeline():
    return make_pipeline(
        FunctionTransformer(lambda X: X.iloc[:, 0].map(parse_number).to_frame(), ),
        box_cox_pipeline()
    )


def installs_pipeline():
    return make_pipeline(
        FunctionTransformer(lambda X: X.iloc[:, 0].map(parse_number).to_frame(), ),
        log_pipeline()
    )


def drop_na(X):
    return X.dropna()

def type_pipeline():
    return make_pipeline(
        OrdinalEncoder(),
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
        OrdinalEncoder()
    )


def review_group_pipeline():
    return make_pipeline(
        FunctionTransformer(lambda X: X.iloc[:, 0]
                            .map(group_reviews_count)
                            .to_frame(),
                            validate=False),
        category_pipeline()
    )


def reviews_pipeline():
    return FeatureUnion([
        ("reviews_cat", ordinal_category_pipeline()),
        ("reviews_num", reviews_numerical_pipeline()),
        ("reviews_group", review_group_pipeline())
    ])


def price_pipeline():
    return make_pipeline(
        FunctionTransformer(lambda X: X.iloc[:, 0].map(parse_number).to_frame(), ),
        log_pipeline()
    )


def age_rating_pipeline():
    return make_pipeline(
        FunctionTransformer(combine_everyone_age_rating),
        category_pipeline()
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
            ("year_group", year_group_pipeline()),
            ("holiday_group", holiday_group_pipeline()),
            ("is_holiday", FunctionTransformer(
                lambda df: pd.DataFrame({
                    "is_holiday": df['last_updated'].dt.date.isin(us_holidays)
                })
            )),

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
