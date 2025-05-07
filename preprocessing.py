from sklearn.compose import ColumnTransformer
from sklearn.pipeline import FeatureUnion, make_pipeline
from sklearn.preprocessing import PowerTransformer, FunctionTransformer, OrdinalEncoder, OneHotEncoder
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from utils import *
import holidays


us_holidays = holidays.US(years=range(2010, 2019))

def box_cox_pipeline():
    return make_pipeline(
        PowerTransformer(method="box-cox", standardize=True))


def category_pipeline():
    return make_pipeline(
        OneHotEncoder(handle_unknown="ignore")
    )



def downloads_pipeline():
    return make_pipeline(
        FunctionTransformer(lambda X: X.iloc[:, 0].map(parse_number).to_frame(), ),
        IterativeImputer(
            missing_values=np.nan,
            add_indicator=True,
            random_state=42,
        ),
        log_pipeline(),
    )


def ordinal_category_pipeline():
    return make_pipeline(
        OrdinalEncoder()
    )


def reviews_numerical_pipeline():
    return make_pipeline(
        FunctionTransformer(lambda X: X.iloc[:, 0].map(parse_number).to_frame(), ),
        box_cox_pipeline()
    )


def review_group_pipeline():
    return make_pipeline(
        FunctionTransformer(lambda X: X.iloc[:, 0]
                            .map(group_reviews_count)  # apply per value
                            .to_frame(), ),
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
        FunctionTransformer(lambda X: pd.DataFrame(X, columns=["release_date"])["release_date"]
                          .dt.year
                          .map(group_years)
                          .to_frame(name="year_group"),),
        category_pipeline()
    )


def holiday_group_pipeline():
    return make_pipeline(
        # Step 1: Map dates to holiday names
        FunctionTransformer(lambda df: df['release_date']
                            .map(us_holidays)
                            .fillna('Not Holiday')
                            .to_frame(name="holiday_name")),

        # Step 2: Group the holidays into rating categories
        FunctionTransformer(group_holidays),

        # Step 3: One-hot encode the resulting groups
        OneHotEncoder(sparse_output=False)
    )


def release_date_pipeline():
    return make_pipeline(
        FunctionTransformer(lambda df: df.assign(release_date=pd.to_datetime(df["release_date"]))),
        FeatureUnion([
            ("year", FunctionTransformer(lambda df: df["release_date"].dt.year.to_frame())),
            ("year_group", year_group_pipeline()),
            ("holiday_group", holiday_group_pipeline()),
            ("is_holiday", FunctionTransformer(
                lambda df: pd.DataFrame({
                    "is_holiday": df['release_date'].dt.date.isin(us_holidays)
                })
            )),

            ("weekday", FunctionTransformer(lambda df: df["release_date"].dt.weekday.to_frame(name="weekday"))),
        ])
    )


def os_version_pipeline():
    return make_pipeline(
        FunctionTransformer(lambda X: X.iloc[:, 0].map(extract_min_base_os).to_frame(name="min_os")),
        FunctionTransformer(lambda v: v.astype(float))
    )


def log_pipeline():
    return make_pipeline(FunctionTransformer(np.log1p))

