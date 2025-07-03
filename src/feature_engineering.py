from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from xverse.transformer import MonotonicBinning, WOE

def create_preprocessing_pipeline(numeric_features, categorical_features):
    """Create feature engineering pipeline"""
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('binning', MonotonicBinning())
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])
    
    return preprocessor

def create_full_pipeline(preprocessor, classifier):
    """Create end-to-end pipeline including WOE transformation"""
    return Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('woe', WOE()),
        ('classifier', classifier)
    ])
