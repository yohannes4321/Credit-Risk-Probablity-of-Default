import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, roc_auc_score
from src.feature_engineering import create_preprocessing_pipeline,create_full_pipeline
def train_models(X, y):
    """Train and evaluate models with hyperparameter tuning"""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Define models and parameters
    models = {
        'LogisticRegression': {
            'model': LogisticRegression(),
            'params': {'classifier__C': [0.1, 1, 10]}
        },
        'RandomForest': {
            'model': RandomForestClassifier(),
            'params': {
                'classifier__n_estimators': [50, 100],
                'classifier__max_depth': [3, 5]
            }
        },
        'XGBoost': {
            'model': XGBClassifier(),
            'params': {
                'classifier__n_estimators': [50, 100],
                'classifier__learning_rate': [0.01, 0.1]
            }
        }
    }
    
    best_model = None
    best_score = 0
    
    for name, config in models.items():
        with mlflow.start_run(run_name=name):
            # Create pipeline
            pipeline = create_full_pipeline(
                create_preprocessing_pipeline(
                    numeric_features=X_train.select_dtypes(include='number').columns.tolist(),
                    categorical_features=X_train.select_dtypes(exclude='number').columns.tolist()
                ),
                config['model']
            )
            
            # Hyperparameter tuning
            search = RandomizedSearchCV(
                pipeline,
                config['params'],
                cv=3,
                scoring='roc_auc'
            )
            search.fit(X_train, y_train)
            
            # Evaluate
            y_pred = search.predict(X_test)
            report = classification_report(y_test, y_pred)
            roc_auc = roc_auc_score(y_test, search.predict_proba(X_test)[:, 1])
            
            # Log metrics
            mlflow.log_params(search.best_params_)
            mlflow.log_metric("roc_auc", roc_auc)
            print(f"{name} ROC-AUC: {roc_auc:.4f}")
            
            # Register best model
            if roc_auc > best_score:
                best_score = roc_auc
                best_model = search.best_estimator_
                mlflow.sklearn.log_model(best_model, "best_model")
                mlflow.register_model(
                    f"runs:/{mlflow.active_run().info.run_id}/best_model",
                    "RiskModel"
                )
    
    return best_model