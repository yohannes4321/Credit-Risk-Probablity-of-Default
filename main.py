# main.py
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from src.data_processing import load_data, create_customer_features, calculate_rfm
from src.feature_engineering import create_preprocessing_pipeline
from src.model_training import train_models
import mlflow

def main():
    # Load and process data
    raw_data = load_data('data/raw/data.csv')
    customer_features = create_customer_features(raw_data)
    
    # Create target variable
    snapshot_date = raw_data['TransactionStartTime'].max() + pd.Timedelta(days=1)
    rfm = calculate_rfm(raw_data, snapshot_date)
    
    # Scale and cluster
    rfm_scaled = rfm[['Recency', 'Frequency', 'Monetary']].copy()
    rfm_scaled = (rfm_scaled - rfm_scaled.mean()) / rfm_scaled.std()
    
    kmeans = KMeans(n_clusters=3, random_state=42)
    rfm['cluster'] = kmeans.fit_predict(rfm_scaled)
    
    # Identify high-risk cluster
    cluster_stats = rfm.groupby('cluster')[['Recency', 'Frequency', 'Monetary']].mean()
    high_risk_cluster = cluster_stats['Frequency'].idxmin()  # Lowest frequency
    
    # Create target
    rfm['is_high_risk'] = (rfm['cluster'] == high_risk_cluster).astype(int)
    
    # Merge target with features
    final_data = customer_features.merge(
        rfm[['CustomerId', 'is_high_risk']], 
        on='CustomerId'
    )
    
    # Train models
    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment("RiskModeling")
    
    X = final_data.drop(['CustomerId', 'is_high_risk'], axis=1)
    y = final_data['is_high_risk']
    
    best_model = train_models(X, y)
    best_model.fit(X, y)  # Final training on full dataset
    
    # Save final model
    mlflow.sklearn.save_model(best_model, "models/final_model")

if __name__ == "__main__":
    main()