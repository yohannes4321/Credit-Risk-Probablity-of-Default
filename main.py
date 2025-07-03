# main.py
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from src.data_processing import load_data, create_customer_features, calculate_rfm,scaler_rfm
from src.feature_engineering import create_preprocessing_pipeline
from src.model_training import train_models
import mlflow

def main():
   
    raw_data = load_data('data/raw/data.csv')
    customer_features = create_customer_features(raw_data)
    
    rfm = calculate_rfm(raw_data)
    rfm_data=scaler_rfm(rfm)
    kmeans=KMeans(n_clusters=3,random_state=42)
    rfm_data['cluster']=kmeans.fit_predict(rfm_data[['Recency','Frequency','Monetary']])
    cluster_center=pd.DataFrame(
        kmeans.cluster_centers_,
        columns=['Recency','Frequency','Monetary']
    )
    highest_risk_index=cluster_center.sort_values(
        by=['Recency','Frequency','Monetary'],
        ascending=[False,True,True]  # determining the worest value
    ).index[0]
  
    rfm_data['high_risk']=(rfm_data['cluster']==highest_risk_index).astype(int)
    
    # # Merge target with features
    final_data = customer_features.merge(
        rfm_data[['CustomerId', 'high_risk']], 
        on='CustomerId'
    )
    final_data=final_data.drop(['BatchId','AccountId','SubscriptionId'],axis=1)
    final_data.to_csv('data/processed/final.csv')
    # Train models
    mlflow.set_tracking_uri("http://localhost:5000")
  
    mlflow.set_experiment("RiskModeling")
    
    X = final_data.drop(['CustomerId', 'is_high_risk'], axis=1)
    y = final_data['is_high_risk']
    
    best_model = train_models(X, y)
    best_model.fit(X, y)  # Final training on full dataset
    
    # # Save final model
    mlflow.sklearn.save_model(best_model, "models/final_model")

if __name__ == "__main__":
    main()