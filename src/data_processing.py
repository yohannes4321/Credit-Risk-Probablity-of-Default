import pandas as pd
import numpy as np
from datetime import datetime

def load_data(file_path):
    return pd.read_csv(file_path)

def create_customer_features(df):
    """Aggregate transaction data to customer-level features"""
    # Convert to datetime
    df['TransactionStartTime'] = pd.to_datetime(df['TransactionStartTime'])
    
    # Feature extraction
    df['Transaction_Hour'] = df['TransactionStartTime'].dt.hour
    df['Transaction_Day'] = df['TransactionStartTime'].dt.day
    df['Transaction_Month'] = df['TransactionStartTime'].dt.month
    df['Transaction_Year'] = df['TransactionStartTime'].dt.year
    
    # Customer aggregations
    agg_funcs = {
        'Amount': ['sum', 'mean', 'std', 'count'],
        'Transaction_Hour': ['mean', 'std'],
        'Transaction_Day': ['mean', 'std'],
        'Transaction_Month': ['mean'],
        'Transaction_Year': ['max'],
        'ProviderId': pd.Series.nunique,
        'ProductCategory': pd.Series.nunique,
        'FraudResult': 'sum'
    }
    
    customer_features = df.groupby('CustomerId').agg(agg_funcs)
    customer_features.columns = [
        'Total_Amount', 'Avg_Amount', 'Std_Amount', 'Transaction_Count',
        'Avg_Hour', 'Std_Hour', 'Avg_Day', 'Std_Day', 'Avg_Month', 
        'Max_Year', 'Unique_Providers', 'Unique_Categories', 'Fraud_Count'
    ]
    
    return customer_features.reset_index()

def calculate_rfm(df, snapshot_date):
    """Calculate RFM metrics for each customer"""
    recency = df.groupby('CustomerId')['TransactionStartTime'].max()
    recency = (snapshot_date - recency).dt.days
    
    frequency = df.groupby('CustomerId').size()
    
    monetary = df.groupby('CustomerId')['Amount'].sum()
    
    rfm = pd.DataFrame({
        'Recency': recency,
        'Frequency': frequency,
        'Monetary': monetary
    })
    
    return rfm.reset_index()