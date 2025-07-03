import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.preprocessing import StandardScaler
import pytz
def load_data(file_path):
    return pd.read_csv(file_path)

def create_customer_features(df):
    df['TransactionStartTime'] = pd.to_datetime(df['TransactionStartTime'])

    df['Transaction_Hour'] = df['TransactionStartTime'].dt.hour
    df['Transaction_Day'] = df['TransactionStartTime'].dt.day
    df['Transaction_Month'] = df['TransactionStartTime'].dt.month
    df['Transaction_Year'] = df['TransactionStartTime'].dt.year
   
    df['std_Amounts']=df.groupby('CustomerId')['Amount'].std()
    data=df.copy()
    
    
    return data.reset_index()

def calculate_rfm(df):
    snapshot_date=datetime(2025,7,3)
    

    snapshot_date = datetime(2025, 7, 3, tzinfo=pytz.UTC)

    recent_time=df.groupby('CustomerId')['TransactionStartTime'].max()
    recency = (snapshot_date - recent_time).dt.days
    
    frequency = df.groupby('CustomerId')['TransactionId'].nunique()
    
    monetary = df.groupby('CustomerId')['Value'].sum()
    
    rfm = pd.DataFrame({
        'Recency': recency,
        'Frequency': frequency,
        'Monetary': monetary
    })
    
    return rfm.reset_index()
def scaler_rfm(rfm):
   
    scaler = StandardScaler()
    data = rfm.copy()

     
    threshold_m = data['Monetary'].quantile(0.99)
    threshold_f = data['Frequency'].quantile(0.99)
    data = data[(data['Monetary'] < threshold_m) & (data['Frequency'] < threshold_f)]

    
    customer_ids = data['CustomerId'].values
 
    numerical_data = data.select_dtypes(include=['float64', 'int64'])
 
    rfm_scaled = scaler.fit_transform(numerical_data)

  
    rfm_data = pd.DataFrame(rfm_scaled, columns=numerical_data.columns, index=data.index)

   
    rfm_data['CustomerId'] = customer_ids


    return rfm_data
