import pandas as pd
from datetime import datetime
from src.data_processing import create_customer_features, calculate_rfm

def test_feature_engineering():
    """Test customer feature creation"""
    data = pd.DataFrame({
        'CustomerId': ['C1', 'C1', 'C2'],
        'Amount': [100, 200, 50],
        'TransactionStartTime': [
            '2023-01-01 08:00',
            '2023-01-02 12:00',
            '2023-01-03 18:00'
        ],
        'ProviderId': ['P1', 'P2', 'P1'],
        'ProductCategory': ['A', 'B', 'A']
    })
    
    features = create_customer_features(data)
    
    assert features.shape == (2, 14)
    assert features.loc[0, 'Total_Amount'] == 300
    assert features.loc[0, 'Transaction_Count'] == 2
    assert features.loc[0, 'Unique_Providers'] == 2

def test_rfm_calculation():
    """Test RFM metric calculation"""
    data = pd.DataFrame({
        'CustomerId': ['C1', 'C1', 'C2'],
        'TransactionStartTime': pd.to_datetime([
            '2023-01-01', '2023-01-05', '2023-01-10'
        ]),
        'Amount': [100, 200, 150]
    })
    
    snapshot_date = datetime(2023, 1, 15)
    rfm = calculate_rfm(data, snapshot_date)
    
    assert rfm.loc[0, 'Recency'] == 10
    assert rfm.loc[0, 'Frequency'] == 2
    assert rfm.loc[0, 'Monetary'] == 300