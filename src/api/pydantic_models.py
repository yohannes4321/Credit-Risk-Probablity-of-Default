from pydantic import BaseModel

class CustomerData(BaseModel):
    CustomerId: str
    Total_Amount: float
    Avg_Amount: float
    Std_Amount: float
    Transaction_Count: int
    Avg_Hour: float
    Std_Hour: float
    Avg_Day: float
    Std_Day: float
    Avg_Month: float
    Max_Year: int
    Unique_Providers: int
    Unique_Categories: int
    Fraud_Count: int