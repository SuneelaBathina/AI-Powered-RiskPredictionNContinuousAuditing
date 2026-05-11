import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

np.random.seed(42)
random.seed(42)

def generate_large_sample(n_records=1000):
    data = []
    transaction_types = ['Payment', 'Transfer', 'Withdrawal', 'Deposit', 'Refund', 'Purchase']
    locations = ['New York', 'London', 'Tokyo', 'Singapore', 'Dubai', 'Mumbai', 'Shanghai', 'Paris', 'Sydney', 'Toronto']
    devices = ['Desktop', 'Mobile', 'Tablet']
    accounts = [f'ACC-{i:04d}' for i in range(1000, 2000)]
    
    for i in range(n_records):
        # Generate realistic amount (lognormal distribution)
        amount = np.random.lognormal(mean=8, sigma=1.5)
        
        # Calculate risk score based on amount and random factors
        amount_risk = min(60, (amount / 100000) * 40)
        type_risk = 20 if np.random.choice(transaction_types) in ['Transfer', 'Withdrawal'] else 0
        location_risk = 15 if np.random.random() > 0.9 else 0
        risk_score = min(100, amount_risk + type_risk + location_risk + np.random.uniform(0, 15))
        
        # Determine status based on risk score
        if risk_score > 85:
            status = 'critical'
            is_fraudulent = True
        elif risk_score > 70:
            status = 'flagged'
            is_fraudulent = True if np.random.random() > 0.5 else False
        elif risk_score > 40:
            status = 'reviewing'
            is_fraudulent = False
        else:
            status = 'approved'
            is_fraudulent = False
        
        data.append({
            'transaction_id': f'TX-{datetime.now().strftime("%Y%m%d")}-{i:06d}',
            'amount': round(amount, 2),
            'timestamp': (datetime.now() - timedelta(days=random.randint(0, 90), hours=random.randint(0, 23))).strftime('%Y-%m-%d %H:%M:%S'),
            'account_id': random.choice(accounts),
            'transaction_type': random.choice(transaction_types),
            'location': random.choice(locations),
            'device': random.choice(devices),
            'risk_score': round(risk_score, 1),
            'is_fraudulent': is_fraudulent,
            'status': status
        })
    
    return pd.DataFrame(data)

# Generate 1000 transactions
df = generate_large_sample(1000)
df.to_csv('financial_transaction_large.csv', index=False)
print(f"Generated {len(df)} transactions")
print(f"File saved as: financial_transaction_large.csv")
print(f"\nSummary:")
print(f"  - Total amount: ${df['amount'].sum():,.2f}")
print(f"  - Avg amount: ${df['amount'].mean():,.2f}")
print(f"  - High risk transactions (>70%): {len(df[df['risk_score'] > 70])}")
print(f"  - Fraudulent transactions: {df['is_fraudulent'].sum()}")