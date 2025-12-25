
import sys
import os
import pandas as pd
import numpy as np
import logging

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.db.session import SessionLocal
from src.db.models import FinancialSnapshot

def verify_dataframe():
    session = SessionLocal()
    try:
        print("--- Step 1: Verify Dataframe ---")
        
        # 1. Load All Data (Simplified for prototype)
        # In production, we'd select specific columns
        query = session.query(FinancialSnapshot)
        df = pd.read_sql(query.statement, session.bind)
        
        print(f"Total Snapshots Loaded: {len(df)}")
        
        if df.empty:
            print("❌ No data found.")
            return

        # 2. Filter Latest Snapshot per Ticker
        # Sort by ticker and date descending, then group by ticker and take head(1)
        # Assuming 'snapshot_timestamp' or 'filing_date' or 'period_end_date'
        # Let's use 'snapshot_timestamp' as it's the primary key part
        df['snapshot_timestamp'] = pd.to_datetime(df['snapshot_timestamp'])
        
        # Sort desc by timestamp
        df_sorted = df.sort_values(['ticker', 'snapshot_timestamp'], ascending=[True, False])
        
        # Keep first (latest)
        df_latest = df_sorted.drop_duplicates(subset=['ticker'], keep='first').copy()
        
        print(f"Unique Tickers (Latest Snapshot): {len(df_latest)}")
        
        # 3. Construct Features (X) and Target (y)
        # Target: Log Market Cap
        # market_cap_t0 is in raw units (presumably) or millions?
        # Let's check a sample value
        sample_mcap = df_latest['market_cap_t0'].iloc[0]
        print(f"Sample Market Cap (Raw): {sample_mcap}")
        
        # Filter where market cap is valid
        df_clean = df_latest.dropna(subset=['market_cap_t0'])
        df_clean = df_clean[df_clean['market_cap_t0'] > 0]
        print(f"Tickers with valid Market Cap: {len(df_clean)}")
        
        # Create Target: Log Market Cap
        # Using base 10 or natural log? "log-mcap" usually implies natural log (ln) in financial contexts or log10.
        # Let's use np.log (natural log)
        df_clean['target_log_mcap'] = np.log(df_clean['market_cap_t0'])
        
        # Features X: 
        # Exclude: ticker, snapshot_timestamp, prices, market_cap, etc.
        # Include: Financial metrics
        
        exclude_cols = [
            'ticker', 'snapshot_timestamp', 'period_end_date', 'filing_date', 'release_date',
            'market_cap_t0', 'price_t0', 'price_t_minus_1', 'price_t_plus_1', 
            'original_currency', 'stored_currency', 'fx_rate_to_usd',
            'created_at', 'updated_at', 'validation_warnings', 'data_quality_score'
        ]
        
        # Start with numeric columns
        numeric_cols = df_clean.select_dtypes(include=[np.number]).columns.tolist()
        
        # Remove excluded
        feature_cols = [c for c in numeric_cols if c not in exclude_cols]
        
        # Drop columns with too many NaNs?
        # Simple imputation: Fill 0 for now (naive)
        X = df_clean[feature_cols].fillna(0)
        
        print(f"Feature Columns ({len(feature_cols)}): {feature_cols[:5]} ...")
        
        # Verify Shapes
        y = df_clean['target_log_mcap']
        
        print(f"Final Data Shape: X={X.shape}, y={y.shape}")
        
        if len(y) > 0:
            print("✅ DataFrame Construction Verified.")
            print(f"   Mean Log Mcap: {y.mean():.4f}")
            print(f"   Std Log Mcap:  {y.std():.4f}")
        else:
            print("❌ DataFrame is empty after processing.")

    finally:
        session.close()

if __name__ == "__main__":
    verify_dataframe()
