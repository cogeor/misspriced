
import sys
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.db.session import SessionLocal
from src.db.models import FinancialSnapshot

def single_cv_round():
    session = SessionLocal()
    try:
        print("--- Step 2: Single CV Round ---")
        
        # --- DATA LOADING (Same as Step 1) ---
        query = session.query(FinancialSnapshot)
        df = pd.read_sql(query.statement, session.bind)
        df['snapshot_timestamp'] = pd.to_datetime(df['snapshot_timestamp'])
        df_sorted = df.sort_values(['ticker', 'snapshot_timestamp'], ascending=[True, False])
        df_latest = df_sorted.drop_duplicates(subset=['ticker'], keep='first').copy()
        
        df_clean = df_latest.dropna(subset=['market_cap_t0'])
        df_clean = df_clean[df_clean['market_cap_t0'] > 0]
        
        df_clean['target_log_mcap'] = np.log(df_clean['market_cap_t0'])
        
        exclude_cols = [
            'ticker', 'snapshot_timestamp', 'period_end_date', 'filing_date', 'release_date',
            'market_cap_t0', 'price_t0', 'price_t_minus_1', 'price_t_plus_1', 
            'original_currency', 'stored_currency', 'fx_rate_to_usd',
            'created_at', 'updated_at', 'validation_warnings', 'data_quality_score'
        ]
        numeric_cols = df_clean.select_dtypes(include=[np.number]).columns.tolist()
        feature_cols = [c for c in numeric_cols if c not in exclude_cols]
        
        X = df_clean[feature_cols].fillna(0).values
        y = df_clean['target_log_mcap'].values
        tickers = df_clean['ticker'].values
        
        print(f"Data Prepared: {X.shape[0]} samples")
        
        # --- CV ROUND ---
        # 5-fold CV, just running the first fold
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        
        train_index, test_index = next(kf.split(X))
        
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        tickers_test = tickers[test_index]
        
        print(f"Split: Train={len(train_index)}, Test={len(test_index)}")
        
        # Model: Gradient Boosting
        print("Training GradientBoostingRegressor...")
        model = GradientBoostingRegressor(n_estimators=100, max_depth=3, random_state=42)
        model.fit(X_train, y_train)
        
        # Predict
        y_pred_log = model.predict(X_test)
        
        # Metrics (Log Scale)
        mse_log = mean_squared_error(y_test, y_pred_log)
        r2_log = r2_score(y_test, y_pred_log)
        print(f"Log-Scale Metrics: MSE={mse_log:.4f}, R2={r2_log:.4f}")
        
        # Convert back to actual scale for interpretation
        y_pred_actual = np.exp(y_pred_log)
        y_test_actual = np.exp(y_test)
        
        # Relative Error
        # (Pred - Actual) / Actual = Pred/Actual - 1
        mispricing = (y_pred_actual - y_test_actual) / y_test_actual
        
        print("\n--- Sample Predictions ---")
        results_df = pd.DataFrame({
            'Ticker': tickers_test[:5],
            'Actual ($)': y_test_actual[:5],
            'Predicted ($)': y_pred_actual[:5],
            'Mispricing (%)': mispricing[:5]
        })
        print(results_df.to_string())
            
        print("\n✅ Step 2 Completed.")

    finally:
        session.close()

if __name__ == "__main__":
    single_cv_round()
