
import asyncio
import sys
import os
import pandas as pd
from pathlib import Path

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.db.session import SessionLocal
from src.db.repositories.ticker_repo import TickerRepository
from src.db.models.ticker import Ticker

def ingest_csv(csv_path: str):
    if not os.path.exists(csv_path):
        print(f"File not found: {csv_path}")
        return

    print(f"Reading {csv_path}...")
    df = pd.read_csv(csv_path)
    
    # Clean column names (strip spaces)
    df.columns = df.columns.str.strip()
    
    # Expected columns: Symbol, Name, Sector, Industry, Country, IPO Year
    # Map them.
    
    db = SessionLocal()
    repo = TickerRepository(db)
    
    total = len(df)
    print(f"Found {total} tickers. Ingesting...")
    
    count = 0
    for _, row in df.iterrows():
        try:
            symbol = str(row['Symbol']).strip()
            name = str(row['Name']).strip()
            sector = str(row['Sector']).strip() if not pd.isna(row['Sector']) else None
            industry = str(row['Industry']).strip() if not pd.isna(row['Industry']) else None
            country = str(row['Country']).strip() if not pd.isna(row['Country']) else None
            
            ipo_year = row['IPO Year']
            if pd.isna(ipo_year):
                ipo_year = None
            else:
                try:
                    ipo_year = int(ipo_year)
                except:
                    ipo_year = None
            
            # Create Ticker object
            ticker = Ticker(
                ticker=symbol,
                company_name=name,
                sector=sector,
                industry=industry,
                country=country,
                ipo_year=ipo_year,
                # original_currency defaults to null or we can guess USD if country is US? 
                # Better leave for provider to fill/confirm.
            )
            
            repo.upsert(ticker, commit=False)
            count += 1
            if count % 100 == 0:
                repo.session.commit()
                print(f"Processed {count}/{total}...")
                
        except Exception as e:
            print(f"Error processing {row.get('Symbol', 'unknown')}: {e}")
            
    repo.session.commit() # Final commit
    print(f"✅ Ingestion complete. Processed {count} tickers.")
    db.close()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Ingest tickers from CSV")
    parser.add_argument("csv_file", help="Path to CSV file")
    
    args = parser.parse_args()
    ingest_csv(args.csv_file)
