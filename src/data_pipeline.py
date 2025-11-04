#Script Responsible for fetching, cleaning, and saving all financial data
#import all libraries
import yfinance as yf  #Financial Data!
import pandas as pd #Data Manipulation & analysis --> Load, Clean, Transform, Analyze structured data
import numpy as np #Library of functions in Python (coded in C & C++) takes a reduced amount of memory and allocation to run large programs
import os     #Helps with interacting with Operating System (File and Directory Options, Path Manipulation, Environment Variables, Process Management, System Information)


#Config

#Define assets (nodes) for our graph
ASSET_TICKERS = ['KC=F','CL=F','^TNX', 'DX-Y.NYB', 'BRL=X', 'ZIM']
#KC = 5 --> benchmark futures contract for "coffee C" -->This is your target asset. It's the price you are ultimately trying to hedge. Its volatility and price movements are what you are trying to protect your "coffee roaster" client from.
#CL = F --> Crude Oil futures --> Oil is a key cost of production and transport. Higher oil prices mean it's more expensive to run farm machinery, process the beans, and ship the coffee around the world. This cost can eventually impact the coffee price.
#^TNX --> 10-year treasury yield -->This is a proxy for global economic health and interest rates. Rising yields can signal inflation or economic growth, which affects global demand. It also influences the U.S. Dollar, which is critical for commodity pricing.
#DX-Y.NYB --> U.S. Dollar Index --> When the dollar gets stronger (DXY goes up), it takes fewer dollars to buy the same amount of coffee. This often pushes the coffee price down.
#BRL = x --> Exchange between dollar and Bralizian Real (BRL) --> When the Brazilian Real gets weaker (BRL=X goes up), Brazilian farmers receive more local currency for their USD-priced coffee. This incentivizes them to sell more, increasing global supply and often pushing coffee prices down.
#'ZIM' --> a publicly traded shipping company, and its stock price is an excellent proxy for the health and cost of the global shipping industry.
START_DATE= '2005-01-01'
END_DATE = pd.to_datetime('today').strftime('%Y-%m-%d')


#Define feature calculation windows
VOLATILITY_WINDOW = 30
RETURN_WINDOW = 5

#Define file paths
DATA_DIR = 'data'
CLEANED_PRICES_PATH = os.path.join(DATA_DIR, 'cleaned_prices.csv')
NODE_FEATURES_PATH = os.path.join(DATA_DIR, 'node_features.csv')

#Main Function

def fetch_data(tickers, start, end):
    #Fetches adjusted close prices for a list of tickers from yFinance
    print(f"Fetching data for: {','.join(tickers)}...")

    raw_data = yf.download(tickers, start=start, end=end, interval='1d')
    
    # Check 1: Did the download fail completely?
    if raw_data.empty:
        print("--- !!! DOWNLOAD FAILED !!! ---")
        print("yfinance returned an empty DataFrame. Check internet or ticker symbols.")
        return pd.DataFrame() # Return an empty DataFrame
    
    # --- FINAL ROBUST SELECTION ---
    data = None
    
    # Check 2: Is it a MultiIndex (successful download of multiple tickers)?
    if isinstance(raw_data.columns, pd.MultiIndex):
        # Try to get 'Adj Close' first
        if 'Adj Close' in raw_data.columns.get_level_values(0):
            print("Found 'Adj Close' column.")
            data = raw_data['Adj Close']
        # If not, fall back to 'Close'
        elif 'Close' in raw_data.columns.get_level_values(0):
            print("Warning: 'Adj Close' not found. Falling back to 'Close'.")
            data = raw_data['Close']
        else:
            print("--- !!! DOWNLOAD ERROR !!! ---")
            print("Downloaded data, but 'Adj Close' AND 'Close' columns are missing.")
            print("Available columns:", raw_data.columns)
            return pd.DataFrame()
            
    # Check 3: Is it a regular Index (successful download of one ticker)?
    else:
        if 'Adj Close' in raw_data.columns:
            print("Found 'Adj Close' column for single ticker.")
            data = raw_data['Adj Close'].to_frame()
        elif 'Close' in raw_data.columns:
            print("Warning: 'Adj Close' not found. Falling back to 'Close' for single ticker.")
            data = raw_data['Close'].to_frame()
        else:
            print("--- !!! DOWNLOAD ERROR !!! ---")
            print("Data was returned, but 'Adj Close' AND 'Close' are missing.")
            print("DataFrame columns are:", raw_data.columns)
            return pd.DataFrame()

    # --- END FINAL SELECTION ---

    # This check is still good to have for the single-ticker case
    if len(tickers) == 1 and isinstance(data, pd.Series):
        data = data.to_frame(tickers[0])
    
    print("Data fetch complete")
    return data


def clean_align_data(df):
    # Cleaning & Aligning
    #    1. Re-Samples for consistent daily frequency (business days)
    #    2. Forward-fills missing values (ex: holidays)
    #    3. Back-fills any remaining NaNs at the beginning.

    print("Cleaning and alinging data...")
    df_aligned = df.asfreq('B')
    df_filled = df_aligned.ffill()
    df_final = df_filled.bfill()

    print("Completed Data Cleaning!")
    return df_final

def calculate_node_features(df):
    
    #Calc node features for the GNN
    #Features: Log Returns and Realized Volatility.
    
    print("Calculating node features...")
    # 1. Log Returns (Price changes)
    log_returns = np.log(df / df.shift(1))
    
    # 2. Realized Volatility (Rolling std dev of log returns)
    volatility = log_returns.rolling(window=VOLATILITY_WINDOW).std() * np.sqrt(252) # Annualized
    
    # Rename columns for clarity (e.g., 'KC=F' -> 'KC=F_return', 'KC=F_vol')
    returns_df = log_returns.add_suffix('_return')
    volatility_df = volatility.add_suffix('_vol')
    
    # Combine features into a single DataFrame
    features_df = pd.concat([returns_df, volatility_df], axis=1)
    
    # Drop initial NaNs created by rolling windows
    features_df = features_df.dropna()
    
    print("Feature calculation complete.")
    return features_df

def build_features():
    # main function run data pipeline and save features

    #Ensures data directory exists
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)

    price_data = fetch_data(ASSET_TICKERS,START_DATE,END_DATE)
    
    # --- ADD THIS CHECK ---
    # Stop if the download failed
    if price_data.empty:
        print("Data fetching failed. Cannot build features. Exiting.")
        return # Stop the function
    # --- END CHECK ---

    price_data_cleaned = clean_align_data(price_data)
    node_features = calculate_node_features(price_data_cleaned)

    #save data
    price_data_cleaned.to_csv(CLEANED_PRICES_PATH)
    node_features.to_csv(NODE_FEATURES_PATH)

    print("\n--- Data Pipeline Success! ---")
    print(f"Cleaned prices saved to '{CLEANED_PRICES_PATH}'")
    print(f"Node features saved to '{NODE_FEATURES_PATH}'")
   
    # Add a final check to see if the features are empty
    if node_features.empty:
        print("\n--- WARNING: EMPTY FEATURES ---")
        print("The final feature set is empty. This is likely because the START_DATE ('2005-01-01')")
        print("is long before the 'ZIM' ticker's IPO (2021). The .dropna() is removing all old rows.")
        print("This is NOT an error, but confirms your data starts around 2021.")
    else:
        print("\nSample Features:")
        print(node_features.tail())
    

if __name__ == "__main__":
    build_features()