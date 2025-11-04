import pandas as pd
import numpy as np
import os
import pickle

# Config
CORRELATION_WINDOW = 60 # Use a 60-day rolling window
DATA_DIR = 'data'
CLEANED_PRICES_PATH = os.path.join(DATA_DIR, 'cleaned_prices.csv')
DYNAMIC_GRAPH_PATH = os.path.join(DATA_DIR, 'dynamic_graph.pkl') # Save as a pickle file

# Main jawn

def build_dynamic_graph(price_data):
    """
    Calculates the dynamic graph structure (adjacency matrices) based on
    rolling correlations of log returns.
    
    Returns:
    - A dictionary where keys are dates (timestamps) and
      values are the (num_assets x num_assets) correlation matrices for that day.
    """
    print("Building dynamic graph (rolling correlations)...")
    
    log_returns = np.log(price_data / price_data.shift(1))
    rolling_corr = log_returns.rolling(window=CORRELATION_WINDOW).corr()
    rolling_corr = rolling_corr.dropna()
    
    dynamic_graph = {}
    all_dates = rolling_corr.index.get_level_values(0).unique()
    
    for date in all_dates:
        corr_matrix = rolling_corr.loc[date]
        dynamic_graph[date] = corr_matrix.values
        
    print(f"Dynamic graph built. {len(dynamic_graph)} daily adjacency matrices created.")
    return dynamic_graph

if __name__ == "__main__":
    try:
        prices = pd.read_csv(CLEANED_PRICES_PATH, index_col=0, parse_dates=True)
    except FileNotFoundError:
        print(f"Error: '{CLEANED_PRICES_PATH}' not found.")
        print("Please run 'python src/data_pipeline.py' first.")
        exit()

    dynamic_graph = build_dynamic_graph(prices)
    
    # Save graph object
    with open(DYNAMIC_GRAPH_PATH, 'wb') as f:
        pickle.dump(dynamic_graph, f)
        
    print(f"Dynamic graph saved to '{DYNAMIC_GRAPH_PATH}'")
    
    # ex output
    print("\n--- Graph Builder Success! ---")
    last_date = list(dynamic_graph.keys())[-1]
    print(f"Example adjacency matrix for date: {last_date.strftime('%Y-%m-%d')}")
    print(pd.DataFrame(dynamic_graph[last_date], index=prices.columns, columns=prices.columns))