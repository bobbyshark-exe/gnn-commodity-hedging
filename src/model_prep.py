import pandas as pd
import numpy as np
import pickle
import os
from sklearn.preprocessing import StandardScaler

# --- Configuration ---
DATA_DIR = 'data'
FEATURES_PATH = os.path.join(DATA_DIR, 'node_features.csv')
GRAPH_PATH = os.path.join(DATA_DIR, 'dynamic_graph.pkl')
PROCESSED_DATA_PATH = os.path.join(DATA_DIR, 'processed_data.pkl')

# ML Task Configuration
TARGET_ASSET = 'KC=F'       # We want to predict Coffee volatility
TARGET_FEATURE = f'{TARGET_ASSET}_vol'
PREDICTION_HORIZON = 30     # We're predicting 30 (trading) days into the future
SEQUENCE_LENGTH = 10        # We'll use the last 10 days of data as input (X)

# --- Main Functions ---

def load_data():
    """Loads the features and graph data from Stage 1."""
    print("Loading data from Stage 1...")
    try:
        features = pd.read_csv(FEATURES_PATH, index_col=0, parse_dates=True)
        
        with open(GRAPH_PATH, 'rb') as f:
            dynamic_graph = pickle.load(f)
            
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please run 'python src/data_pipeline.py' and 'python src/graph_builder.py' first.")
        return None, None
        
    # CRITICAL: Align data by ensuring we only use dates present in BOTH datasets
    aligned_dates = features.index.intersection(dynamic_graph.keys())
    features = features.loc[aligned_dates]
    
    # Prune the graph dictionary to only include dates that are in the features
    aligned_graph = {date: dynamic_graph[date] for date in aligned_dates}
    
    print(f"Data aligned. Using {len(features)} common observations.")
    
    return features, aligned_graph

def create_labels(features):
    """
    Creates the target variable (y) for our classification task.
    
    We define 3 volatility regimes based on the future 30-day volatility:
    - 0: 'Calm' (bottom 33% of volatility)
    - 1: 'Volatile' (middle 33% of volatility)
    - 2: 'Extreme' (top 33% of volatility)
    """
    print("Creating classification labels...")
    
    # 1. Calculate the future volatility
    #    We use .shift(-PREDICTION_HORIZON) to look *forward* in time.
    future_vol = features[TARGET_FEATURE].shift(-PREDICTION_HORIZON)
    
    # 2. Define the quantiles (thresholds) for our regimes
    #    We calculate this *only* on the available future data (excluding NaNs)
    quantiles = future_vol.dropna().quantile([0.33, 0.67])
    q_low = quantiles[0.33]
    q_high = quantiles[0.67]
    
    print(f"Regime thresholds: Calm < {q_low:.4f} < Volatile < {q_high:.4f} < Extreme")
    
    # 3. Create the labels
    labels = pd.Series(np.nan, index=features.index, dtype=int)
    
    labels.loc[future_vol <= q_low] = 0  # Calm
    labels.loc[(future_vol > q_low) & (future_vol <= q_high)] = 1  # Volatile
    labels.loc[future_vol > q_high] = 2  # Extreme
    
    return labels

def create_sequences(features, dynamic_graph, labels, first_valid_date):
    """
    Creates the final "snapshot" dataset for the GNN.
    
    Each snapshot consists of:
    - X_seq (Features): The last {SEQUENCE_LENGTH} days of node features.
    - A (Graph): The adjacency matrix for the *current* day.
    - y (Label): The volatility regime {PREDICTION_HORIZON} days in the future.
    """
    print(f"Creating sequences with {SEQUENCE_LENGTH}-day lookback...")
    
    # Scale features
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    # Convert back to DataFrame for easier handling
    features_scaled_df = pd.DataFrame(features_scaled, index=features.index, columns=features.columns)
    
    num_nodes = len(features.columns) // 2
    num_features_per_node = 2

    X, A, y = [], [], []

    # --- THIS IS THE NEW, CORRECTED LOOP ---
    
    # We can only create sequences for dates where we have a valid label
    # AND we have enough historical data.
    valid_label_dates = labels.dropna().index
    
    # Find the common dates that satisfy BOTH conditions
    eligible_dates = valid_label_dates.intersection(features_scaled_df.loc[first_valid_date:].index)
    
    print(f"Found {len(eligible_dates)} eligible dates for sequence creation...")

    # Now, loop ONLY through the dates we know will work
    for current_date in eligible_dates:
        
        # Find the integer position of the current date
        current_idx = features_scaled_df.index.get_loc(current_date)
        
        # Get the 10-day sequence ending at this index
        seq_df = features_scaled_df.iloc[current_idx - (SEQUENCE_LENGTH - 1) : current_idx + 1]
        
        # This check should always pass now, but it's good practice
        if len(seq_df) == SEQUENCE_LENGTH:
            
            # --- The rest of the code is the same ---
            
            # 1. Get feature data
            X_seq_data = seq_df.values.T # Transpose to (num_features, seq_length)
            
            # 2. Reshape to (num_nodes, num_features_per_node, seq_length)
            X_seq_reshaped = X_seq_data.reshape(num_nodes, num_features_per_node, SEQUENCE_LENGTH)
            
            # 3. Transpose to (num_nodes, seq_length, num_features_per_node)
            X_seq_final = X_seq_reshaped.transpose(0, 2, 1)

            # Get the graph (adjacency matrix) for the *current* day
            adj_matrix = dynamic_graph[current_date]
            
            # Get the label
            label = labels.loc[current_date]
            
            # Add to our lists
            X.append(X_seq_final)
            A.append(adj_matrix)
            y.append(label)

    # --- END FIXED LOOP ---
    
    print(f"Created {len(y)} sequences.")
    return X, A, y

if __name__ == "__main__":
    features_df, graph_dict = load_data()
    
    if features_df is not None:
        labels_series = create_labels(features_df)
        
        # We can only start creating sequences after our first {SEQUENCE_LENGTH} days
        first_valid_date = features_df.index[SEQUENCE_LENGTH - 1]
        
        X_list, A_list, y_list = create_sequences(features_df, graph_dict, labels_series, first_valid_date)
        
        # --- Save Processed Data ---
        
        # Add a check to ensure list is not empty before saving/printing
        if X_list:
            processed_data = {
                'X': X_list,
                'A': A_list,
                'y': y_list,
                'assets': list(features_df.columns[features_df.columns.str.contains('_return')].str.replace('_return', ''))
            }
            
            with open(PROCESSED_DATA_PATH, 'wb') as f:
                pickle.dump(processed_data, f)
                
            print("\n--- Model Prep Success! ---")
            print(f"Processed data saved to '{PROCESSED_DATA_PATH}'")
            print("\n--- Data Specs ---")
            print(f"Total snapshots: {len(y_list)}")
            print(f"Assets: {processed_data['assets']}")
            print(f"X shape (one sample): {X_list[0].shape} (Nodes, Seq_Len, Features)")
            print(f"A shape (one sample): {A_list[0].shape} (Nodes, Nodes)")
            print(f"y sample (one sample): {y_list[0]}")
        
        else:
            print("\n--- !!! Error !!! ---")
            print("No sequences were created. Check data alignment and sequence logic.")
    
    # This 'else' block will now run if 'features_df' is None (i.e., loading failed)
    else:
        print("\n--- !!! SCRIPT FAILED TO RUN !!! ---")
        print("Data loading returned 'None'.")
        print("Please ensure 'data/node_features.csv' and 'data/dynamic_graph.pkl' exist.")
        print("Try running the prerequisite scripts first.")