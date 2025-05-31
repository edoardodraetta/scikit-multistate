import pandas as pd
from pathlib import Path
import numpy as np

AIDSSI_PATH = "../src/skms/data/samples/aidssi.csv"


def load_aidssi(prepare=False) -> pd.DataFrame:
    """Loads the AIDS SI dataset.

    Raises:
        FileNotFoundError: If the dataset file cannot be found.

    Returns:
        pd.DataFrame: AIDS SI dataset.
    """

    file_path = Path(AIDSSI_PATH)
    if not file_path.exists():
        raise FileNotFoundError(f"AIDS SI dataset not found at {file_path}")

    df = pd.read_csv(file_path, index_col=0)
    if prepare:
        df = prepare_aidssi(df)
    return df


def prepare_aidssi(df):
    """
    Prepare multistate data with counterfactuals, preserving covariates.

    Parameters:
    -----------
    df : DataFrame
        Input data with columns: patnr, time, status, and any covariates

    """
    transitions = df.copy()

    covariate_cols = ["ccr5"]

    # Add tstart column
    transitions["tstart"] = 0
    # Add origin state column
    transitions["origin_state"] = 0
    # Rename columns
    transitions.rename(columns={"time": "tstop", "status": "target_state"}, inplace=True)
    # Add censoring column
    transitions["status"] = np.where(transitions["target_state"] == 0, 0, 1)
    # Add dummy target state to censored observations
    transitions.loc[transitions["target_state"] == 0, "target_state"] = 1

    # Add counterfactuals for each observation
    counterfactuals = []
    for _, row in transitions.iterrows():
        # For each transition to state k, add counterfactuals to all other possible states
        possible_states = [1, 2]  # Adjust based on your actual states
        current_target = row["target_state"]

        for state in possible_states:
            if state != current_target:
                # Create counterfactual transition
                counterfactual = row.copy()
                counterfactual["target_state"] = state
                counterfactual["status"] = 0  # Counterfactuals are censored
                counterfactuals.append(counterfactual)

    # Combine original transitions with counterfactuals
    if counterfactuals:
        counterfactuals_df = pd.DataFrame(counterfactuals)
        transitions = pd.concat([transitions, counterfactuals_df], ignore_index=True)

    # Sort by patient number and target state for clarity
    transitions = transitions.sort_values(["patnr", "target_state"]).reset_index(drop=True)

    # Return all columns including covariates
    base_cols = ["patnr", "tstart", "tstop", "origin_state", "target_state", "status"]
    return transitions[base_cols + covariate_cols]


def list_available_datasets() -> dict:
    """Show all available sample datasets with descriptions.

    Returns:
        dict: Dictionary mapping dataset names to their descriptions and availability
    """
    datasets = {
        "aidssi": {
            "description": "AIDS SI dataset - Survival information for AIDS patients",
            "function": "load_aidssi()",
            "path": AIDSSI_PATH,
            "shape": (None, None),
            "columns": [],
        },
    }

    # Check which datasets are actually available
    for _, info in datasets.items():
        file_path = Path(info["path"])
        if file_path.exists():
            try:
                # Get basic info about the dataset
                df = pd.read_csv(file_path, index_col=0)
                info["shape"] = df.shape
                info["columns"] = list(df.columns)
            except Exception as e:
                info["error"] = str(e)

    return datasets
