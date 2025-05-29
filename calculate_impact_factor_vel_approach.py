import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


DATA_PATH = "trajectories.csv"
WINDOW_SIZE = 3
SELECTED_ID_PAIR = [1, 3]


def load_and_prepare_data(path: str) -> pd.DataFrame:
    """
    Load CSV data, parse timestamps, sort, and compute
    position differences, velocities, and accelerations.
    Returns a DataFrame with all computed columns.
    """
    df = pd.read_csv(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df.sort_values(by=["id", "timestamp"], inplace=True)

    # Position differences
    df["x_diff"] = df.groupby("id")["x"].diff()
    df["y_diff"] = df.groupby("id")["y"].diff()

    # Time difference in seconds
    df["dt"] = df.groupby("id")["timestamp"].diff().dt.total_seconds()

    # Velocity components
    df["v_x"] = df["x_diff"] / df["dt"]
    df["v_y"] = df["y_diff"] / df["dt"]

    # Velocity magnitude
    df["velocity"] = np.sqrt(df["v_x"] ** 2 + df["v_y"] ** 2)

    # Velocity differences for acceleration
    df["v_x_diff"] = df.groupby("id")["v_x"].diff()
    df["v_y_diff"] = df.groupby("id")["v_y"].diff()

    # Acceleration components
    df["a_x"] = df["v_x_diff"] / df["dt"]
    df["a_y"] = df["v_y_diff"] / df["dt"]

    # Acceleration magnitude
    df["acceleration"] = np.sqrt(df["a_x"] ** 2 + df["a_y"] ** 2)

    return df


def apply_rolling_smoothing(df: pd.DataFrame, window: int) -> pd.DataFrame:
    """
    Apply rolling average smoothing with a given window size
    to velocity and acceleration columns.
    """
    cols_to_smooth = ["v_x", "v_y", "velocity", "a_x", "a_y", "acceleration"]
    for col in cols_to_smooth:
        smoothed_col = f"{col}_smooth"
        df[smoothed_col] = df.groupby("id")[col].transform(
            lambda x: x.rolling(window=window, center=True, min_periods=1).mean()
        )
    return df


def calculate_impact_factor(df: pd.DataFrame, id_pair: list[int]) -> pd.DataFrame:
    """
    Calculate impact factor between two individuals given by id_pair,
    based on relative distances and relative velocities.
    Returns the dataframe filtered for the first id in the pair,
    with a new 'impact_factor' column.
    """
    df_1 = df[df["id"] == id_pair[0]].reset_index(drop=True)
    df_2 = df[df["id"] == id_pair[1]].reset_index(drop=True)

    # Relative position components
    r_x = df_2["x"] - df_1["x"]
    r_y = df_2["y"] - df_1["y"]

    # Relative velocity components (smoothed)
    v_x_rel = df_2["v_x_smooth"] - df_1["v_x_smooth"]
    v_y_rel = df_2["v_y_smooth"] - df_1["v_y_smooth"]

    # Magnitudes
    r_norm = np.sqrt(r_x ** 2 + r_y ** 2)
    v_norm = np.sqrt(v_x_rel ** 2 + v_y_rel ** 2)

    # Dot product and cosine angle
    dot_rv = r_x * v_x_rel + r_y * v_y_rel
    cos_theta = dot_rv / (r_norm * v_norm)

    # Replace NaNs and infinite values
    cos_theta = cos_theta.fillna(0)
    impact_factor = (r_norm / v_norm) * cos_theta
    impact_factor.replace([np.inf, -np.inf], 0, inplace=True)

    df_1["impact_factor"] = impact_factor
    return df_1


def plot_metrics(df: pd.DataFrame, df_impact: pd.DataFrame, ids: list[int] = SELECTED_ID_PAIR) -> None:
    params = [
        ("v_x_smooth", "Smoothed v_x", "v_x (units/s)"),
        ("v_y_smooth", "Smoothed v_y", "v_y (units/s)"),
        ("velocity_smooth", "Smoothed Speed (|v|)", "Speed (units/s)"),
        ("a_x_smooth", "Smoothed a_x", "a_x (units/s²)"),
        ("a_y_smooth", "Smoothed a_y", "a_y (units/s²)"),
        ("acceleration_smooth", "Smoothed Acceleration magnitude", "|a| (units/s²)"),
    ]

    total_plots = len(params) + 1  # add 1 for impact factor subplot
    fig, axes = plt.subplots(total_plots, 1, figsize=(12, 3 * total_plots), sharex=True)

    for i, (col, title, ylabel) in enumerate(params):
        ax = axes[i]
        ax.set_title(f"{title} over time (IDs {', '.join(map(str, ids))})")
        ax.set_ylabel(ylabel)
        ax.grid(True)

        for pid in ids:
            df_pid = df[df["id"] == pid]
            ax.plot(df_pid["timestamp"], df_pid[col], label=f"ID {pid}")
        ax.legend()

    # Impact factor plot in last subplot
    ax = axes[-1]
    ax.plot(df_impact["timestamp"], df_impact["impact_factor"], color="purple",
            label=f"Impact Factor (IDs {ids[0]} & {ids[1]})")
    ax.set_title(f"Impact Factor Over Time Between Individuals {ids[0]} and {ids[1]}")
    ax.set_xlabel("Timestamp")
    ax.set_ylabel("Impact Factor")
    ax.legend()
    ax.grid(True)

    plt.tight_layout()
    plt.show()



def main():
    df = load_and_prepare_data(DATA_PATH)
    df = apply_rolling_smoothing(df, WINDOW_SIZE)

    impact_df = calculate_impact_factor(df, SELECTED_ID_PAIR)
    print(impact_df[["timestamp", "impact_factor"]].head())

    plot_metrics(df, impact_df, SELECTED_ID_PAIR)

    impact_df[["timestamp", "impact_factor"]].to_csv("impact_factor.csv", index=False)
    impact_df[["impact_factor"]].to_csv("impact_factor_no_time.csv", index=False)



if __name__ == "__main__":
    main()

