# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt

# WINDOW_SIZE = 3
# SELECTED_ID_PAIR = [1, 3]  # Selected people IDs


# def load_and_prepare_data(filepath: str) -> pd.DataFrame:
#     df = pd.read_csv(filepath)
#     df["timestamp"] = pd.to_datetime(df["timestamp"])
#     df.sort_values(by=["id", "timestamp"], inplace=True)

#     df["x_diff"] = df.groupby("id")["x"].diff()
#     df["y_diff"] = df.groupby("id")["y"].diff()
#     df["dt"] = df.groupby("id")["timestamp"].diff().dt.total_seconds()

#     df["v_x"] = df["x_diff"] / df["dt"]
#     df["v_y"] = df["y_diff"] / df["dt"]
#     df["velocity"] = np.sqrt(df["v_x"]**2 + df["v_y"]**2)

#     df["v_x_diff"] = df.groupby("id")["v_x"].diff()
#     df["v_y_diff"] = df.groupby("id")["v_y"].diff()
#     df["a_x"] = df["v_x_diff"] / df["dt"]
#     df["a_y"] = df["v_y_diff"] / df["dt"]
#     df["acceleration"] = np.sqrt(df["a_x"]**2 + df["a_y"]**2)

#     return df


# def apply_smoothing(df: pd.DataFrame, window_size: int = WINDOW_SIZE) -> pd.DataFrame:
#     cols_to_smooth = ["v_x", "v_y", "velocity", "a_x", "a_y", "acceleration"]
#     for col in cols_to_smooth:
#         df[f"{col}_smooth"] = (
#             df.groupby("id")[col]
#               .transform(lambda x: x.rolling(window=window_size, center=True, min_periods=1).mean())
#         )
#     return df


# def compute_impact_factor(df1: pd.DataFrame, df2: pd.DataFrame) -> pd.Series:
#     r_x = df2["x"] - df1["x"]
#     r_y = df2["y"] - df1["y"]

#     a_x_rel = df2["a_x_smooth"] - df1["a_x_smooth"]
#     a_y_rel = df2["a_y_smooth"] - df1["a_y_smooth"]

#     r_norm = np.sqrt(r_x**2 + r_y**2)
#     a_norm = np.sqrt(a_x_rel**2 + a_y_rel**2)

#     dot_product = r_x * a_x_rel + r_y * a_y_rel
#     cos_theta = dot_product / (r_norm * a_norm)

#     cos_theta = cos_theta.fillna(0)
#     impact_factor = (r_norm / a_norm) * cos_theta
#     impact_factor.replace([np.inf, -np.inf], 0, inplace=True)

#     return impact_factor


# def plot_metrics(df: pd.DataFrame, df_impact: pd.DataFrame, ids: list[int] = SELECTED_ID_PAIR) -> None:
#     params = [
#         ("v_x_smooth", "Smoothed v_x", "v_x (units/s)"),
#         ("v_y_smooth", "Smoothed v_y", "v_y (units/s)"),
#         ("velocity_smooth", "Smoothed Speed (|v|)", "Speed (units/s)"),
#         ("a_x_smooth", "Smoothed a_x", "a_x (units/s²)"),
#         ("a_y_smooth", "Smoothed a_y", "a_y (units/s²)"),
#         ("acceleration_smooth", "Smoothed Acceleration magnitude", "|a| (units/s²)"),
#     ]

#     total_plots = len(params) + 1  # plus one for impact factor
#     fig, axes = plt.subplots(total_plots, 1, figsize=(12, 3 * total_plots), sharex=True)
    
#     # Plot kinematic params
#     for i, (col, title, ylabel) in enumerate(params):
#         ax = axes[i]
#         ax.set_title(f"{title} over time (IDs {', '.join(map(str, ids))})")
#         ax.set_ylabel(ylabel)
#         ax.grid(True)

#         for pid in ids:
#             df_pid = df[df["id"] == pid]
#             ax.plot(df_pid["timestamp"], df_pid[col], label=f"ID {pid}")
#         ax.legend()

#     # Plot impact factor in the last subplot
#     ax = axes[-1]
#     ax.plot(df_impact["timestamp"], df_impact["impact_factor"], color="purple",
#             label=f"Impact Factor (ID {ids[0]} & {ids[1]})")
#     ax.set_title(f"Impact Factor Over Time Between Individuals {ids[0]} and {ids[1]}")
#     ax.set_xlabel("Timestamp")
#     ax.set_ylabel("Impact Factor")
#     ax.legend()
#     ax.grid(True)

#     plt.tight_layout()
#     plt.show()


# def main():
#     df = load_and_prepare_data("trajectories.csv")
#     df = apply_smoothing(df)

#     df_0 = df[df["id"] == SELECTED_ID_PAIR[0]].reset_index(drop=True)
#     df_2 = df[df["id"] == SELECTED_ID_PAIR[1]].reset_index(drop=True)

#     impact_factor = compute_impact_factor(df_0, df_2)
#     df_0["impact_factor"] = impact_factor

#     plot_metrics(df, df_0[["timestamp", "impact_factor"]], ids=SELECTED_ID_PAIR)

#     df_0[["timestamp", "impact_factor"]].to_csv("impact_factor.csv", index=False)

# if __name__ == "__main__":
#     main()


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def calculate_impact_factor_by_acc(window_size=3, selected_id_pair=[1, 3]):
    WINDOW_SIZE = window_size
    SELECTED_ID_PAIR = selected_id_pair  # Selected people IDs


    def load_and_prepare_data(filepath: str) -> pd.DataFrame:
        df = pd.read_csv(filepath)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df.sort_values(by=["id", "timestamp"], inplace=True)

        df["x_diff"] = df.groupby("id")["x"].diff()
        df["y_diff"] = df.groupby("id")["y"].diff()
        df["dt"] = df.groupby("id")["timestamp"].diff().dt.total_seconds()

        df["v_x"] = df["x_diff"] / df["dt"]
        df["v_y"] = df["y_diff"] / df["dt"]
        df["velocity"] = np.sqrt(df["v_x"]**2 + df["v_y"]**2)

        df["v_x_diff"] = df.groupby("id")["v_x"].diff()
        df["v_y_diff"] = df.groupby("id")["v_y"].diff()
        df["a_x"] = df["v_x_diff"] / df["dt"]
        df["a_y"] = df["v_y_diff"] / df["dt"]
        df["acceleration"] = np.sqrt(df["a_x"]**2 + df["a_y"]**2)

        return df


    def apply_smoothing(df: pd.DataFrame, window_size: int = WINDOW_SIZE) -> pd.DataFrame:
        cols_to_smooth = ["v_x", "v_y", "velocity", "a_x", "a_y", "acceleration"]
        for col in cols_to_smooth:
            df[f"{col}_smooth"] = (
                df.groupby("id")[col]
                .transform(lambda x: x.rolling(window=window_size, center=True, min_periods=1).mean())
            )
        return df


    def compute_impact_factor(df1: pd.DataFrame, df2: pd.DataFrame) -> pd.Series:
        r_x = df2["x"] - df1["x"]
        r_y = df2["y"] - df1["y"]

        a_x_rel = df2["a_x_smooth"] - df1["a_x_smooth"]
        a_y_rel = df2["a_y_smooth"] - df1["a_y_smooth"]

        r_norm = np.sqrt(r_x**2 + r_y**2)
        a_norm = np.sqrt(a_x_rel**2 + a_y_rel**2)

        dot_product = r_x * a_x_rel + r_y * a_y_rel
        cos_theta = dot_product / (r_norm * a_norm)

        cos_theta = cos_theta.fillna(0)
        impact_factor = (r_norm / a_norm) * cos_theta
        impact_factor.replace([np.inf, -np.inf], 0, inplace=True)

        return impact_factor


    def plot_metrics(df: pd.DataFrame, df_impact: pd.DataFrame, ids: list[int] = SELECTED_ID_PAIR) -> None:
        params = [
            ("v_x_smooth", "Smoothed v_x", "v_x (units/s)"),
            ("v_y_smooth", "Smoothed v_y", "v_y (units/s)"),
            ("velocity_smooth", "Smoothed Speed (|v|)", "Speed (units/s)"),
            ("a_x_smooth", "Smoothed a_x", "a_x (units/s²)"),
            ("a_y_smooth", "Smoothed a_y", "a_y (units/s²)"),
            ("acceleration_smooth", "Smoothed Acceleration magnitude", "|a| (units/s²)"),
        ]

        total_plots = len(params) + 1  # plus one for impact factor
        fig, axes = plt.subplots(total_plots, 1, figsize=(12, 3 * total_plots), sharex=True)
        
        # Plot kinematic params
        for i, (col, title, ylabel) in enumerate(params):
            ax = axes[i]
            ax.set_title(f"{title} over time (IDs {', '.join(map(str, ids))})")
            ax.set_ylabel(ylabel)
            ax.grid(True)

            for pid in ids:
                df_pid = df[df["id"] == pid]
                ax.plot(df_pid["timestamp"], df_pid[col], label=f"ID {pid}")
            ax.legend()

        # Plot impact factor in the last subplot
        ax = axes[-1]
        ax.plot(df_impact["timestamp"], df_impact["impact_factor"], color="purple",
                label=f"Impact Factor (ID {ids[0]} & {ids[1]})")
        ax.set_title(f"Impact Factor Over Time Between Individuals {ids[0]} and {ids[1]}")
        ax.set_xlabel("Timestamp")
        ax.set_ylabel("Impact Factor")
        ax.legend()
        ax.grid(True)

        plt.tight_layout()
        plt.show()


    df = load_and_prepare_data("trajectories.csv")
    df = apply_smoothing(df)

    df_0 = df[df["id"] == SELECTED_ID_PAIR[0]].reset_index(drop=True)
    df_2 = df[df["id"] == SELECTED_ID_PAIR[1]].reset_index(drop=True)

    impact_factor = compute_impact_factor(df_0, df_2)
    df_0["impact_factor"] = impact_factor

    plot_metrics(df, df_0[["timestamp", "impact_factor"]], ids=SELECTED_ID_PAIR)

    df_0[["timestamp", "impact_factor"]].to_csv("impact_factor.csv", index=False)

    return impact_factor

# calculate_impact_factor_by_acc()