import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load data
df = pd.read_csv("trajectories.csv")
df["timestamp"] = pd.to_datetime(df["timestamp"])

# Sort for correct diff computation
df.sort_values(by=["id", "timestamp"], inplace=True)

# Compute differences and time delta
df["x_diff"] = df.groupby("id")["x"].diff()
df["y_diff"] = df.groupby("id")["y"].diff()
df["dt"] = df.groupby("id")["timestamp"].diff().dt.total_seconds()

# Velocity components
df["v_x"] = df["x_diff"] / df["dt"]
df["v_y"] = df["y_diff"] / df["dt"]
df["velocity"] = np.sqrt(df["v_x"]**2 + df["v_y"]**2)

# Acceleration components
df["v_x_diff"] = df.groupby("id")["v_x"].diff()
df["v_y_diff"] = df.groupby("id")["v_y"].diff()
df["a_x"] = df["v_x_diff"] / df["dt"]
df["a_y"] = df["v_y_diff"] / df["dt"]
df["acceleration"] = np.sqrt(df["a_x"]**2 + df["a_y"]**2)

# Sliding window smoothing using rolling mean
window_size = 3
for col in ["v_x", "v_y", "velocity", "a_x", "a_y", "acceleration"]:
    df[f"{col}_smooth"] = df.groupby("id")[col].transform(
        lambda x: x.rolling(window=window_size, center=True, min_periods=1).mean()
    )

# Optional: Show summary of smoothed data
for unique_id in df['id'].unique():
    print(f"\n--- ID: {unique_id} ---")
    print(df[df["id"] == unique_id][[
        "timestamp", "v_x_smooth", "v_y_smooth", "velocity_smooth",
        "a_x_smooth", "a_y_smooth", "acceleration_smooth"
    ]].head())

# Plotting
all_ids = df["id"].unique()

# Create subplots
fig_vx, ax_vx = plt.subplots()
fig_vy, ax_vy = plt.subplots()
fig_v, ax_v = plt.subplots()
fig_ax, ax_ax = plt.subplots()
fig_ay, ax_ay = plt.subplots()
fig_a, ax_a = plt.subplots()

# Plot smoothed values
for person_id in all_ids:
    person_df = df[df["id"] == person_id]

    # Velocity
    ax_vx.plot(person_df["timestamp"], person_df["v_x_smooth"], label=f"ID {person_id}")
    ax_vy.plot(person_df["timestamp"], person_df["v_y_smooth"], label=f"ID {person_id}")
    ax_v.plot(person_df["timestamp"], person_df["velocity_smooth"], label=f"ID {person_id}")

    # Acceleration
    ax_ax.plot(person_df["timestamp"], person_df["a_x_smooth"], label=f"ID {person_id}")
    ax_ay.plot(person_df["timestamp"], person_df["a_y_smooth"], label=f"ID {person_id}")
    ax_a.plot(person_df["timestamp"], person_df["acceleration_smooth"], label=f"ID {person_id}")

# Customize velocity plots
ax_vx.set_title("Smoothed v_x over time")
ax_vx.set_xlabel("Timestamp")
ax_vx.set_ylabel("v_x (units/s)")
ax_vx.legend()

ax_vy.set_title("Smoothed v_y over time")
ax_vy.set_xlabel("Timestamp")
ax_vy.set_ylabel("v_y (units/s)")
ax_vy.legend()

ax_v.set_title("Smoothed Speed |v| over time")
ax_v.set_xlabel("Timestamp")
ax_v.set_ylabel("Speed (units/s)")
ax_v.legend()

# Customize acceleration plots
ax_ax.set_title("Smoothed a_x over time")
ax_ax.set_xlabel("Timestamp")
ax_ax.set_ylabel("a_x (units/s²)")
ax_ax.legend()

ax_ay.set_title("Smoothed a_y over time")
ax_ay.set_xlabel("Timestamp")
ax_ay.set_ylabel("a_y (units/s²)")
ax_ay.legend()

ax_a.set_title("Smoothed Acceleration Magnitude |a| over time")
ax_a.set_xlabel("Timestamp")
ax_a.set_ylabel("Acceleration (units/s²)")
ax_a.legend()

plt.show()
