import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load and parse the timestamp
df = pd.read_csv("trajectories.csv")
df["timestamp"] = pd.to_datetime(df["timestamp"])

# Sort by id and timestamp
df.sort_values(by=["id", "timestamp"], inplace=True)

# Compute position differences
df["x_diff"] = df.groupby("id")["x"].diff()
df["y_diff"] = df.groupby("id")["y"].diff()

# Compute time difference in seconds
df["dt"] = df.groupby("id")["timestamp"].diff().dt.total_seconds()

# Compute velocity components
df["v_x"] = df["x_diff"] / df["dt"]
df["v_y"] = df["y_diff"] / df["dt"]

# Compute total speed (magnitude of velocity vector)
df["velocity"] = np.sqrt(df["v_x"]**2 + df["v_y"]**2)

# Compute velocity differences
df["v_x_diff"] = df.groupby("id")["v_x"].diff()
df["v_y_diff"] = df.groupby("id")["v_y"].diff()

# Compute acceleration components
df["a_x"] = df["v_x_diff"] / df["dt"]
df["a_y"] = df["v_y_diff"] / df["dt"]

# Compute total acceleration
df["acceleration"] = np.sqrt(df["a_x"]**2 + df["a_y"]**2)

# ------------------------------
# Apply rolling average smoothing (window=3)
# ------------------------------
window_size = 3
for col in ["v_x", "v_y", "velocity", "a_x", "a_y", "acceleration"]:
    df[f"{col}_smooth"] = df.groupby("id")[col].transform(
        lambda x: x.rolling(window=window_size, center=True, min_periods=1).mean()
    )

# Show a sample
print(df[["id", "timestamp", "x", "y", "v_x_smooth", "v_y_smooth", "velocity_smooth", "a_x_smooth", "a_y_smooth", "acceleration_smooth"]].head())

print(df.head(5))


# Filter data for IDs 1 and 2
df_1 = df[df["id"] == 0].reset_index(drop=True)
df_2 = df[df["id"] == 2].reset_index(drop=True)

# Calculate relative position vector components
r_x = df_2["x"] - df_1["x"]
r_y = df_2["y"] - df_1["y"]


# Calculate relative acceleration vector components (using smoothed acceleration)
a_x_rel = df_2["a_x_smooth"] - df_1["a_x_smooth"]
a_y_rel = df_2["a_y_smooth"] - df_1["a_y_smooth"]

# Compute magnitudes
r_norm = np.sqrt(r_x**2 + r_y**2)
a_norm = np.sqrt(a_x_rel**2 + a_y_rel**2)

# Dot product of relative position and relative acceleration
dot_ra = r_x * a_x_rel + r_y * a_y_rel

# Cosine of the angle between relative position and relative acceleration
cos_theta_a = dot_ra / (r_norm * a_norm)

# Handle divide-by-zero or NaNs in cos_theta_a
cos_theta_a = cos_theta_a.fillna(0)

# Compute Impact Factor using acceleration
impact_factor = (r_norm / a_norm) * cos_theta_a

# Handle infinite values where acceleration norm might be zero
impact_factor.replace([np.inf, -np.inf], 0, inplace=True)

# Store in df_1
df_1["impact_factor"] = impact_factor


# # Calculate relative velocity vector components (use smoothed velocity if preferred)
# v_x_rel = df_2["v_x_smooth"] - df_1["v_x_smooth"]
# v_y_rel = df_2["v_y_smooth"] - df_1["v_y_smooth"]

# # Compute magnitudes
# r_norm = np.sqrt(r_x**2 + r_y**2)
# v_norm = np.sqrt(v_x_rel**2 + v_y_rel**2)

# # Dot product of relative position and relative velocity
# dot_rv = r_x * v_x_rel + r_y * v_y_rel

# # Cosine of the angle theta
# cos_theta = dot_rv / (r_norm * v_norm)

# # Handle divide-by-zero or NaNs in cos_theta
# cos_theta = cos_theta.fillna(0)

# # Compute Impact Factor
# impact_factor = (r_norm / v_norm) * cos_theta

# # Handle infinite values where velocity norm might be zero
# impact_factor.replace([np.inf, -np.inf], 0, inplace=True)

# # Add to df_1 or create new df for results if needed
# df_1["impact_factor"] = impact_factor

print(df_1[["timestamp", "impact_factor"]].head())

# Your existing plotting code for velocity and acceleration
selected_ids = [0, 2]

# Create subplots for velocity
fig_vx, ax_vx = plt.subplots()
fig_vy, ax_vy = plt.subplots()
fig_v, ax_v = plt.subplots()

# Create subplots for acceleration
fig_ax, ax_ax = plt.subplots()
fig_ay, ax_ay = plt.subplots()
fig_a, ax_a = plt.subplots()

for person_id in selected_ids:
    person_df = df[df["id"] == person_id]

    # Velocity plots (smoothed)
    ax_vx.plot(person_df["timestamp"], person_df["v_x_smooth"], label=f"ID {person_id}")
    ax_vy.plot(person_df["timestamp"], person_df["v_y_smooth"], label=f"ID {person_id}")
    ax_v.plot(person_df["timestamp"], person_df["velocity_smooth"], label=f"ID {person_id}")

    # Acceleration plots (smoothed)
    ax_ax.plot(person_df["timestamp"], person_df["a_x_smooth"], label=f"ID {person_id}")
    ax_ay.plot(person_df["timestamp"], person_df["a_y_smooth"], label=f"ID {person_id}")
    ax_a.plot(person_df["timestamp"], person_df["acceleration_smooth"], label=f"ID {person_id}")

# Impact Factor plot
fig_if, ax_if = plt.subplots(figsize=(10, 4))
ax_if.plot(df_1["timestamp"], df_1["impact_factor"], color="purple", label="Impact Factor (ID 1 & 2)")
ax_if.set_title("Impact Factor Over Time Between Individuals 1 and 2")
ax_if.set_xlabel("Timestamp")
ax_if.set_ylabel("Impact Factor")
ax_if.legend()
ax_if.grid(True)

# Velocity plot customization
ax_vx.set_title("Smoothed v_x over time (IDs 1, 2)")
ax_vx.set_xlabel("Timestamp")
ax_vx.set_ylabel("v_x (units/s)")
ax_vx.legend()

ax_vy.set_title("Smoothed v_y over time (IDs 1, 2)")
ax_vy.set_xlabel("Timestamp")
ax_vy.set_ylabel("v_y (units/s)")
ax_vy.legend()

ax_v.set_title("Smoothed Speed (|v|) over time (IDs 1, 2)")
ax_v.set_xlabel("Timestamp")
ax_v.set_ylabel("Speed (units/s)")
ax_v.legend()

# Acceleration plot customization
ax_ax.set_title("Smoothed a_x over time (IDs 1, 2)")
ax_ax.set_xlabel("Timestamp")
ax_ax.set_ylabel("a_x (units/s²)")
ax_ax.legend()

ax_ay.set_title("Smoothed a_y over time (IDs 1, 2)")
ax_ay.set_xlabel("Timestamp")
ax_ay.set_ylabel("a_y (units/s²)")
ax_ay.legend()

ax_a.set_title("Smoothed Acceleration magnitude over time (IDs 1, 2)")
ax_a.set_xlabel("Timestamp")
ax_a.set_ylabel("|a| (units/s²)")
ax_a.legend()

# Save impact factor with timestamps to CSV
df_1[["timestamp", "impact_factor"]].to_csv("impact_factor.csv", index=False)
df_1[["impact_factor"]].to_csv("impact_factor_no_time.csv", index=False)

# Show all plots
plt.show()
