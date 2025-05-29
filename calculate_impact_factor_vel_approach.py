# # import pandas as pd
# # import numpy as np
# # from itertools import combinations

# # # Load data
# # df = pd.read_csv("projected_points.csv")

# # # Compute velocities
# # df[['vx', 'vy', 'vz']] = 0.0

# # # Sort by pedestrian ID and frame
# # df.sort_values(by=['pedestrian_id', 'frame'], inplace=True)

# # # Calculate velocities per pedestrian
# # for pid in df['pedestrian_id'].unique():
# #     ped_df = df[df['pedestrian_id'] == pid]
# #     ped_df = ped_df.sort_values(by='frame')
# #     vel = ped_df[['X', 'Y', 'Z']].diff().fillna(0)
# #     df.loc[vel.index, ['vx', 'vy', 'vz']] = vel.values

# # # Compute interaction (impact factor) per frame
# # impact_factors = []

# # sigma = 1.0  # Decay parameter

# # for frame in sorted(df['frame'].unique()):
# #     frame_data = df[df['frame'] == frame]
# #     for (i1, p1), (i2, p2) in combinations(frame_data.iterrows(), 2):
# #         pos1 = np.array([p1['X'], p1['Y'], p1['Z']])
# #         pos2 = np.array([p2['X'], p2['Y'], p2['Z']])
# #         vel1 = np.array([p1['vx'], p1['vy'], p1['vz']])
# #         vel2 = np.array([p2['vx'], p2['vy'], p2['vz']])
        
# #         dist = np.linalg.norm(pos1 - pos2)
# #         vel_diff = np.linalg.norm(vel1 - vel2)
        
# #         impact = np.exp(-dist / sigma) / (1 + vel_diff)
        
# #         impact_factors.append({
# #             'frame': frame,
# #             'id1': int(p1['pedestrian_id']),
# #             'id2': int(p2['pedestrian_id']),
# #             'impact_factor': impact
# #         })

# # # Save to CSV
# # impact_df = pd.DataFrame(impact_factors)
# # impact_df.to_csv("impact_factors.csv", index=False)



# import csv
# from collections import defaultdict

# def read_trajectories(csv_path):
#     """Reads tracking data from CSV and returns a dictionary of trajectories."""
#     trajectories = defaultdict(list)  # {track_id: [(X, Y, Z), ...]}

#     with open(csv_path, mode='r') as file:
#         reader = csv.DictReader(file)
#         for row in reader:
#             track_id = int(row["track_id"])
#             X = float(row["X"])
#             Y = float(row["Y"])
#             Z = float(row["Z"])
#             trajectories[track_id].append((X, Y, Z))

#     return trajectories

# # Example usage
# csv_file_path = "trajectories.csv"
# trajectories = read_trajectories(csv_file_path)

# # Print example output
# for track_id, coords in trajectories.items():
#     print(f"Track ID {track_id}: {len(coords)} points")
#     for point in coords[:5]:  # Print first 5 points per track
#         print(f"\t{point}")


# # import pandas as pd
# # import numpy as np
# # from itertools import combinations

# # df = pd.read_csv("projected_points.csv")
# # df[['vx', 'vy', 'vz']] = 0.0
# # df.sort_values(by=['pedestrian_id', 'frame'], inplace=True)

# # for pid in df['pedestrian_id'].unique():
# #     ped_df = df[df['pedestrian_id'] == pid].sort_values(by='frame')
# #     vel = ped_df[['X', 'Y', 'Z']].diff().fillna(0)
# #     df.loc[vel.index, ['vx', 'vy', 'vz']] = vel.values

# # sigma = 1.0
# # impact_records = []

# # for frame in sorted(df['frame'].unique()):
# #     frame_data = df[df['frame'] == frame]
# #     for (i1, p1), (i2, p2) in combinations(frame_data.iterrows(), 2):
# #         pos1 = np.array([p1.X, p1.Y, p1.Z])
# #         pos2 = np.array([p2.X, p2.Y, p2.Z])
# #         vel1 = np.array([p1.vx, p1.vy, p1.vz])
# #         vel2 = np.array([p2.vx, p2.vy, p2.vz])
# #         rel_pos = pos2 - pos1
# #         rel_vel = vel2 - vel1

# #         distance = np.linalg.norm(rel_pos) + 1e-6
# #         if distance > 0:
# #             direction = rel_pos / distance
# #             impact_factor = np.dot(rel_vel, direction) * np.exp(-distance / sigma)

# #             impact_records.append({
# #                 'frame': frame,
# #                 'id1': p1.pedestrian_id,
# #                 'id2': p2.pedestrian_id,
# #                 'x1': p1.X, 'y1': p1.Y, 'z1': p1.Z,
# #                 'x2': p2.X, 'y2': p2.Y, 'z2': p2.Z,
# #                 'impact_factor': impact_factor,
# #                 'dx': direction[0],
# #                 'dy': direction[1],
# #                 'dz': direction[2]
# #             })

# # impact_df = pd.DataFrame(impact_records)
# # impact_df.to_csv("impact_factors.csv", index=False)


# # ==============================
# import pandas as pd
# import numpy as np

# # Load and parse the timestamp
# df = pd.read_csv("trajectories.csv")
# df["timestamp"] = pd.to_datetime(df["timestamp"])

# # Sort by id and timestamp
# df.sort_values(by=["id", "timestamp"], inplace=True)

# # Compute position differences
# df["x_diff"] = df.groupby("id")["x"].diff()
# df["y_diff"] = df.groupby("id")["y"].diff()

# # Compute time difference in seconds
# df["dt"] = df.groupby("id")["timestamp"].diff().dt.total_seconds()

# # Compute velocity components
# df["v_x"] = df["x_diff"] / df["dt"]
# df["v_y"] = df["y_diff"] / df["dt"]

# # Compute total speed (magnitude of velocity vector)
# df["velocity"] = np.sqrt(df["v_x"]**2 + df["v_y"]**2)

# print(df[["id", "timestamp", "x", "y", "v_x", "v_y", "velocity"]].head())

# import matplotlib.pyplot as plt

# # Unique IDs (each person)
# unique_ids = df["id"].unique()

# # Create subplots: one figure for each of v_x, v_y, and velocity
# fig_vx, ax_vx = plt.subplots()
# fig_vy, ax_vy = plt.subplots()
# fig_v, ax_v = plt.subplots()

# # Loop through each person and plot their velocity components
# for person_id in unique_ids:
#     person_df = df[df["id"] == person_id]

#     ax_vx.plot(person_df["timestamp"], person_df["v_x"], label=f"ID {person_id}")
#     ax_vy.plot(person_df["timestamp"], person_df["v_y"], label=f"ID {person_id}")
#     ax_v.plot(person_df["timestamp"], person_df["velocity"], label=f"ID {person_id}")

# # Customize vx plot
# ax_vx.set_title("v_x over time")
# ax_vx.set_xlabel("Timestamp")
# ax_vx.set_ylabel("v_x (units/s)")
# ax_vx.legend()

# # Customize vy plot
# ax_vy.set_title("v_y over time")
# ax_vy.set_xlabel("Timestamp")
# ax_vy.set_ylabel("v_y (units/s)")
# ax_vy.legend()

# # Customize velocity plot
# ax_v.set_title("Speed (|v|) over time")
# ax_v.set_xlabel("Timestamp")
# ax_v.set_ylabel("Speed (units/s)")
# ax_v.legend()

# # Show plots
# plt.show()



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

# Calculate relative velocity vector components (use smoothed velocity if preferred)
v_x_rel = df_2["v_x_smooth"] - df_1["v_x_smooth"]
v_y_rel = df_2["v_y_smooth"] - df_1["v_y_smooth"]

# Compute magnitudes
r_norm = np.sqrt(r_x**2 + r_y**2)
v_norm = np.sqrt(v_x_rel**2 + v_y_rel**2)

# Dot product of relative position and relative velocity
dot_rv = r_x * v_x_rel + r_y * v_y_rel

# Cosine of the angle theta
cos_theta = dot_rv / (r_norm * v_norm)

# Handle divide-by-zero or NaNs in cos_theta
cos_theta = cos_theta.fillna(0)

# Compute Impact Factor
impact_factor = (r_norm / v_norm) * cos_theta

# Handle infinite values where velocity norm might be zero
impact_factor.replace([np.inf, -np.inf], 0, inplace=True)

# Add to df_1 or create new df for results if needed
df_1["impact_factor"] = impact_factor

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


# plt.figure(figsize=(10, 6))
# plt.plot(df_1["timestamp"], df_1["impact_factor"], label="Impact Factor (ID 1 & 2)")

# plt.title("Impact Factor Over Time Between Individuals 1 and 2")
# plt.xlabel("Timestamp")
# plt.ylabel("Impact Factor")
# plt.legend()
# plt.grid(True)
# plt.tight_layout()

# # Plotting for selected IDs
# selected_ids = [1, 2]

# # Create subplots for velocity
# fig_vx, ax_vx = plt.subplots()
# fig_vy, ax_vy = plt.subplots()
# fig_v, ax_v = plt.subplots()

# # Create subplots for acceleration
# fig_ax, ax_ax = plt.subplots()
# fig_ay, ax_ay = plt.subplots()
# fig_a, ax_a = plt.subplots()

# for person_id in selected_ids:
#     person_df = df[df["id"] == person_id]

#     # Velocity plots (smoothed)
#     ax_vx.plot(person_df["timestamp"], person_df["v_x_smooth"], label=f"ID {person_id}")
#     ax_vy.plot(person_df["timestamp"], person_df["v_y_smooth"], label=f"ID {person_id}")
#     ax_v.plot(person_df["timestamp"], person_df["velocity_smooth"], label=f"ID {person_id}")

#     # Acceleration plots (smoothed)
#     ax_ax.plot(person_df["timestamp"], person_df["a_x_smooth"], label=f"ID {person_id}")
#     ax_ay.plot(person_df["timestamp"], person_df["a_y_smooth"], label=f"ID {person_id}")
#     ax_a.plot(person_df["timestamp"], person_df["acceleration_smooth"], label=f"ID {person_id}")



# # Velocity plot customization
# ax_vx.set_title("Smoothed v_x over time (IDs 1, 2)")
# ax_vx.set_xlabel("Timestamp")
# ax_vx.set_ylabel("v_x (units/s)")
# ax_vx.legend()

# ax_vy.set_title("Smoothed v_y over time (IDs 1, 2)")
# ax_vy.set_xlabel("Timestamp")
# ax_vy.set_ylabel("v_y (units/s)")
# ax_vy.legend()

# ax_v.set_title("Smoothed Speed (|v|) over time (IDs 1, 2)")
# ax_v.set_xlabel("Timestamp")
# ax_v.set_ylabel("Speed (units/s)")
# ax_v.legend()

# # Acceleration plot customization
# ax_ax.set_title("Smoothed a_x over time (IDs 1, 2)")
# ax_ax.set_xlabel("Timestamp")
# ax_ax.set_ylabel("a_x (units/s²)")
# ax_ax.legend()

# ax_ay.set_title("Smoothed a_y over time (IDs 1, 2)")
# ax_ay.set_xlabel("Timestamp")
# ax_ay.set_ylabel("a_y (units/s²)")
# ax_ay.legend()

# ax_a.set_title("Smoothed Acceleration magnitude over time (IDs 1, 2)")
# ax_a.set_xlabel("Timestamp")
# ax_a.set_ylabel("|a| (units/s²)")
# ax_a.legend()

# # Show all plots
# plt.show()

