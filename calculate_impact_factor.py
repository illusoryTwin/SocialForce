# import pandas as pd
# import numpy as np
# from itertools import combinations

# # Load data
# df = pd.read_csv("projected_points.csv")

# # Compute velocities
# df[['vx', 'vy', 'vz']] = 0.0

# # Sort by pedestrian ID and frame
# df.sort_values(by=['pedestrian_id', 'frame'], inplace=True)

# # Calculate velocities per pedestrian
# for pid in df['pedestrian_id'].unique():
#     ped_df = df[df['pedestrian_id'] == pid]
#     ped_df = ped_df.sort_values(by='frame')
#     vel = ped_df[['X', 'Y', 'Z']].diff().fillna(0)
#     df.loc[vel.index, ['vx', 'vy', 'vz']] = vel.values

# # Compute interaction (impact factor) per frame
# impact_factors = []

# sigma = 1.0  # Decay parameter

# for frame in sorted(df['frame'].unique()):
#     frame_data = df[df['frame'] == frame]
#     for (i1, p1), (i2, p2) in combinations(frame_data.iterrows(), 2):
#         pos1 = np.array([p1['X'], p1['Y'], p1['Z']])
#         pos2 = np.array([p2['X'], p2['Y'], p2['Z']])
#         vel1 = np.array([p1['vx'], p1['vy'], p1['vz']])
#         vel2 = np.array([p2['vx'], p2['vy'], p2['vz']])
        
#         dist = np.linalg.norm(pos1 - pos2)
#         vel_diff = np.linalg.norm(vel1 - vel2)
        
#         impact = np.exp(-dist / sigma) / (1 + vel_diff)
        
#         impact_factors.append({
#             'frame': frame,
#             'id1': int(p1['pedestrian_id']),
#             'id2': int(p2['pedestrian_id']),
#             'impact_factor': impact
#         })

# # Save to CSV
# impact_df = pd.DataFrame(impact_factors)
# impact_df.to_csv("impact_factors.csv", index=False)


import pandas as pd
import numpy as np
from itertools import combinations

df = pd.read_csv("projected_points.csv")
df[['vx', 'vy', 'vz']] = 0.0
df.sort_values(by=['pedestrian_id', 'frame'], inplace=True)

for pid in df['pedestrian_id'].unique():
    ped_df = df[df['pedestrian_id'] == pid].sort_values(by='frame')
    vel = ped_df[['X', 'Y', 'Z']].diff().fillna(0)
    df.loc[vel.index, ['vx', 'vy', 'vz']] = vel.values

sigma = 1.0
impact_records = []

for frame in sorted(df['frame'].unique()):
    frame_data = df[df['frame'] == frame]
    for (i1, p1), (i2, p2) in combinations(frame_data.iterrows(), 2):
        pos1 = np.array([p1.X, p1.Y, p1.Z])
        pos2 = np.array([p2.X, p2.Y, p2.Z])
        vel1 = np.array([p1.vx, p1.vy, p1.vz])
        vel2 = np.array([p2.vx, p2.vy, p2.vz])
        rel_pos = pos2 - pos1
        rel_vel = vel2 - vel1

        distance = np.linalg.norm(rel_pos) + 1e-6
        if distance > 0:
            direction = rel_pos / distance
            impact_factor = np.dot(rel_vel, direction) * np.exp(-distance / sigma)

            impact_records.append({
                'frame': frame,
                'id1': p1.pedestrian_id,
                'id2': p2.pedestrian_id,
                'x1': p1.X, 'y1': p1.Y, 'z1': p1.Z,
                'x2': p2.X, 'y2': p2.Y, 'z2': p2.Z,
                'impact_factor': impact_factor,
                'dx': direction[0],
                'dy': direction[1],
                'dz': direction[2]
            })

impact_df = pd.DataFrame(impact_records)
impact_df.to_csv("impact_factors.csv", index=False)
