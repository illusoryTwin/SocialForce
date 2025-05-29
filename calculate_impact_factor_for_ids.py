from calculate_impact_factor_acc_approach import calculate_impact_factor_by_acc
from run_simulation import simulate

KEY_ID = 0
RELATIVE_IDS = [1, 3]

impact_factors = []

for i in range(len(RELATIVE_IDS)):
    impact_factor = calculate_impact_factor_by_acc(selected_id_pair=[KEY_ID, RELATIVE_IDS[i]])
    impact_factors.append(impact_factor)
    # simulate(key_id=0, relative_id=RELATIVE_IDS[i], vector_magnitude=impact_factor)
print(impact_factors)


# from calculate_impact_factor_acc_approach import calculate_impact_factor_by_acc
# from run_simulation import simulate
# import matplotlib.pyplot as plt
# import numpy as np

# KEY_ID = 0
# RELATIVE_IDS = [1, 3]

# impact_factors = []
# all_vectors = {}

# for rel_id in RELATIVE_IDS:
#     impact_factor = calculate_impact_factor_by_acc(selected_id_pair=[KEY_ID, rel_id])
#     impact_factors.append(impact_factor)

#     vectors = simulate(key_id=KEY_ID, relative_id=rel_id, vector_magnitude=impact_factor, return_vectors=True)
#     all_vectors[rel_id] = vectors

# # -------- Plot vector magnitude over time --------
# plt.figure(figsize=(10, 6))
# for rel_id, vectors in all_vectors.items():
#     frames, dxs, dys = zip(*vectors)
#     magnitudes = np.sqrt(np.array(dxs)**2 + np.array(dys)**2)
#     plt.plot(frames, magnitudes, label=f"ID {rel_id}")

# plt.xlabel("Frame")
# plt.ylabel("Vector Magnitude")
# plt.title(f"Vector Magnitudes Over Time (KEY_ID = {KEY_ID})")
# plt.legend()
# plt.grid(True)
# plt.tight_layout()
# plt.show()

# # -------- Plot quiver (directional vectors) --------
# plt.figure(figsize=(8, 8))
# for rel_id, vectors in all_vectors.items():
#     _, dxs, dys = zip(*vectors)
#     plt.quiver([0]*len(dxs), [0]*len(dys), dxs, dys,
#                angles='xy', scale_units='xy', scale=1,
#                label=f'ID {rel_id}', alpha=0.7)

# plt.title(f"Direction Vectors from ID {KEY_ID}")
# plt.xlabel("dx")
# plt.ylabel("dy")
# plt.axhline(0, color='gray', lw=1)
# plt.axvline(0, color='gray', lw=1)
# plt.grid(True)
# plt.legend()
# plt.axis("equal")
# plt.show()
