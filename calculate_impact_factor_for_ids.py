from calculate_impact_factor_acc_approach import calculate_impact_factor_by_acc
from calculate_impact_factor_vel_approach import calculate_impact_factor_by_vel
from run_simulation import simulate

KEY_ID = 1
RELATIVE_IDS = [0, 2, 3]

# KEY_ID = 2
# RELATIVE_IDS = [0, 1, 3]

impact_factors = []

for i in range(len(RELATIVE_IDS)):
    impact_factor_csv_path = calculate_impact_factor_by_vel(selected_id_pair=[KEY_ID, RELATIVE_IDS[i]])
    # impact_factor_csv_path = calculate_impact_factor_by_acc(selected_id_pair=[KEY_ID, RELATIVE_IDS[i]])
    simulate(impact_factor_csv_path, key_id=KEY_ID, relative_id=RELATIVE_IDS[i])
