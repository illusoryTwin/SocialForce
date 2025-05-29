from calculate_impact_factor_acc_approach import calculate_impact_factor_by_acc
from simulate import simulate

KEY_ID = 0
RELATIVE_IDS = [1, 3]

impact_factors = []

for i in range(len(RELATIVE_IDS)):
    impact_factor = calculate_impact_factor_by_acc(selected_id_pair=[KEY_ID, RELATIVE_IDS[i]])
    impact_factors.append(impact_factor)

simulate()
print(impact_factors)
