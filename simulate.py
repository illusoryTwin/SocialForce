import cv2
import numpy as np
import time

# Example trajectories (replace these with your actual tracked data)
trajectories = {
    1: [(1125, 373), (1123, 390), (1122, 378), (1119, 390), (1115, 402), (1109, 409), (1105, 397), (1109, 407), (1131, 397), (1141, 404), (1122, 408), (1115, 409), (1113, 407), (1113, 404), (1130, 405), (1129, 390), (1126, 403), (1120, 406)],
    2: [(808, 552), (818, 552), (834, 554), (843, 555), (851, 555), (860, 555), (866, 558), (872, 555), (875, 554), (878, 553), (885, 548), (891, 547), (906, 549), (911, 550), (912, 550), (922, 550), (930, 553), (934, 553)],
    6: [(748, 547), (760, 549), (765, 551), (770, 551), (778, 552), (783, 552), (790, 548), (806, 550), (815, 548), (826, 546), (829, 541), (829, 534), (834, 541), (837, 543), (841, 545), (850, 544), (855, 544)]
}

# Create a blank canvas (adjust resolution to your dataset)
canvas_height, canvas_width = 720, 1280
canvas = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)

# Assign colors to each ID
np.random.seed(42)
colors = {track_id: tuple(np.random.randint(0, 255, 3).tolist()) for track_id in trajectories.keys()}

# Find the maximum trajectory length
max_len = max(len(points) for points in trajectories.values())

# Animate point movement
for i in range(max_len):
    frame = canvas.copy()

    for track_id, points in trajectories.items():
        color = colors[track_id]

        # Draw all previous points
        for j in range(1, min(i + 1, len(points))):
            cv2.line(frame, points[j - 1], points[j], color, 2)

        # Draw current position
        if i < len(points):
            cv2.circle(frame, points[i], 5, color, -1)
            cv2.putText(frame, f"ID {track_id}", (points[i][0] + 10, points[i][1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    # Show animation
    cv2.imshow("Trajectory Animation", frame)
    if cv2.waitKey(150) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
