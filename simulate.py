# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# from matplotlib.animation import FuncAnimation


# # Load and preprocess
# df = pd.read_csv("trajectories.csv")
# df["timestamp"] = pd.to_datetime(df["timestamp"])
# df.sort_values(by=["timestamp", "id"], inplace=True)

# # Normalize timestamps to integers for frame mapping
# df["time_index"] = (df["timestamp"] - df["timestamp"].min()).dt.total_seconds().astype(int)

# # Unique time steps and IDs
# time_steps = sorted(df["time_index"].unique())
# all_ids = df["id"].unique()

# # Color map for IDs
# colors = plt.cm.get_cmap("tab10", len(all_ids))
# id_color = {id_: colors(i) for i, id_ in enumerate(all_ids)}

# # Prepare the plot
# fig, ax = plt.subplots(figsize=(8, 6))
# scat = ax.scatter([], [], s=40)
# text_labels = []  # Store text elements

# # Axis limits
# margin = 1
# x_min, x_max = df["x"].min() - margin, df["x"].max() + margin
# y_min, y_max = df["y"].min() - margin, df["y"].max() + margin
# ax.set_xlim(x_min, x_max)
# ax.set_ylim(y_min, y_max)
# ax.set_title("Dynamic Trajectory Simulation with IDs")
# ax.set_xlabel("X")
# ax.set_ylabel("Y")

# # Timestamp display
# time_text = ax.text(0.02, 0.95, "", transform=ax.transAxes)

# # Animation update function
# def update(frame_index):
#     global text_labels
#     t = time_steps[frame_index]
#     current_data = df[df["time_index"] == t]
#     positions = np.column_stack((current_data["x"], current_data["y"]))
#     colors_list = [id_color[id_] for id_ in current_data["id"]]
#     scat.set_offsets(positions)
#     scat.set_color(colors_list)

#     # Update time text
#     time_label = current_data["timestamp"].iloc[0].strftime("%Y-%m-%d %H:%M:%S")
#     time_text.set_text(f"Time: {time_label}")

#     # Remove old text labels
#     for txt in text_labels:
#         txt.remove()
#     text_labels = []

#     # Add new text labels
#     for _, row in current_data.iterrows():
#         txt = ax.text(row["x"] + 0.2, row["y"] + 0.2, str(row["id"]), fontsize=9)
#         text_labels.append(txt)

#     return scat, time_text, *text_labels

# # Create animation
# ani = FuncAnimation(fig, update, frames=len(time_steps), interval=100, blit=False)

# plt.show()


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import FancyArrowPatch

def simulate(key_id=0, relative_id=1, vector_magnitude=2.0):
        
    # Load and preprocess
    df = pd.read_csv("trajectories.csv")
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df.sort_values(by=["timestamp", "id"], inplace=True)

    # Normalize timestamps to integers for frame mapping
    df["time_index"] = (df["timestamp"] - df["timestamp"].min()).dt.total_seconds().astype(int)

    # Unique time steps and IDs
    time_steps = sorted(df["time_index"].unique())
    all_ids = df["id"].unique()

    # Color map for IDs
    colors = plt.cm.get_cmap("tab10", len(all_ids))
    id_color = {id_: colors(i) for i, id_ in enumerate(all_ids)}

    # Prepare the plot
    fig, ax = plt.subplots(figsize=(8, 6))
    scat = ax.scatter([], [], s=40)
    text_labels = []  # Store text elements
    vector_arrow = None  # Handle for the vector arrow

    # Axis limits
    margin = 1
    x_min, x_max = df["x"].min() - margin, df["x"].max() + margin
    y_min, y_max = df["y"].min() - margin, df["y"].max() + margin
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_title("Dynamic Trajectory Simulation with IDs and Vector")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")

    # Timestamp display
    time_text = ax.text(0.02, 0.95, "", transform=ax.transAxes)

    # Animation update function
    def update(frame_index):
        nonlocal text_labels, vector_arrow  # Fix: allow modification from enclosing scope
        t = time_steps[frame_index]
        current_data = df[df["time_index"] == t]
        positions = np.column_stack((current_data["x"], current_data["y"]))
        colors_list = [id_color[id_] for id_ in current_data["id"]]
        scat.set_offsets(positions)
        scat.set_color(colors_list)

        # Update time text
        time_label = current_data["timestamp"].iloc[0].strftime("%Y-%m-%d %H:%M:%S")
        time_text.set_text(f"Time: {time_label}")

        # Remove old text labels
        for txt in text_labels:
            txt.remove()
        text_labels = []

        # Add new text labels
        for _, row in current_data.iterrows():
            txt = ax.text(row["x"] + 0.2, row["y"] + 0.2, str(row["id"]), fontsize=9)
            text_labels.append(txt)

        # Remove previous vector
        if vector_arrow is not None:
            vector_arrow.remove()

        # Draw vector if both key_id and relative_id are present at this time
        key_row = current_data[current_data["id"] == key_id]
        rel_row = current_data[current_data["id"] == relative_id]

        if not key_row.empty and not rel_row.empty:
            x0, y0 = key_row.iloc[0][["x", "y"]]
            x1, y1 = rel_row.iloc[0][["x", "y"]]
            dx, dy = x1 - x0, y1 - y0
            norm = np.sqrt(dx**2 + dy**2)
            if norm > 0:
                dx_scaled = (dx / norm) * vector_magnitude
                dy_scaled = (dy / norm) * vector_magnitude
                vector_arrow = FancyArrowPatch((x0, y0), (x0 + dx_scaled, y0 + dy_scaled),
                                               arrowstyle='->', color='red', mutation_scale=15)
                ax.add_patch(vector_arrow)
            else:
                vector_arrow = None

        return scat, time_text, *text_labels

    # Create animation
    ani = FuncAnimation(fig, update, frames=len(time_steps), interval=100, blit=False)

    plt.show()

# simulate()