import argparse
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Rectangle, FancyArrowPatch, Wedge
from matplotlib.animation import FuncAnimation
import matplotlib.colors as mcolors
import torch
from env2 import Restaurant
import numpy as np
from train2 import PPO

# Room and tables setup
room_width = 13  # Width
room_height = 7  # Height

# Adjusted 3x2 grid of tables (3x2 units each)
# Each table: [x1, x2, y1, y2]
tables = [
    [1, 4, 4, 6],   # Top-left
    [5, 8, 4, 6],   # Top-center
    [9, 12, 4, 6],  # Top-right
    [1, 4, 1, 3],   # Bottom-left
    [5, 8, 1, 3],   # Bottom-center
    [9, 12, 1, 3],  # Bottom-right
]

image_path = "./the_waiter.png"  # Replace with your PNG file path
image = plt.imread(image_path)

max_color_val = 300

def get_meter_color(value):
    if value == -1:
        return "white"  # Blank
    elif value == 0:
        return "lightgreen"
    elif value <= max_color_val:
        # Gradient from green to red
        norm = mcolors.Normalize(vmin=0, vmax=max_color_val)
        cmap = plt.cm.RdYlGn_r  # Access colormap directly
        return cmap(norm(value))
    else:
        return "black"  # Above 100

def main():
    parser = argparse.ArgumentParser(description="Simulate environment using PPO weights.")
    parser.add_argument("--weights", type=str, required=True,
                        help="Path to the model weights file.")
    args = parser.parse_args()

    # Set up the environment
    restaurant = Restaurant(room_width, room_height, tables, v=0.1, p=0.001, 
                            table_waiting_time=300, table_threshold=10, time_reward=0.001)
    # Initialize PPO but do not train, just load weights
    # The constructor arguments must match how PPO is defined. 
    # For demonstration, we pass some dummy values for pbatches, pbatchsize, learning_steps, etc.
    ppo = PPO(restaurant, advantage=False, pbatches=1, pbatchsize=1, learning_steps=1)

    # Load the provided weights
    state_dict = torch.load(args.weights, map_location=torch.device('cpu'))
    ppo.policy.load_state_dict(state_dict)

    # Now create a trajectory with the loaded policy
    states, _, rewards = ppo.create_trajectory(ppo.policy)

    # Create the figure and axis
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_xlim(0, room_width)
    ax.set_ylim(0, room_height)
    ax.set_aspect('equal', adjustable='box')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("Reward: ")

    # Add room boundary
    ax.plot([0, room_width, room_width, 0, 0], [0, 0, room_height, room_height, 0], 'k-')

    # Add tables
    table_patches = []
    arrow_patches = []
    semicircles = []
    arrow_data = []
    for table in tables:
        x1, x2, y1, y2 = table
        rect = Rectangle((x1, y1), x2 - x1, y2 - y1, edgecolor='blue', facecolor='none', zorder=1)
        ax.add_patch(rect)
        table_patches.append(rect)

        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2 
        wedge = Wedge(center=(center_x, center_y), r=0.5, theta1=0, theta2=180, facecolor='white', edgecolor='black', zorder=2)
        ax.add_patch(wedge)
        semicircles.append(wedge)

        arrow = FancyArrowPatch(
            (center_x, center_y),
            (center_x + 0.8, center_y),
            color='black',
            arrowstyle='->',
            mutation_scale=10,
            zorder=3,
        )
        ax.add_patch(arrow)
        arrow_patches.append(arrow)
        arrow_data.append((center_x, center_y))

    image_height, image_width, _ = image.shape
    image_aspect = image_width / image_height
    image_size = 0.8  # adjust size of server image

    # Initial position
    # The initial agent position is given by restaurant.agent, 
    # but after creating a trajectory we have the states array that contains the agent positions.
    # We'll use the first frame of the trajectory to initialize.
    initial_state = states[0].tolist()
    times_start_index = len(restaurant.tables) * 4 + 2
    agent = initial_state[times_start_index + len(restaurant.tables): times_start_index + len(restaurant.tables) + 2]

    agent_image = ax.imshow(
        image,
        extent=[
            agent[0] - image_size / 2,
            agent[0] + image_size / 2,
            agent[1] - image_size / (2 * image_aspect),
            agent[1] + image_size / (2 * image_aspect),
        ],
        zorder=3,
    )
    tot_reward = 0

    def update(frame):
        nonlocal tot_reward
        state = states[frame].tolist()
        times_start_index = len(restaurant.tables) * 4 + 2
        times = state[times_start_index : times_start_index + len(restaurant.tables)]
        agent = state[times_start_index + len(restaurant.tables): times_start_index + len(restaurant.tables) + 2]
        reward = rewards[frame]
        tot_reward += reward

        # set pos of agent
        ax.set_title(f"Reward: {tot_reward}")
        agent_image.set_extent([
            agent[0] - image_size / 2,
            agent[0] + image_size / 2,
            agent[1] - image_size / (2 * image_aspect),
            agent[1] + image_size / (2 * image_aspect),
        ])

        for wedge, (arrow, (cx, cy)), value in zip(semicircles, zip(arrow_patches, arrow_data), times):
            color = get_meter_color(value)
            wedge.set_facecolor(color)  # Update the semicircle color

            # Update arrow position
            arrow_len = 0.8
            if value < 1:
                theta = np.pi
            elif value <= max_color_val:
                theta = np.pi * (1 - (value / max_color_val))  # Compute angle in radians
            else:
                theta = 0
            dx = arrow_len * np.cos(theta)  # x-offset for arrow endpoint
            dy = arrow_len * np.sin(theta)  # y-offset for arrow endpoint
            arrow.set_positions((cx, cy), (cx + dx, cy + dy))  # Update arrow endpoint

        return [agent_image, ax.title] + arrow_patches + semicircles

    ani = FuncAnimation(fig, update, frames=len(states), interval=3, blit=False)
    plt.show()

if __name__ == "__main__":
    main()
