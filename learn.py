import matplotlib.pyplot as plt
from train import PPO
from env import Restaurant

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

res = Restaurant(room_width, room_height, tables, v=0.1, p=0.03)

ppo = PPO(res, 1000)

objectives = ppo.learn()

objective_vals = [obj.item() for obj in objectives]

plt.plot(objective_vals, marker='o', label='Objectives')

# Add labels and a title
plt.xlabel('Index')
plt.ylabel('Objective Value')
plt.title('Objective Values Over Time')

# Add a legend
plt.legend()

# Show the plot
plt.show()
