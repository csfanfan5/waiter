from env2 import Restaurant
import argparse 
from classes import PPO 

tables = [
    [1, 4, 4, 6],   # Top-left
    [5, 8, 4, 6],   # Top-center
    [9, 12, 4, 6],  # Top-right
    [1, 4, 1, 3],   # Bottom-left
    [5, 8, 1, 3],   # Bottom-center
    [9, 12, 1, 3],  # Bottom-right
]

room_width = 13  # Width
room_height = 7  # Height

def get_args():
    parser = argparse.ArgumentParser(description="Run PPO with specified parameters.")
    parser.add_argument("--advantage", action="store_true",
                        help="Use advantage if this flag is set. Otherwise, no advantage.")
    parser.add_argument("--pbatches", type=int, required=True,
                        help="Number of policy batches to process per optimization step.")
    parser.add_argument("--pbatchsize", type=int, required=True,
                        help="Number of samples per batch.")
    parser.add_argument("--learning_steps", type=int, required=True,
                        help="Number of learning steps to run.")
    parser.add_argument("--lr", type=float, default=3e-4,
                        help="Learning rate for the optimizer.")
    parser.add_argument("--lamb", type=float, default=0.1,
                        help="Lambda parameter for PPO.")
    parser.add_argument("--save_path", type=str, required=True,
                        help="Path to save model weights.")

    args = parser.parse_args()

    print("Arguments received:")
    print(f"advantage: {args.advantage}")
    print(f"pbatches: {args.pbatches}")
    print(f"pbatchsize: {args.pbatchsize}")
    print(f"learning_steps: {args.learning_steps}")
    print(f"lr: {args.lr}")
    print(f"lamb: {args.lamb}")
    print(f"save_path: {args.save_path}")

    return args 

if __name__ == "__main__":
    print("Starting...")
    args = get_args() 
    restaurant = Restaurant(room_width, room_height, tables, v=5, p=0.001, table_waiting_time=300, table_threshold=10, time_reward=0.001)

    ppo = PPO(restaurant, args.advantage, args.pbatches, args.pbatchsize, args.learning_steps, args.lr, args.lamb)
    ppo.learn() 
    print("Done learning!")
    ppo.save_nns(args.save_path)
