import pygame
import os
from game.game import MinesweeperEnv
from constants import BOARD_ORIGIN, BRICK_SIZE
from training.trainRL import Agent
from models.cnn import CNN
wandb = None
try:
    WANDB_MODE = os.environ.get("WANDB_MODE", "disabled")
    if WANDB_MODE != "disabled":
        import wandb
        wandb.init(project="minesweeper-cnn-rl", mode=WANDB_MODE)
    else:
        wandb = None
except Exception:
    wandb = None


def main():
    env = MinesweeperEnv(mode="rgb_array")
    agent = Agent(env, model=CNN(), num_episodes=30, epochs=10)  
    obs, info = env.reset()

    num_updates = 500  
    update_count = 0

    print(f"Starting RL training with {num_updates} updates...")
    print(f"Each update collects {agent.num_episodes} episodes and trains for {agent.epochs} epochs\n")

    while update_count < num_updates:
        agent.play()

        if len(agent.episodes) == 0 and agent.current_episode == 0:
            update_count += 1
            print(f"Completed update {update_count}/{num_updates}.")
            if update_count % 100 == 0 or update_count == num_updates:
                agent.validate()
    try:
        if wandb is not None:
            wandb.finish()
    except Exception:
        pass

    print("\ntraining complete")


if __name__ == "__main__":
    main()

