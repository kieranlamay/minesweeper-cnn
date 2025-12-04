import pygame
import os
from game.game import MinesweeperEnv
from constants import BOARD_ORIGIN, BRICK_SIZE
from training.train import Agent
from models.cnn import CNN

# Optional WandB init — run in offline mode by default
try:
    import wandb
    WANDB_MODE = os.environ.get("WANDB_MODE", "offline")
    wandb.init(project="minesweeper-cnn", mode=WANDB_MODE)
except Exception:
    wandb = None


# def coord_to_action(x, y, env):
#     """Convert grid coordinates to environment action."""
#     return x * env.num_cols + y


# def get_cell_from_mouse(pos, env):
#     """Convert pixel mouse position → board (x, y)."""

#     mx, my = pos
#     bx0, by0 = BOARD_ORIGIN

#     # Convert pixel -> cell
#     y = (mx - bx0) // BRICK_SIZE
#     x = (my - by0) // BRICK_SIZE

#     if env.is_valid_coordinate(x, y):
#         return x, y
#     return None


def main():

    # Create environment
    env = MinesweeperEnv(mode="human")
    agent = Agent(env, model=CNN(), num_samples=640, batch_size=128, epochs=20)
    env.reset()

    # Initial training with random data
    num_updates_random_data = 10
    update_count = 0
    while update_count < num_updates_random_data:
        # Initial training with data loading
        agent.train(load_data=True)
        agent.validate()
        update_count += 1
        print(f"Completed update {update_count}/{num_updates_random_data}.")
            
    # Self-play training
    num_updates_self_play = 0
    update_count = 0
    while update_count < num_updates_self_play:
        # Generate new data and continue training
        agent.play()
        if agent.current_sample == 0:
            agent.validate()
            update_count += 1
            print(f"Completed self-play update {update_count}/{num_updates_self_play}.")

    # finish wandb run if active
    try:
        if wandb is not None:
            wandb.finish()
    except Exception:
        pass

    # pygame.init()
    # running = True
    # clock = pygame.time.Clock()

    # while running:
    #     clock.tick(60)
    #     env.render()

    #     for event in pygame.event.get():

    #         # Close window
    #         if event.type == pygame.QUIT:
    #             running = False
    #             break

    #         # Mouse click handling
    #         if event.type == pygame.MOUSEBUTTONDOWN:

    #             if env.reset_button.collidepoint(event.pos):
    #                 obs, info = env.reset()
    #                 continue

    #             cell = get_cell_from_mouse(event.pos, env)
    #             if cell is None:
    #                 continue

    #             x, y = cell
    #             action = coord_to_action(x, y, env)

    #             if event.button == 1:         
    #                 env.button = 'left'
    #             elif event.button == 3:     
    #                 env.button = 'right'
    #             else:
    #                 continue

    #             obs, reward, done, info = env.step(action)

    # pygame.quit()


if __name__ == "__main__":
    main()
