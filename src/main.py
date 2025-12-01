import pygame
from game.game import MinesweeperEnv
from constants import BOARD_ORIGIN, BRICK_SIZE
from training.train import Agent
from models.cnn import CNN


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
    agent = Agent(env, model=CNN())
    obs, info = env.reset()

    num_updates = 50
    update_count = 0

    while update_count < num_updates:
        agent.play()

        if agent.current_sample == 0:
            update_count += 1
            print(f"Completed update {update_count}/{num_updates}.")
            agent.validate()

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
