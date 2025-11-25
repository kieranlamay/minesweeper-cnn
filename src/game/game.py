import numpy as np
import gym
from gym import spaces
import pygame
from src.constants import *
from src.game.init_game import init_game, load_assets


class MinesweeperEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array", "human"]}

    def __init__(self, rows=6, cols=6, mines=4, mode="human"):
        """
        Creates a minesweeper game.

        Parameters
        ----
        rows:   int     num of board's rows
        cols:   int     num of board's cols
        mines:  int     num of mines on the board
        mode:   str     render mode, "human" or "rgb_array"

        Returns
        ----
        None
        """

        self.num_rows = rows
        self.num_cols = cols
        self.num_mines = mines
        self.board = np.zeros((self.num_rows, self.num_cols), dtype=np.int32)
        self.place_mines()
        self.player_board = np.ones((self.num_rows, self.num_cols), dtype=np.int32) * BrickType.UNKNOWN
        self.observation_space = spaces.Box(low=BrickType.MINE.value, high=BrickType.FLAG.value,
                                            shape=(self.num_rows, self.num_cols), dtype=np.int32)
        self.action_space = spaces.Discrete(self.num_rows * self.num_cols)
        self.reward = None
        self.done = False
        self.info = dict()
        self.first_move = True  # To ensure first move is safe
        self.render_mode = mode
        self.stats = {
            'n_win': 0,
            'n_lose': 0,
            'n_progress': 0,
            'n_no_progress': 0,
            'n_guess': 0
        }
        self.button = 'left'
        self.screen = None
        self.smallFont = None
        self.mediumFont = None
        self.largeFont = None
        self.flag_image = None
        self.mine_image = None
        self.reset_button = None
        self.reset_text = None
        self.reset_text_rect = None
        if self.render_mode == "human":
            self.screen = init_game()
            
            # Load assets, set up text and buttons
            self.smallFont, self.mediumFont, self.largeFont, self.flag_image, self.mine_image = load_assets()
            self.reset_button = pygame.Rect(WINDOW_WIDTH - 5 * BOARD_PADDING - 50, BOARD_PADDING, 100, 40)
            self.reset_text = self.mediumFont.render("reset", True, BLACK)
            self.reset_text_rect = self.reset_text.get_rect(center=self.reset_button.center)


    def reset(self):
        """
        Reset a new game episode.

        Parameters
        ----
        See gym.Env.reset()

        Returns
        ----
        state:  np.array    the initial state of the player's board after reset.
        info:   dict        additional game information.
        """
        self.board = np.zeros((self.num_rows, self.num_cols), dtype=np.int32)
        self.place_mines()
        self.player_board = np.ones((self.num_rows, self.num_cols), dtype=np.int32) * BrickType.UNKNOWN
        self.reward = None
        self.done = False
        self.info = dict()
        self.first_move = True  # To ensure first move is safe
        for key in self.stats:
            self.stats[key] = 0
        self.button = 'left'
        return self.player_board, self.info


    def step(self, action):
        """
        Take an action in the game environment.

        Parameters
        ----
        action:     np.array    the 1D location on the board where the player clicks.
        
        Returns
        ----
        next_state: np.array    the state of the player's board after taking the action.
        reward:     float       the reward received after taking the action.
        done:       bool        a flag indicating whether the game has ended.
        info:       dict        additional game information.
        """
        x, y = divmod(action, self.num_cols)

        # Ensure first move is safe
        if self.first_move and self.button == 'left':  
            if self.board[x, y] == BrickType.MINE:     
                while self.board[x, y] == BrickType.MINE:  
                    self.board = np.zeros((self.num_rows, self.num_cols), dtype=np.int32)
                    self.place_mines()
            self.first_move = False
        
        # Handle left click action
        if self.button == 'left' and not self.done:

            # No progress if clicked on a previously revealed brick
            if not self.is_brick_unknown(x, y) or self.is_brick_flagged(x, y):
                self.reward = -0.5
                self.done = False
                self.info["button"] = "left"
                self.info["status"] = "no_progress"
                self.update_stats("n_no_progress")
            else:
                guess = True if (self.count_neighbor_unknowns(x, y) == 8) else False
                self.player_board[x, y] = self.board[x, y]

                # Recursively open neighbor bricks if this brick is EMPTY
                if self.player_board[x, y] == BrickType.EMPTY:
                    self.open_neighbor_bricks(x, y)

                # Check if the player has won or lost
                status = self.check_game_status()
                if status == GameState.WIN:
                    self.reward = self.num_rows * self.num_cols
                    self.done = True
                    self.info["button"] = "left"
                    self.info["status"] = "win"
                    self.update_stats("n_win")
                elif status == GameState.LOSE:
                    self.reward = -self.num_rows * self.num_cols
                    self.done = True
                    self.info["button"] = "left"
                    self.info["status"] = "lose"
                    self.update_stats("n_lose")
                
                # Reward player for not guessing randomly
                elif not guess:
                    self.reward = 1.0
                    self.done = False
                    self.info["button"] = "left"
                    self.info["status"] = "progress"
                    self.update_stats("n_progress")
                else:
                    self.reward = -0.5
                    self.done = False
                    self.info["button"] = "left"
                    self.info["status"] = "guess"
                    self.update_stats("n_guess")
        
        # Handle right click action, for flagging/unflagging bricks
        elif self.button == 'right' and not self.done:
            if not self.is_brick_unknown(x, y) and not self.is_brick_flagged(x, y):
                self.reward = 0.0
                self.done = False
                self.info["button"] = "right"
                self.info["status"] = "playing"
            elif self.is_brick_flagged(x, y):
                self.player_board[x, y] = BrickType.UNKNOWN
                self.reward = 0.0
                self.done = False
                self.info["button"] = "right"
                self.info["status"] = "playing"
            else:
                self.player_board[x, y] = BrickType.FLAG
                self.reward = 0.0
                self.done = False
                self.info["button"] = "right"
                self.info["status"] = "playing"
        else:
            self.reward = None
            self.done = True
            self.info = {}
        return self.player_board, self.reward, self.done, self.info


    def render(self, mode='human'):
        """
        Render the current state of the game.

        Depending on the selected render mode, this method either draws the game
        on the screen (for human players) or prepares the game state for
        rendering as an RGB array (for potential machine learning applications).

        Supported render modes:
        - 'human': Displays the game board in a window using Pygame.
        - 'rgb_array': Prepares the game state as an RGB array (currently not implemented).

        Raises
        ------
        ValueError
            If an unsupported render mode is specified.
        """
        if self.render_mode == 'human':
            self.draw_board()
            pygame.display.flip()
        elif self.render_mode == 'rgb_array':
            pass
        else:
            raise ValueError("Invalid render mode. Supported modes are 'human' and 'rgb_array'.")


    def is_brick_unknown(self, x, y):
        """ Returns true if this is not an already clicked brick. """
        return self.player_board[x, y] == BrickType.UNKNOWN


    def is_brick_flagged(self, x, y):
        """ Returns true if this is a flagged brick. """
        return self.player_board[x, y] == BrickType.FLAG


    def is_valid_coordinate(self, x, y):
        """ Returns if the coordinate is valid. """
        return 0 <= x < self.num_rows and 0 <= y < self.num_cols


    def check_game_status(self):
        """ Check the current game status: WIN, LOSE, or PLAYING. """

        # Lose if any mine is revealed
        if np.count_nonzero(self.player_board == BrickType.MINE) > 0:
            return GameState.LOSE

        # Win if all non-mine cells are revealed
        elif np.count_nonzero(self.player_board == BrickType.UNKNOWN) + \
                np.count_nonzero(self.player_board == BrickType.FLAG) == self.num_mines:
            return GameState.WIN
        
        # Otherwise, the game is still ongoing
        else:
            return GameState.PLAYING


    def count_neighbor_mines(self, x, y):
        """ Returns the number of mines in neighbor cells given an x-y coordinate. """
        
        # Loop over neighboring cells
        neighbor_mines = 0
        for a in range(x - 1, x + 2):
            for b in range(y - 1, y + 2):

                # Skip center or invalid coordinates
                if (a, b) != (x, y) and self.is_valid_coordinate(a, b):
                    if self.board[a, b] == BrickType.MINE:
                        neighbor_mines += 1
        return neighbor_mines


    def open_neighbor_bricks(self, x, y):
        """ Recursively opens neighbor bricks if they are EMPTY. """

        # Loop over neighboring cells
        for a in range(x - 1, x + 2):
            for b in range(y - 1, y + 2):

                # Skip center or invalid coordinates
                if (a, b) != (x, y) and self.is_valid_coordinate(a, b):
                    if self.is_brick_unknown(a, b):
                        self.player_board[a, b] = self.board[a, b]

                        # Recursively open neighbors if this brick is also EMPTY
                        if self.player_board[a, b] == BrickType.EMPTY:
                            self.open_neighbor_bricks(a, b)


    def count_neighbor_unknowns(self, x, y):
        """ Returns the number of UNKNOWN cells in neighbor cells given an x-y coordinate. """

        # Loop over neighboring cells
        neighbor_unknowns = 0
        for a in range(x - 1, x + 2):
            for b in range(y - 1, y + 2):
                
                # Skip center
                if (a, b) == (x, y):
                    continue
                
                # Count unknowns
                if self.is_valid_coordinate(a, b):
                    if self.player_board[a, b] == BrickType.UNKNOWN:
                        neighbor_unknowns += 1
                else:
                    neighbor_unknowns += 1
        return neighbor_unknowns


    def place_mines(self):
        """ Generates a board and places mines randomly. """

        # Place mines
        mines_placed = 0
        while mines_placed < self.num_mines:
            i = np.random.randint(0, self.num_rows)
            j = np.random.randint(0, self.num_cols)
            if self.board[i, j] != BrickType.MINE:
                self.board[i, j] = BrickType.MINE
                mines_placed += 1

        # Calculate numbers for non-mine cells
        for i in range(self.num_rows):
            for j in range(self.num_cols):
                if self.board[i, j] != BrickType.MINE:
                    self.board[i, j] = self.count_neighbor_mines(i, j)


    def draw_board(self):
        """ Draw the game board using Pygame. """
        
        # Initialize drawing parameters and loop over board cells
        self.screen.fill(BLACK)
        for i in range(self.num_rows):
            for j in range(self.num_cols):

                # Draw each brick based on its state
                rect = pygame.Rect(
                    BOARD_ORIGIN[0] + j * BRICK_SIZE,
                    BOARD_ORIGIN[1] + i * BRICK_SIZE,
                    BRICK_SIZE, BRICK_SIZE
                )
                pygame.draw.rect(self.screen, GRAY, rect)
                pygame.draw.rect(self.screen, WHITE, rect, 3)

                # Draw mine or flag as necessary
                if self.player_board[i, j] == BrickType.MINE:
                    self.screen.blit(self.mine_image, rect)
                elif self.player_board[i, j] == BrickType.FLAG:
                    self.screen.blit(self.flag_image, rect)
                
                # Draw revealed number as necessary
                elif BrickType.EMPTY <= self.player_board[i, j] <= BrickType.EIGHT:
                    revealed_text = self.smallFont.render(str(self.player_board[i, j]), True, BLACK)
                    revealed_text_rect = revealed_text.get_rect(center=rect.center)
                    self.screen.blit(revealed_text, revealed_text_rect)
        
        # Draw reset button and status panel
        pygame.draw.rect(self.screen, WHITE, self.reset_button)
        self.screen.blit(self.reset_text, self.reset_text_rect)
        self.update_status_panel()


    def update_status_panel(self):
        """ Update the status panel showing current game status. """

        # Check game status
        text = self.info.get("status", "playing").capitalize() if not self.done else \
               ("Win" if self.check_game_status() == GameState.WIN else "Lose")
        color = GREEN if (self.done and text == "Win") \
            else (RED if self.done and text == "Lose" else GRAY)
        text = "Status: " + text
        
        # Render status text with appropriate color
        status_text = self.smallFont.render(text, True, color)
        status_text_rect = status_text.get_rect(center=(WINDOW_WIDTH - 5 * BOARD_PADDING,
                                                        self.reset_button.height + 2 * BOARD_PADDING))
        self.screen.blit(status_text, status_text_rect)


    def update_stats(self, stat_key: str):
        """ Update game statistics given a stat key. """

        if stat_key in self.stats:
            self.stats[stat_key] += 1
        else:
            raise KeyError(f"Invalid stat_key: {stat_key}. Must be one of {list(self.stats.keys())}.")