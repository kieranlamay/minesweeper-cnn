class BrickType(IntEnum):
    MINE = -2
    UNKNOWN = -1
    EMPTY = 0
    ONE = 1
    TWO = 2
    THREE = 3
    FOUR = 4
    FIVE = 5
    SIX = 6
    SEVEN = 7
    EIGHT = 8
    FLAG = 9


class GameState(IntEnum):
    LOSE = -1
    PLAYING = 0
    WIN = 1


class Button(IntEnum):
    LEFT = 1
    RIGHT = 3

# Difficulty Constants
HEIGHT, WIDTH, NUM_MINES = 6, 6, 4

# Board Constants
BOARD_PADDING = 20
BRICK_SIZE = 30

BOARD_ORIGIN = (BOARD_PADDING, BOARD_PADDING)
BOARD_WIDTH = WIDTH * BRICK_SIZE
BOARD_HEIGHT = HEIGHT * BRICK_SIZE

WINDOW_WIDTH = BOARD_PADDING + BOARD_WIDTH + BOARD_PADDING * 10
WINDOW_HEIGHT = BOARD_PADDING + BOARD_HEIGHT + BOARD_PADDING
WINDOW_SIZE = (WINDOW_WIDTH, WINDOW_HEIGHT)

# Pygame Visual Constants
FLAG_IMG_PATH = "assets/images/flag.png"
MINE_IMG_PATH = "assets/images/mine.png"

OPEN_SANS = "assets/fonts/simkai.ttf"

BLACK = (0, 0, 0)
GRAY = (180, 180, 180)
WHITE = (255, 255, 255)
RED = (255, 0, 0)
GREEN = (0, 255, 0)