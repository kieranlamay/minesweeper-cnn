# init_game.py
import pygame
from constants import *

def init_game():
    pygame.init()
    screen = pygame.display.set_mode(WINDOW_SIZE)
    return screen

def load_assets():
    # Load fonts
    smallFont = pygame.font.Font(OPEN_SANS, 16)
    mediumFont = pygame.font.Font(OPEN_SANS, 32)
    largeFont = pygame.font.Font(OPEN_SANS, 48)
    
    # Load and scale images
    flag_image = pygame.image.load(FLAG_IMG_PATH)
    flag_image = pygame.transform.scale(flag_image, (BRICK_SIZE, BRICK_SIZE))
    mine_image = pygame.image.load(MINE_IMG_PATH)
    mine_image = pygame.transform.scale(mine_image, (BRICK_SIZE, BRICK_SIZE))

    return smallFont, mediumFont, largeFont, flag_image, mine_image