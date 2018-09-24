import numpy as np
import pygame as pg
from pygame.locals import *
from sys import exit as kill_everything
from random import randrange

"""
Comments: 
coordinate (coord)  : Position on the grid, ie. x=5, y=5 => (5, 5)
position (pos)      : Position on the board in pixels, calculated with coord*tilesize, ie x=5, y=5 => (200, 200)  

Requirements: pygame, a pc from at least 1657, anything older may not be able to run at 5 fps
"""


# ###### Initialization ####### #

pg.init()
clock = pg.time.Clock()

# Dimensions of board/tileset, prefeably an odd number
width = 17
height = 17

# I decided to use (coord*tilesize) for placing my objects instead of using a tileset, as it's simpler and this code
# is not intended to be used outside of this script.
tilesize = 40  # px, prefably an even number

# The board that the entire game plays out on
board = pg.display.set_mode((width*tilesize, height*tilesize))
pg.display.set_caption("Snake!")

# Colours
light_green = (11, 163, 46)
dark_green = (7, 150, 40)
black = (0, 0, 0)
red = (214, 29, 29)

# Degrees to rotate for each direction
NORTH = 90
EAST = 0
SOUTH = 270
WEST = 180
# Distance between each part when facing a direction in X, Y. This is the delta of movement in a direction.
offsets = {NORTH: (0, -tilesize),
           EAST: (tilesize, 0),
           SOUTH: (0, tilesize),
           WEST: (-tilesize, 0)}

# Alternative directional keys
k_up_a = K_w
k_down_a = K_s
k_right_a = K_d
k_left_a = K_a

# Star variables for the snake. The Snake class does have its own defaults but these exist for easy adjustments
snake_spawn_length = 5
snake_spawn_direction = EAST
snake_spawn_coord = (width // 2, height // 2)  # Center of board

# The target fps, i am not taking into account deltatime with my movement, because even a toaster can run this
# at 5 frames a second. If I had a more demanging game you could account for unstable fps with:
# movement = move_step * deltatime
target_fps = 5


class Snake:
    def __init__(self, spawn_coord=(width//2, height//2), spawn_length=3, spawn_dir=NORTH):
        """
        :param spawn_coord: Where the snakes head should spawn when game is started
        :param spawn_length: Snakes starting length
        """
        # === Class variables === #

        self.length = spawn_length-1  # -1 to acount for the head
        # self.pos is a list of all the coords of the snakes parts, and the direction they face, [[[X, Y], direction],]
        self.pos = []
        # The direction the snake is moving in
        self.direction = spawn_dir

        self.head_sprite = pg.image.load("res/head.png").convert_alpha()
        self.body_sprite = pg.image.load("res/body.png").convert_alpha()

        # === Initial snake spawning === #

        # Convert coords to usable pixel values
        spawn_coord = [spawn_coord[0]*tilesize, spawn_coord[1]*tilesize]

        # Head position, [[X, Y], direction]
        self.pos.append([spawn_coord, spawn_dir])

        # Body parts offset from eachother according to the spawn_dir
        x_offset = offsets[spawn_dir][0]
        y_offset = offsets[spawn_dir][1]

        # Add coords and directions for the initial body parts
        for i in range(self.length):
            # The offsets are how much the snake should move when moving in that direction, so the body parts are placed
            # in the opposite direction of that. A north facing snake will have its parts placed south of the head.
            self.pos.append([[spawn_coord[0]-x_offset*(i+1), spawn_coord[1]-y_offset*(i+1)], spawn_dir])

    def change_direction(self, direction):
        """
        Update the direction that the snake is moving, the snake will move in this direction until it is changed again.
        :param direction:
        :return:
        """
        # There's probaby a better way to do this
        not_allowed = {SOUTH: NORTH, NORTH: SOUTH, EAST: WEST, WEST: EAST}
        if not_allowed[direction] != self.direction:
            self.direction = direction

    def draw(self):
        """
        Draw sprites for every position in self.pos, note that this only draws the coords, without updating them.
        :return:
        """

        # Draw all parts after the head
        for part in self.pos[1:]:
            board.blit(pg.transform.rotate(self.body_sprite, part[1]), part[0])

        # Draw head last, as it it useful to draw it ontop of the body parts
        board.blit(pg.transform.rotate(self.head_sprite, self.pos[0][1]), self.pos[0][0])

    def move(self):
        """
        Update the coordinates of all the snakes parts so that they may be drawn in a new place, simulating movement.
        :return:
        """

        # Start at the back, and set each parts coord and direction to the part in front of it's coord and direction.
        for i, _ in enumerate(reversed(self.pos[1:])):
            self.pos[-i-1] = self.pos[-i-2]

        # Change head coord and direction according to self.direction, this is not done in the previous loop
        drct = self.direction
        self.pos[0] = [[self.pos[0][0][0] + offsets[drct][0], self.pos[0][0][1] + offsets[drct][1]], drct]

    def check_collision(self, fruit):
        """
        After the snake has moved, check if the head collided with anything
        :return: True: Player died, False: Player hit the fruit, None: Nothing collided
        """
        pos = self.pos[0][0]

        # Snake's head is in the same position as another body part, meaning it has crashed
        if pos in [part_pos[0] for part_pos in self.pos[1:]]:
            return True

        # Snake's head is out of bounds
        elif 0 > pos[0] or (width-1)*tilesize < pos[0] or 0 > pos[1] or (height-1)*tilesize < pos[1]:
            return True

        elif pos == fruit.normal_pos:
            self.grow()
            return False

        return None

    def grow(self):
        self.length += 1
        self.pos.append(self.pos[-1])  # Add a copy of the last body part

    def update(self):
        """
        Helper function that updated the snakes position, then draws the snake in the new position.
        Note that this does not erase the previous snake, but relies on that being overwritten by the board
        :return:
        """
        self.move()
        self.draw()


class Fruit:
    def __init__(self, avoid_snake: Snake):
        """
        Generate a fruit on a random tile that the snake is not currently occupying
        :param avoid_snake: the snake to avoid
        """
        impossible_spawns = []
        spawn_pos = []

        # Add all the snakes current positions to the impossible spawns list
        # This could be made into one massive inline loop, but this is more readable
        for xy in [pos[0] for pos in avoid_snake.pos]:
            impossible_spawns.append(xy)

        # I could have made a large array of possible_spawns, but it turns out this is faster
        # Check if spawn_pos is impossible or nonexistant, if it is, generate a new one.
        while spawn_pos in impossible_spawns or not spawn_pos:
            spawn_pos = [randrange(0, width)*tilesize, randrange(0, height)*tilesize]

        self.normal_pos = spawn_pos
        self.pos = [spawn_pos[0]+tilesize//2, spawn_pos[1]+tilesize//2]  # Center of tile

    def draw(self):
        # The fruit
        pg.draw.circle(board, red, self.pos, tilesize//2-2)


class CheckerBoard:
    """
    Board didn't really need to have its own class, but I felt I should make one for consitencys sake
    """
    def __init__(self,  b_width, b_height, b_tilesize):
        self.width = b_width
        self.height = b_height
        self.tilesize = b_tilesize

    def draw(self):
        colour = dark_green
        # For loop to create the board. I decided against using a tileset or something similair, simply unnecassary
        for w in range(self.width):
            for h in range(self.height):
                pg.draw.rect(board,                                     # Where to draw the rect
                             colour,                                    # The colour
                             (w * self.tilesize, h * self.tilesize,     # Position of the rect
                              self.tilesize, self.tilesize))            # Size of the rect
                # This creates the checkerboard effect
                colour = light_green if colour == dark_green else dark_green


def game_over(score):
    cont = False  # Whether or not the player wants to play again

    s = pg.Surface((width*tilesize, height*tilesize))  # Black background with an alpha making it translucent
    s.fill(black)
    s.set_alpha(100)
    board.blit(s, (0, 0))

    big_font = pg.font.SysFont("Arial", 40, bold=True)
    font = pg.font.SysFont("Arial", 24)

    game_over_text = big_font.render("Game Over!", True, black)  # Antialias=True, color=black
    score_string = "Your score: " + str(score)
    score_text = font.render(score_string, True, black)
    continue_text = font.render("Press SPACE to try again or ESCAPE to quit.", True, black)

    board.blit(game_over_text, (20, height//2*tilesize - 45))
    board.blit(score_text, (20, height//2*tilesize))
    board.blit(continue_text, (20, height//2*tilesize + 26))

    # Nested game loop
    while True:
        # Check for use input
        for ev in pg.event.get():  # ev : event
            if ev.type == QUIT:
                pg.quit()
                kill_everything()

            elif ev.type == KEYDOWN:
                k = ev.key
                if k == K_ESCAPE:
                    pg.quit()
                    kill_everything()
                elif k == K_SPACE:
                    cont = True
                    break
            break
        if cont:
            break

        # Update the display, this might aswell be high to give a more responsive feel
        clock.tick(60)

        pg.display.flip()


# I try to handle all top-level game logic in main, ie. object creation, game start, game over, game restart, score
def main():
    # How many fruits collected, displayed on game over
    score = 0

    # The snake (:
    snake = Snake(spawn_coord=snake_spawn_coord, spawn_length=snake_spawn_length, spawn_dir=snake_spawn_direction)

    # Generate the first fruit
    fruit = Fruit(avoid_snake=snake)

    # The gameboard
    checker_board = CheckerBoard(b_width=width, b_height=height, b_tilesize=tilesize)

    # ###### Main game loop ###### #
    while True:

        for ev in pg.event.get():  # ev : event
            if ev.type == QUIT:
                pg.quit()
                kill_everything()

            elif ev.type == KEYDOWN:
                k = ev.key
                # Could use a for loop, but this is more readable
                if k == K_UP or k == k_up_a:
                    snake.change_direction(NORTH)
                elif k == K_RIGHT or k == k_right_a:
                    snake.change_direction(EAST)
                elif k == K_DOWN or k == k_down_a:
                    snake.change_direction(SOUTH)
                elif k == K_LEFT or k == k_left_a:
                    snake.change_direction(WEST)
                # Limit user to one keypress per frame, this has the drawback that pressing two keys in succession fast
                # enough will only register the first key if both are pressed during the same frame, without the break
                # a user can change direction twice in a frame, allowing the snake to make a 180turn in one frame.
                break

        # Draw objects
        checker_board.draw()
        fruit.draw()
        snake.update()  # move + draw

        # I placed this outside of the Snake class, as it seemed to make more sense for it to be in the game loop
        collided = snake.check_collision(fruit)
        if collided:
            game_over(score)
            # This code is only run if player continues, it overwrites the old snake with a new snake:
            snake = Snake(spawn_coord=snake_spawn_coord,
                          spawn_length=snake_spawn_length,
                          spawn_dir=snake_spawn_direction)
            score = 0
        elif collided == False:  # Ignore the pep-8 violation, collided can also return None, and None == not True
            #  a new fruit
            fruit = Fruit(snake)
            score += 1

        # Update the display
        pg.display.flip()

        clock.tick(target_fps)


if __name__ == '__main__':
    main()
