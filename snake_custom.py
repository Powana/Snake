import pygame as pg
import numpy as np
from pygame.locals import *
from sys import exit as kill_everything
from random import randrange

"""
Comments: 
coordinate (coord)  : Position on the grid, ie. x=5, y=5 => (5, 5)
position (pos)      : Position on the board in pixels, calculated with coord*tilesize, ie x=5, y=5 => (200, 200)

Anything annoted with 'NN:' was added to adapt the game for the neural network.

Requirements: pygame, numpy, a pc from at least 1657, anything older may not be able to run at 5 fps
"""


# Dimensions of board/tileset, prefeably an odd number
width = 9
height = 9

# I decided to use (coord*tilesize) for placing my objects instead of using a tileset, as it's simpler and this code
# is not intended to be used outside of this script.
tilesize = 40  # px, prefably an even number

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

# Start variables for the snake. The Snake class does have its own defaults but these exist for easy adjustments
snake_spawn_length = 5
snake_spawn_direction = EAST
snake_spawn_coord = (width // 2, height // 2)  # Center of board

# NN:
# Max steps a snake can take without getting a fruit before dying
max_steps = 300

# The target fps, i am not taking into account deltatime with my movement, because even a toaster can run this
# at 6 frames a second. If I had a more demanging game you could account for unstable fps with:
# movement = move_step * deltatime
target_fps = 6
"""
NN:
A global coordinate system for the snake and fruits, used by the neural network to quickly check for collision in a
direction;
0: Nothing on tile
1: snake
2: fruit
"""

world = np.zeros((width, height))


class Snake:
    def __init__(self, spawn_coord=(width//2, height//2), spawn_length=3, spawn_dir=NORTH, draw_snake=True):
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

        if draw_snake:
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

        self.update_world_coords()

    def change_direction(self, direction):
        """
        Update the direction that the snake is moving, the snake will move in this direction until it is changed again.
        :param direction:
        :return:
        """
        # NN:
        # Let the NN try to go into itself, it should die
        # There's probaby a better way to do this
        not_allowed = {SOUTH: NORTH, NORTH: SOUTH, EAST: WEST, WEST: EAST}
        if not_allowed[direction] != self.direction:
            self.direction = direction
        # self.direction = direction

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

        # NN:
        # Set coord of old tail to 0
        world[self.pos[-1][0][1] // tilesize][self.pos[-1][0][0] // tilesize] = 0

        # Start at the back, and set each parts coord and direction to the part in front of it's coord and direction.
        for i, _ in enumerate(reversed(self.pos[1:])):
            self.pos[-i-1] = self.pos[-i-2]

        # Change head coord and direction according to self.direction, this is not done in the previous loop
        drct = self.direction
        self.pos[0] = [[self.pos[0][0][0] + offsets[drct][0], self.pos[0][0][1] + offsets[drct][1]], drct]

    def has_collided(self, fruit):
        """
        After the snake has moved, check if the head collided with anything
        :return: 1: Player died, 2: Player hit the fruit, 3: Nothing collided
        """
        pos = self.pos[0][0]

        # Snake's head is in the same position as another body part, meaning it has crashed
        if pos in [part_pos[0] for part_pos in self.pos[1:]]:
            return 1

        # Snake's head is out of bounds
        elif 0 > pos[0] or (width-1)*tilesize < pos[0] or 0 > pos[1] or (height-1)*tilesize < pos[1]:
            return 1

        elif pos == fruit.corner_pos:
            self.grow()
            return 2

        return 3

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

    def look(self):
        """
        NN:
        This is the function the NN uses to 'see' out from the snakes head. It looks in 8 directions from the head, and
        returns the distance to each type of object in that direction. The returned array has only one dimension, but
        it represents an array of 8 arrays of 3 arrays:
        [[distance_to_wall, distance_to_fruit, distance_to_self], [d_t_w, d_t_f, d_t_s], ... ]
        becomes:
        [d_t_w, d_t_f, d_t_s, d_t_w, d_t_f, d_t_s, ... ]
        The array is 'flattened'.

        This is probably the most uneffecient function in this project, as this game was nor written with any sort of
        raycasting in mind, and has no easy to use gameboard coord system.

        :return: An array of what the snake can see in each direction.
        """

        # head_coord
        h_c = [self.pos[0][0][0] // tilesize, self.pos[0][0][1] // tilesize]

        # default distance, not sure what this should be for best results, worth tinkering with
        d_d = -1
        n_d_t_w, n_d_t_f, n_d_t_s = d_d, d_d, d_d
        e_d_t_w, e_d_t_f, e_d_t_s = d_d, d_d, d_d
        s_d_t_w, s_d_t_f, s_d_t_s = d_d, d_d, d_d
        w_d_t_w, w_d_t_f, w_d_t_s = d_d, d_d, d_d

        # Check north
        # n_d_t_w : north_distance_to_wall
        n_d_t_w = h_c[1]
        for i, row in enumerate(reversed(world[:h_c[1]])):
            if row[h_c[0]] == 2:  # Check the tile on the row that aligns with the heads x coordinate for fruit
                n_d_t_f = i

            if row[h_c[0]] == 1:  # Check for snake
                n_d_t_s = i
                break

        # Check south
        s_d_t_w = height - h_c[1] - 1
        for i, row in enumerate(world[h_c[1]+1:]):
            if row[h_c[0]] == 2:  # Check the tile on the row that aligns with the heads x coordinate
                s_d_t_f = i

            if row[h_c[0]] == 1:  # Check for snake
                s_d_t_s = i
                break

        # Check east
        e_d_t_w = width - h_c[0] - 1
        # Check along the row that the snake is on, starting at the head
        for i, tile in enumerate(world[h_c[1]][h_c[0]+1:]):
            if tile == 2:  # Check for fruit
                e_d_t_f = i

            if tile == 1:  # Check for snake
                e_d_t_s = i
                break

        # Check west
        w_d_t_w = h_c[0]
        # Check along the row that the snake is on, starting at the head
        for i, tile in enumerate(reversed(world[h_c[1]][:h_c[0]])):
            if tile == 2:  # Check for fruit
                w_d_t_f = i

            if tile == 1:  # Check for snake
                w_d_t_s = i
                break

        ne_d_t_w, ne_d_t_f, ne_d_t_s = d_d, d_d, d_d
        nw_d_t_w, nw_d_t_f, nw_d_t_s = d_d, d_d, d_d
        se_d_t_w, se_d_t_f, se_d_t_s = d_d, d_d, d_d
        sw_d_t_w, sw_d_t_f, sw_d_t_s = d_d, d_d, d_d

        # Convectional x and y, with 0, 0 in the bootom left
        c_x = h_c[1] + 1
        c_y = height - h_c[0]

        # Northeast
        # r is the distance to the edge northeast, this took a while to get right
        r = -1
        if c_x == c_y:
            r = c_x - 1
        elif c_x < c_y:
            r = c_x - 1
        elif c_x > c_y:
            r = c_y - 1

        ne_d_t_w = r
        for i in range(r):
            tile = world[h_c[1]-i-1][h_c[0]+i+1]
            if tile == 2:
                ne_d_t_f = i
            if tile == 1:
                ne_d_t_s = i
                break

        # Southwest
        r = -1
        if c_x == c_y:
            r = c_x - 1
        elif c_x < c_y:
            r = c_x - 1
        elif c_x > c_y:
            r = c_y - 1

        sw_d_t_w = r
        for i in range(r):
            tile = world[h_c[0]+i+1][h_c[1]-i-1]
            if tile == 2:
                sw_d_t_f = i
            if tile == 1:
                sw_d_t_s = i
                break

        # Northwest
        r = -1
        if (c_x + c_y) == (width + 1):
            r = c_x - 1
        elif (c_x + c_y) < (width + 1):
            r = c_x - 1
        elif (c_x + c_y) > (width + 1):
            r = height - c_y

        nw_d_t_w = r
        for i in range(r):
            tile = world[h_c[1]-i-1][h_c[0]-i-1]
            if tile == 2:
                nw_d_t_f = i
            if tile == 1:
                nw_d_t_s = i
                break

        # Southeast
        r = -1
        if (c_x + c_y) == (width + 1):
            r = width - c_x
        elif (c_x + c_y) < (width + 1):
            r = c_y - 1
        elif (c_x + c_y) > (width + 1):
            r = width - c_x

        se_d_t_w = r
        for i in range(r):
            tile = world[h_c[1]+i+1][h_c[0]+i+1]
            if tile == 2:
                se_d_t_f = i
            if tile == 1:
                se_d_t_s = i
                break

        vision_array = [
            n_d_t_w, n_d_t_f, n_d_t_s,  # yes, no, no flipped?
            ne_d_t_w, ne_d_t_f, ne_d_t_s,  #
            e_d_t_w, e_d_t_f, e_d_t_s,
            se_d_t_w, se_d_t_f, se_d_t_s,
            s_d_t_w, s_d_t_f, s_d_t_s,
            sw_d_t_w, sw_d_t_f, sw_d_t_s,
            w_d_t_w, w_d_t_f, w_d_t_s,
            nw_d_t_w, nw_d_t_f, nw_d_t_s
        ]

        # Zeros are reserved for no object found, when the snake is directly beside an object the distance is one
        for i, _ in enumerate(vision_array):
            vision_array[i] += 1

        sum_visions = sum(vision_array)
        # Normalize array, not used anymore
        # norm_vision_array = list([float(i)/sum_visions for i in vision_array])

        return vision_array

    def update_world_coords(self):
        # NN:
        # Add all the coords in self.pos to world coords, don't really know why I didn't just do this with the game.
        for p in self.pos:
            world[p[0][1] // tilesize][p[0][0] // tilesize] = 1


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

        self.corner_pos = spawn_pos
        self.pos = [spawn_pos[0]+tilesize//2, spawn_pos[1]+tilesize//2]  # Center of tile

        # NN:
        # Add the fruits coord to world coords
        world[spawn_pos[1] // tilesize][spawn_pos[0] // tilesize] = 2

    def draw(self):
        # The fruit
        pg.draw.circle(board, red, self.pos, tilesize//2-2)


class CheckerBoard:
    """
    Board didn't really need to have its own class, but I felt I should make one for consistency's sake
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


# NN:
# Helper class to spawn when not using fruits, easier to do this than to modify the other classes
class FakeFruit:
    def __init__(self):
        # Used by the Snakes collision check
        self.corner_pos = 0

    # Called when updating GUI
    def draw(self):
        pass


# NN:
# Used for the nerual network, the game is still playable without this class.
class SnakeGame:
    def __init__(self, draw_gui, fruits=True):
        """
        I decided to just use all the global variables instead of parsing them here again to save time
        """
        # Start pygame
        if draw_gui:
            init_pg()
        self.draw_gui = draw_gui

        global world
        # Reset world
        world = np.zeros((width, height))

        # How many fruits the snake has eaten
        self.score = 0

        # The snake (:
        self.snake = Snake(spawn_coord=snake_spawn_coord,
                           spawn_length=snake_spawn_length,
                           spawn_dir=snake_spawn_direction,
                           draw_snake=draw_gui)

        # Generate the first fruit, if fruits is true, else spawn a FakeFruit that won't interfere with the snake
        self.fruit = Fruit(avoid_snake=self.snake) if fruits else FakeFruit()

        # The gameboard
        self.checker_board = CheckerBoard(b_width=width, b_height=height, b_tilesize=tilesize)

        self.score = 0
        self.age = 0

        # If the snake has not found a fruit by max_steps, it dies
        self.steps_left = max_steps

        self.last_dir = 0
        self.changes = 0

    def step(self, direction):
        """
        This is the function that would normally be called once a frame, but in our case we want it to be called as
        often as possible as not to slow down the NN. Every step/frame of the game requires an input from the NN.

        :param direction: The NN's input:
        0 : NORTH
        1 : EAST
        2 : SOUTH
        3 : WEST

        :return: False if the snake survived, an Int of the score if the snake died.
        """
        directions = [NORTH, EAST, SOUTH, WEST]

        if direction != self.last_dir:
            self.changes += 1
        self.snake.change_direction(directions[direction])
        self.last_dir = direction

        if self.draw_gui:
            # Draw objects
            self.checker_board.draw()
            self.snake.update()  # move + draw
            self.fruit.draw()
        else:
            self.snake.move()

        # I placed this outside of the Snake class, as it seemed to make more sense for it to be in the game loop
        collided = self.snake.has_collided(self.fruit)

        if collided == 1 or self.steps_left == 0:
            # The snake is dead, return the score and age for the NN to use as fitness
            return self.score, self.age

        elif collided == 2:
            # Snake ate a fruit
            # NN:
            # Change old fruit's coord in world to be a snake
            world[self.fruit.corner_pos[1] // tilesize][self.fruit.corner_pos[0] // tilesize] = 1
            self.fruit = Fruit(self.snake)
            self.score += 1
            # Reset the step counter
            self.steps_left = max_steps

        # Snake survived
        self.snake.update_world_coords()

        if self.draw_gui:
            # Update the display, not needed for the NN, but fun to look at, probably slows it down a ton though
            pg.display.flip()

        self.steps_left -= 1

        # The age of this game/snake increases by one every step
        self.age += 1

        # return False when the snake survived the step/frame
        return False


def main():
    # How many fruits collected, displayed on game over
    score = 0

    # The snake (:
    snake = Snake(spawn_coord=[2, 3], spawn_length=snake_spawn_length, spawn_dir=snake_spawn_direction)

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
        collided = snake.has_collided(fruit)
        if collided == 1:
            game_over(score)
            # This code is only run if player continues, it overwrites the old snake with a new snake:
            snake = Snake(spawn_coord=snake_spawn_coord,
                          spawn_length=snake_spawn_length,
                          spawn_dir=snake_spawn_direction)
            score = 0
        elif collided == 2:
            # Snake ate a fruit
            # Make a new fruit
            fruit = Fruit(snake)
            score += 1

        # Update the display
        pg.display.flip()

        clock.tick(target_fps)


def manaul_mode():
    game = SnakeGame(True)

    while True:
        for ev in pg.event.get():  # ev : event
            if ev.type == KEYDOWN:
                k = ev.key
                if k == K_UP or k == k_up_a:
                    game.step(0)
                elif k == K_RIGHT or k == k_right_a:
                    game.step(1)
                elif k == K_DOWN or k == k_down_a:
                    game.step(2)
                elif k == K_LEFT or k == k_left_a:
                    game.step(3)
                print("o:", str(list([str(d).rjust(5) for d in game.snake.look()])).replace("'", ""))


def init_pg():
    global board, clock
    # Init pygame
    pg.init()
    clock = pg.time.Clock()

    # The board that the entire game plays out on
    board = pg.display.set_mode((width * tilesize, height * tilesize))
    pg.display.set_caption("Snake!")
    pg.display.flip()


if __name__ == '__main__':
    init_pg()

    # Change to manual if you want to play the game step by step
    # manaul_mode()
    main()
