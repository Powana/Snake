import pickle

# Run this to test the currently saved fittest snake

snakenum = 101

with open("fittest_snake_" + str(snakenum) + ".pickle", "rb") as file:
        snake = pickle.load(file)
        print(snake["snake_settings"])

        while True:
            print(snake["snake"].play(True, 0.1, snake["n_fruits"]))
