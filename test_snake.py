import pickle

# Run this to test the currently saved fittest snake

while True:
    with open("fittest_snake.pickle", "rb") as file:
        snake = pickle.load(file)

        print(snake[0].play(True, 0.1, True))
