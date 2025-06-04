import tkinter as tk
import numpy as np
import pickle
import os
import random

# Same grid and parameters as training script
GRID_SIZE = 15
CELL_SIZE = 25
MAX_STEPS = 150

INPUT_SIZE = 6
HIDDEN_SIZE = 10
OUTPUT_SIZE = 4

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

class NeuralNet:
    def __init__(self):
        self.W1 = np.random.randn(INPUT_SIZE, HIDDEN_SIZE)
        self.b1 = np.zeros(HIDDEN_SIZE)
        self.W2 = np.random.randn(HIDDEN_SIZE, OUTPUT_SIZE)
        self.b2 = np.zeros(OUTPUT_SIZE)

    def forward(self, x):
        h = sigmoid(np.dot(x, self.W1) + self.b1)
        out = np.dot(h, self.W2) + self.b2
        return out

    def predict(self, x):
        out = self.forward(x)
        return np.argmax(out)

    def copy(self):
        clone = NeuralNet()
        clone.W1 = np.copy(self.W1)
        clone.b1 = np.copy(self.b1)
        clone.W2 = np.copy(self.W2)
        clone.b2 = np.copy(self.b2)
        return clone

def load_network(filename="neuralsnake.pkl"):
    net = NeuralNet()
    if os.path.exists(filename):
        with open(filename, "rb") as f:
            data = pickle.load(f)
            net.W1 = data['W1']
            net.b1 = data['b1']
            net.W2 = data['W2']
            net.b2 = data['b2']
    else:
        print(f"File {filename} not found, running untrained AI.")
    return net

class SnakeGame:
    def __init__(self, net):
        self.net = net
        self.reset()

    def reset(self):
        self.snake = [(GRID_SIZE // 2, GRID_SIZE // 2)]
        self.direction = (0, -1)
        self.place_food()
        self.steps = 0
        self.score = 0
        self.game_over = False

    def place_food(self):
        while True:
            food = (random.randint(0, GRID_SIZE - 1), random.randint(0, GRID_SIZE - 1))
            if food not in self.snake:
                self.food = food
                break

    def get_state(self):
        head_x, head_y = self.snake[0]
        dir_x, dir_y = self.direction
        food_dx = (self.food[0] - head_x) / GRID_SIZE
        food_dy = (self.food[1] - head_y) / GRID_SIZE

        def danger(direction):
            nx, ny = head_x + direction[0], head_y + direction[1]
            if nx < 0 or nx >= GRID_SIZE or ny < 0 or ny >= GRID_SIZE:
                return 1
            if (nx, ny) in self.snake:
                return 1
            return 0

        def turn_right(d):
            return (d[1], -d[0])

        def turn_left(d):
            return (-d[1], d[0])

        danger_straight = danger(self.direction)
        danger_right = danger(turn_right(self.direction))
        danger_left = danger(turn_left(self.direction))

        return np.array([dir_x, dir_y, food_dx, food_dy, danger_straight, danger_right])

    def update_direction(self):
        state = self.get_state()
        action = self.net.predict(state)
        possible_dirs = [(0, -1), (0, 1), (-1, 0), (1, 0)]
        chosen_dir = possible_dirs[action]

        dx, dy = self.direction
        if (dx + chosen_dir[0] == 0 and dy + chosen_dir[1] == 0):
            return
        self.direction = chosen_dir

    def step(self):
        if self.game_over:
            return

        self.update_direction()

        head_x, head_y = self.snake[0]
        dx, dy = self.direction
        new_head = (head_x + dx, head_y + dy)

        if (new_head[0] < 0 or new_head[0] >= GRID_SIZE or
            new_head[1] < 0 or new_head[1] >= GRID_SIZE or
            new_head in self.snake):
            self.game_over = True
            return

        self.snake.insert(0, new_head)

        if new_head == self.food:
            self.score += 1
            self.place_food()
            self.steps = 0
        else:
            self.snake.pop()

        self.steps += 1
        if self.steps > MAX_STEPS:
            self.game_over = True

class SnakeGUI(tk.Canvas):
    def __init__(self, master, net):
        width = GRID_SIZE * CELL_SIZE
        height = GRID_SIZE * CELL_SIZE
        super().__init__(master, width=width, height=height, bg='black')
        self.pack()
        self.net = net
        self.game = SnakeGame(net)
        self.after(1000, self.game_loop)

    def draw(self):
        self.delete(tk.ALL)
        fx, fy = self.game.food
        self.create_rectangle(fx * CELL_SIZE, fy * CELL_SIZE,
                              (fx + 1) * CELL_SIZE, (fy + 1) * CELL_SIZE,
                              fill='red')
        for i, (x, y) in enumerate(self.game.snake):
            color = 'green' if i == 0 else 'lightgreen'
            self.create_rectangle(x * CELL_SIZE, y * CELL_SIZE,
                                  (x + 1) * CELL_SIZE, (y + 1) * CELL_SIZE,
                                  fill=color)
        self.create_text(5, 5, anchor='nw', fill='white',
                         text=f'Score: {self.game.score}')
        if self.game.game_over:
            self.create_text(self.winfo_width() // 2, self.winfo_height() // 2,
                             text="GAME OVER", fill='white', font=('Arial', 30))

    def game_loop(self):
        if not self.game.game_over:
            self.game.step()
            self.draw()
            self.after(100, self.game_loop)
        else:
            self.game.reset()
            self.after(1000, self.game_loop)

def main():
    root = tk.Tk()
    root.title("Snake AI Play")

    net = load_network()
    gui = SnakeGUI(root, net)

    root.mainloop()

if __name__ == "__main__":
    main()
