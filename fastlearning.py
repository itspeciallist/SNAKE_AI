import numpy as np
import random
import tkinter as tk
import threading
import pickle
import os

# --- Parameters ---
GRID_SIZE = 15
CELL_SIZE = 25
MAX_STEPS = 150

INPUT_SIZE = 6
HIDDEN_SIZE = 6
OUTPUT_SIZE = 4

POPULATION_SIZE = 500
GENERATIONS_PER_UPDATE = 10
MUTATION_RATE = 0.1
MUTATION_STRENGTH = 0.5

SAVE_FILENAME = "best_snake_ai.pkl"

# --- Utility ---
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# --- Neural Network ---
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

    def mutate(self):
        def mutate_matrix(m):
            mask = np.random.rand(*m.shape) < MUTATION_RATE
            mutations = np.random.randn(*m.shape) * MUTATION_STRENGTH
            m += mask * mutations
        mutate_matrix(self.W1)
        mutate_matrix(self.b1)
        mutate_matrix(self.W2)
        mutate_matrix(self.b2)

def save_network(net, filename=SAVE_FILENAME):
    with open(filename, "wb") as f:
        pickle.dump({
            'W1': net.W1,
            'b1': net.b1,
            'W2': net.W2,
            'b2': net.b2
        }, f)

def load_network(filename=SAVE_FILENAME):
    net = NeuralNet()
    if os.path.exists(filename):
        with open(filename, "rb") as f:
            data = pickle.load(f)
            net.W1 = data['W1']
            net.b1 = data['b1']
            net.W2 = data['W2']
            net.b2 = data['b2']
    return net

# --- Snake Game ---
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

    def run(self):
        while not self.game_over:
            self.step()
        return self.score + self.steps * 0.1

# --- Evolution ---
def evolve_population(pop):
    fitnesses = np.array([SnakeGame(net).run() for net in pop])
    n_survivors = max(2, POPULATION_SIZE // 10)
    survivors_idx = fitnesses.argsort()[-n_survivors:]
    survivors = [pop[i] for i in survivors_idx]
    print(f"Best score this generation: {fitnesses[survivors_idx[-1]]:.2f}")

    new_pop = []
    for _ in range(POPULATION_SIZE):
        parent = random.choice(survivors).copy()
        parent.mutate()
        new_pop.append(parent)

    return new_pop, fitnesses[survivors_idx[-1]], survivors[-1]

# --- GUI ---
class SnakeGUI(tk.Canvas):
    def __init__(self, master, net):
        width = GRID_SIZE * CELL_SIZE
        height = GRID_SIZE * CELL_SIZE
        super().__init__(master, width=width, height=height, bg='black')
        self.pack()
        self.net = net
        self.game = SnakeGame(net)
        self.after(3, self.game_loop)

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
            self.after(3, self.game_loop)
        else:
            self.game.reset()
            self.after(3, self.game_loop)

# --- Training Thread ---
class TrainerThread(threading.Thread):
    def __init__(self, update_callback):
        super().__init__()
        best_net = load_network()
        self.best_score = 0
        if os.path.exists(SAVE_FILENAME):
            print("Loaded existing best network from file.")
            self.population = []
            for _ in range(POPULATION_SIZE):
                net = best_net.copy()
                net.mutate()
                self.population.append(net)
            self.best_score = SnakeGame(best_net).run()
            self.best_net = best_net.copy()
        else:
            print("No saved network found, starting fresh population.")
            self.population = [NeuralNet() for _ in range(POPULATION_SIZE)]
            self.best_net = None

        self.update_callback = update_callback
        self.running = True

    def run(self):
        while self.running:
            for _ in range(GENERATIONS_PER_UPDATE):
                self.population, gen_best, best_candidate = evolve_population(self.population)
                if gen_best > self.best_score:
                    self.best_score = gen_best
                    self.best_net = best_candidate.copy()
                    save_network(self.best_net)
                    print(f"New best score: {self.best_score:.2f}, network saved.")

            if self.update_callback:
                self.update_callback(self.best_net)

    def stop(self):
        self.running = False

# --- Main Entry Point ---
def main():
    root = tk.Tk()
    root.title("Snake AI Trainer")

    gui = SnakeGUI(root, load_network())

    def update_best(net):
        gui.net = net
        gui.game = SnakeGame(net)

    trainer = TrainerThread(update_callback=update_best)
    trainer.start()

    def on_close():
        trainer.stop()
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", on_close)
    root.mainloop()

if __name__ == "__main__":
    main()
