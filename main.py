import random
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

moves = ["up", "down", "left", "right"]


class Agent:
    def __init__(self, startRow=1, startColumn=1, goalRow=6, goalColumn=6):
        self.startRow = startRow
        self.startColumn = startColumn
        self.goalRow = goalRow
        self.goalColumn = goalColumn

        self.reset()

        self.grid = {a: {b: "." for b in range(1, 7)} for a in range(1, 7)}
        self.qTable = {
            a: {b: {c: 0.0 for c in moves} for b in range(1, 7)}
            for a in range(1, 7)
        }

    def reset(self):
        self.row = self.startRow
        self.column = self.startColumn
        self.moves = 0
        return (self.row, self.column)

    def update_grid_visual(self):
        for r in self.grid:
            for c in self.grid[r]:
                self.grid[r][c] = "."
        self.grid[self.goalRow][self.goalColumn] = "G"
        self.grid[self.row][self.column] = "A"

    def choose_action(self, state, epsilon=0.2):
        if random.random() < epsilon:
            return random.choice(moves)
        q_vals = self.qTable[state[0]][state[1]]
        return max(q_vals, key=q_vals.get)

    def step(self, direction):
        direction = direction.lower()
        old_state = (self.row, self.column)
        next_row, next_col = self.row, self.column

        if direction == "up":
            next_row -= 1
        elif direction == "down":
            next_row += 1
        elif direction == "left":
            next_col -= 1
        elif direction == "right":
            next_col += 1

        hit_wall = next_row < 1 or next_row > 6 or next_col < 1 or next_col > 6

        if hit_wall:
            reward = -2.0
            done = True
        else:
            self.row, self.column = next_row, next_col
            self.moves += 1
            is_goal = (self.row == self.goalRow and self.column ==
                       self.goalColumn)
            reward = 10.0 if is_goal else -0.1
            done = is_goal

        return old_state, (self.row, self.column), reward, done

    def train(self, episodes=500, alpha=0.1, gamma=0.9, epsilon=0.2):
        for _ in range(episodes):
            state = self.reset()
            done = False

            while not done:
                action = self.choose_action(state, epsilon)
                old_state, next_state, reward, done = self.step(action)

                current_q = self.qTable[old_state[0]][old_state[1]][action]

                if done:
                    new_q = current_q + alpha * (reward - current_q)
                else:
                    max_future_q = max(
                        self.qTable[next_state[0]][next_state[1]].values())
                    new_q = current_q + alpha * \
                        (reward + gamma * max_future_q - current_q)

                self.qTable[old_state[0]][old_state[1]][action] = new_q
                state = next_state

    def print_grid(self):
        self.update_grid_visual()
        for r in self.grid:
            print(" ".join(str(self.grid[r][c]) for c in self.grid[r]))
        print("-" * 12)

    def export_qtable_heatmap(self, filename="qtable_heatmap.png"):
        # Convert dictionary Q-table into a 6x6 numerical matrix of max Q-values
        heatmap_matrix = np.zeros((6, 6))

        for r in range(1, 7):
            for c in range(1, 7):
                # Extract max Q-value for cell (r, c)
                max_q = max(self.qTable[r][c].values())
                heatmap_matrix[r - 1, c - 1] = max_q

        # Set up plot figure
        plt.figure(figsize=(8, 6))
        ax = sns.heatmap(
            heatmap_matrix,
            annot=True,  # Shows Q-values inside each cell
            fmt=".1f",  # Formats numbers to 1 decimal place
            cmap="YlGnBu",  # Color map (Yellow = low, Dark Blue = high)
            xticklabels=range(1, 7),
            yticklabels=range(1, 7),
            cbar_kws={"label": "Max Q-Value"},
        )

        # Highlight Goal and Start positions
        ax.text(
            self.goalColumn - 0.5,
            self.goalRow - 0.5,
            "\n\n[GOAL]",
            ha="center",
            va="center",
            color="red",
            weight="bold",
        )
        ax.text(
            self.startColumn - 0.5,
            self.startRow - 0.5,
            "\n\n[START]",
            ha="center",
            va="center",
            color="green",
            weight="bold",
        )

        plt.title("Q-Table Heatmap (Max Expected Reward per State)")
        plt.xlabel("Column")
        plt.ylabel("Row")

        # Save to file
        plt.savefig(filename, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Heatmap successfully exported as '{filename}'!")


agent = Agent(startRow=1, startColumn=1, goalRow=6, goalColumn=6)

agent.train(episodes=5000)

state = agent.reset()
done = False
print("Optimized Path:")
agent.print_grid()

while not done and agent.moves < 20:
    action = agent.choose_action(state, epsilon=0.0)
    _, state, _, done = agent.step(action)
    agent.print_grid()

print(f"Reached goal in {agent.moves} steps!")
agent.export_qtable_heatmap("grid_qtable_heatmap.png")
