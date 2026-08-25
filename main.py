import random
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

MOVES = ["up", "down", "left", "right"]


class Agent:
    def __init__(self, startRow=1, startColumn=1, goalRow=6, goalColumn=6):
        self.startRow = startRow
        self.startColumn = startColumn
        self.goalRow = goalRow
        self.goalColumn = goalColumn

        self.reset()

        # Initialize 1-indexed 6x6 grid environment and Q-table
        self.grid = {a: {b: "." for b in range(1, 7)} for a in range(1, 7)}
        self.qTable = {
            a: {b: {c: 0.0 for c in MOVES} for b in range(1, 7)}
            for a in range(1, 7)
        }

    def reset(self):
        """Resets the agent to its initial position and resets the step counter."""
        self.row = self.startRow
        self.column = self.startColumn
        self.moves = 0
        return (self.row, self.column)

    def update_grid_visual(self):
        """Refreshes the grid board with current Agent ('A') and Goal ('G') markers."""
        for r in self.grid:
            for c in self.grid[r]:
                self.grid[r][c] = "."
        self.grid[self.goalRow][self.goalColumn] = "G"
        self.grid[self.row][self.column] = "A"

    def choose_action(self, state, epsilon=0.2):
        """Epsilon-greedy policy: explores randomly with probability `epsilon`, 
        otherwise exploits the action with the highest Q-value."""
        if random.random() < epsilon:
            return random.choice(MOVES)
        
        q_vals = self.qTable[state[0]][state[1]]
        return max(q_vals, key=q_vals.get)

    def step(self, direction):
        """Executes a move, checks for wall collisions or goal reach, and returns 
        the step transition tuple: (old_state, new_state, reward, done)."""
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

        # Check for out-of-bounds movement (1-indexed 6x6 grid boundary)
        hit_wall = next_row < 1 or next_row > 6 or next_col < 1 or next_col > 6

        if hit_wall:
            reward = -2.0  # Penalty for hitting a boundary
            done = True
        else:
            self.row, self.column = next_row, next_col
            self.moves += 1
            is_goal = (self.row == self.goalRow and self.column == self.goalColumn)
            reward = 10.0 if is_goal else -0.1  # Step penalty encourages shorter paths
            done = is_goal

        return old_state, (self.row, self.column), reward, done

    def train(self, episodes=500, alpha=0.1, gamma=0.9, epsilon=0.2):
        """Trains the agent using the Q-Learning update rule over a specified number of episodes."""
        for _ in range(episodes):
            state = self.reset()
            done = False

            while not done:
                action = self.choose_action(state, epsilon)
                old_state, next_state, reward, done = self.step(action)

                current_q = self.qTable[old_state[0]][old_state[1]][action]

                # Q-learning Bellman Update formula
                if done:
                    new_q = current_q + alpha * (reward - current_q)
                else:
                    max_future_q = max(self.qTable[next_state[0]][next_state[1]].values())
                    new_q = current_q + alpha * (reward + gamma * max_future_q - current_q)

                self.qTable[old_state[0]][old_state[1]][action] = new_q
                state = next_state

    def print_grid(self):
        """Outputs current state layout to standard output."""
        self.update_grid_visual()
        for r in self.grid:
            print(" ".join(str(self.grid[r][c]) for c in self.grid[r]))
        print("-" * 12)

    def export_qtable_heatmap(self, filename="qtable_heatmap.png"):
        """Generates and saves a Seaborn heatmap illustrating max Q-values across the grid."""
        heatmap_matrix = np.zeros((6, 6))

        for r in range(1, 7):
            for c in range(1, 7):
                heatmap_matrix[r - 1, c - 1] = max(self.qTable[r][c].values())

        plt.figure(figsize=(8, 6))
        ax = sns.heatmap(
            heatmap_matrix,
            annot=True,
            fmt=".1f",
            cmap="YlGnBu",
            xticklabels=range(1, 7),
            yticklabels=range(1, 7),
            cbar_kws={"label": "Max Q-Value"},
        )

        # Annotate Goal and Start positions directly on heatmap cells
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

        plt.savefig(filename, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Heatmap successfully exported as '{filename}'!")
