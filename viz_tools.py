import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matching_functions import *

def plot_two_matchings(X, Y, p, matching1, matching2):
    """Plot two matchings side by side with costs"""
    cost1 = compute_cost(X, Y, matching1, p)
    cost2 = compute_cost(X, Y, matching2, p)

    print(f"Left: cost = {cost1:.6f}")
    print(f"Right: cost = {cost2:.6f}")
    print(f"Difference: {abs(cost1 - cost2):.6f}")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    for ax, match in [(axes[0], matching1), (axes[1], matching2)]:
        ax.axhline(0, color='k', linewidth=1)

        max_r = 0
        for i, j in match:
            center = (X[i] + Y[j]) / 2
            radius = abs(Y[j] - X[i]) / 2
            max_r = max(max_r, radius)
            circle = Circle((center, 0), radius, fill=False, edgecolor='grey', linewidth=1.5)
            ax.add_patch(circle)

        ax.scatter(X, np.zeros_like(X), c='red', s=50)
        ax.scatter(Y, np.zeros_like(Y), c='blue', s=50)

        y_lim = max(max_r * 1.1, 0.15)
        ax.set_ylim(-y_lim, y_lim)
        ax.set_xlim(min(np.min(X), np.min(Y)) - 0.05,
                    max(np.max(X), np.max(Y)) + 0.05)
        ax.set_aspect('equal')
        ax.axis('off')

    plt.tight_layout()
    plt.show()

def plot_cost_progression(history, X, Y, p, optimal):
    """Plot total cost vs number of swaps"""
    costs = [compute_cost(X, Y, history[k], p) for k in range(len(history))]
    opt_cost = compute_cost(X, Y, optimal, p)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(range(len(costs)), costs, marker='o', linewidth=1)
    ax.axhline(opt_cost, color='k', linestyle='--', linewidth=1, label='Optimal')
    ax.set_xlabel('Number of Swaps')
    ax.set_ylabel('Total Cost')
    ax.set_title(f'Cost Progression (n={len(X)}, p={p})')
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    plt.show()