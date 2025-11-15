from scipy.optimize import linear_sum_assignment
import numpy as np
import heapq
from numba import jit

def generate_random_points(n):
    """Generate n random sorted points on [0,1] for each set"""
    X = np.sort(np.random.rand(n))
    Y = np.sort(np.random.rand(n))
    return X, Y


def compute_cost(X, Y, matching, p):
    """Compute total cost of a matching"""
    return sum(abs(X[i] - Y[j])**p for i, j in matching)


def matchings_equal(m1, m2):
    """Check if two matchings are identical"""
    return set(m1) == set(m2)
  

def compute_optimal(X, Y, p):
    """Compute optimal matching using Jonker-Volgenant algorithm"""
    n = len(X)
    C = np.array([[abs(X[i] - Y[j])**p for j in range(n)] for i in range(n)])
    row_ind, col_ind = linear_sum_assignment(C)
    return list(zip(row_ind, col_ind)), C[row_ind, col_ind].sum()
  

@jit
def compute_dyck(X, Y, p=0.5):
    """Compute Dyck matching using level set algorithm"""
    n = len(X)
    red_index = 0
    blue_index = 0
    matching = []
    stack = []
    stack_is_red = True
    cost = 0

    while red_index < n or blue_index < n:
        # Determine next point to process
        if red_index == n:
            next_is_red = False
        elif blue_index == n:
            next_is_red = True
        else:
            next_is_red = X[red_index] <= Y[blue_index]

        if next_is_red:
            # Next point is red
            if not stack or stack_is_red:
                stack.append(red_index)
                stack_is_red = True
            else:
                matching.append((red_index, stack[-1]))
                cost += abs(X[red_index] - Y[stack.pop()])**p
            red_index += 1
        else:
            # Next point is blue
            if not stack or not stack_is_red:
                stack.append(blue_index)
                stack_is_red = False
            else:
                matching.append((stack[-1], blue_index))
                cost += abs(X[stack.pop()] - Y[blue_index])**p
            blue_index += 1

    return matching, cost


@jit
def compute_greedy(X, Y, p=0.5):
    """Compute greedy matching by using a heap to select next closest available pair"""
    matching = []
    cost = 0

    X_rem = list(enumerate(X))
    Y_rem = list(enumerate(Y))
    heap = [(abs(x - y), i, j) for i, x in X_rem for j, y in Y_rem]
    heapq.heapify(heap)
    
    added = 0
    addedSetX = set()
    addedSetY = set()
    while added < len(X):
        dist, i, j = heapq.heappop(heap)
        if i in addedSetX or j in addedSetY:
            continue
        matching.append((i, j))
        added += 1
        addedSetX.add(i)
        addedSetY.add(j)
        cost += dist**p

    return matching, cost


@jit
def compute_hybrid(X, Y, p, k):
    """Apply k greedy matchings globally, then Dyck to residual"""
    n = len(X)
    cost = 0
    matched = []
    
    if k > n:
        raise Exception("Maximum number of greedy steps must be n")
    
    if k < 0:
        raise Exception("Minimum number of greedy steps must be 0")

    X_rem = list(enumerate(X))
    Y_rem = list(enumerate(Y))
    heap = [(abs(x - y), i, j) for i, x in X_rem for j, y in Y_rem]
    heapq.heapify(heap)
    
    # Greedy for k steps
    added = 0
    addedSetX = set()
    addedSetY = set()
    while heap and added < k:
        dist, i, j = heapq.heappop(heap)
        if i in addedSetX or j in addedSetY:
            continue
        matched.append((i, j))
        added += 1
        addedSetX.add(i)
        addedSetY.add(j)
        cost += dist**p
    
    # removes indices in reverse order so that indices don't change mid removal
    for idx in sorted(addedSetX, reverse=True):
        del X_rem[int(idx)]
    for idx in sorted(addedSetY, reverse=True):
        del Y_rem[int(idx)]
    
    # Dyck on whatever is left
    if X_rem and Y_rem:
        X_sub = np.array([x for _, x in X_rem])
        Y_sub = np.array([y for _, y in Y_rem])
        dyck_pairs, dyck_cost = compute_dyck(X_sub, Y_sub, p)
        cost += dyck_cost
        for local_i, local_j in dyck_pairs:
            orig_i = X_rem[local_i][0]
            orig_j = Y_rem[local_j][0]
            matched.append((orig_i, orig_j))

    return matched, cost
  

@jit
def compute_swap_best(X, Y, p, initial_matching):
    """Iteratively perform greedy swaps until no swap improves total cost"""
    history = [initial_matching]
    current_matching = {i: j for i, j in initial_matching}
    num_swaps = 0

    while True:
        pair_costs = [(abs(X[i] - Y[j])**p, i, j) for i, j in current_matching.items()]
        pair_costs.sort(reverse=True)

        best_improvement = 0
        best_swap = None

        for _, i_high, j_high in pair_costs:
            curr_high = abs(X[i_high] - Y[j_high])**p
            for i_other, j_other in current_matching.items():
                if i_other == i_high:
                    continue
                curr_other = abs(X[i_other] - Y[j_other])**p
                new_cost_1 = abs(X[i_high] - Y[j_other])**p
                new_cost_2 = abs(X[i_other] - Y[j_high])**p
                improvement = (curr_high + curr_other) - (new_cost_1 + new_cost_2)

                if improvement > best_improvement:
                    best_improvement = improvement
                    best_swap = (i_high, j_high, i_other, j_other)

        if not best_swap or best_improvement <= 1e-10:
            break

        i_high, j_high, i_other, j_other = best_swap
        current_matching[i_high] = j_other
        current_matching[i_other] = j_high
        num_swaps += 1
        history.append(list(current_matching.items()))

    return list(current_matching.items()), num_swaps, history


@jit
def compute_hybrid_costs_for_all_k(X, Y, p):
    """
    Return the costs of matchings given by hybrid-k for all k from 0 to n
    """
    n = len(X)
    results = []
  
    X_rem = list(X)
    Y_rem = list(Y)
    
    # This heap is where we will grab the next greedy match from
    heap = [(abs(x - y), x, y) for x in X for y in Y]
    heapq.heapify(heap)
    greedyPairs = 0
    greedy_cost = 0
    
    for k in range(n+1):
      # Find the next greedy matching and update greedy_cost
      while greedyPairs < k:
          dist, x, y = heapq.heappop(heap)
          if x in X_rem and y in Y_rem:
              greedy_cost += dist**p
              X_rem.remove(x)
              Y_rem.remove(y)
              greedyPairs += 1

      # Compute dyck_cost for the residual
      dyck_cost = 0
      if k < n:
          _, dyck_cost = compute_dyck(X_rem, Y_rem, p)
      
      results.append(greedy_cost + dyck_cost)

    return results
  