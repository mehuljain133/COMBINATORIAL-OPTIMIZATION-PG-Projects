# Unit-II Integer Linear Programming: Cutting plane algorithms, branch and bound technique and approximation algorithms for traveling salesman problem

import numpy as np
from scipy.optimize import linprog

# --- 1. Integer Linear Programming (ILP) via Branch and Bound ---

# Simple ILP example:
# maximize z = x + y
# subject to:
#   2x + y <= 4
#   x + 2y <= 5
#   x,y >= 0 and integer

def ilp_branch_and_bound():
    best_solution = None
    best_obj = -np.inf

    def branch(bounds):
        nonlocal best_solution, best_obj

        # Solve LP relaxation
        c = [-1, -1]  # maximize x+y -> minimize -x -y
        A = [[2,1],[1,2]]
        b = [4,5]
        bounds_local = bounds
        res = linprog(c, A_ub=A, b_ub=b, bounds=bounds_local, method='highs')
        if not res.success:
            return
        x = res.x
        obj = -res.fun

        # If solution is integer, update best solution
        if all(np.isclose(x, np.round(x), atol=1e-5)):
            if obj > best_obj:
                best_obj = obj
                best_solution = np.round(x).astype(int)
            return

        # Else branch on first fractional variable
        for i, val in enumerate(x):
            if not np.isclose(val, round(val), atol=1e-5):
                break
        else:
            return  # all integers

        # Branch 1: x_i <= floor(val)
        bounds1 = bounds_local.copy()
        bounds1[i] = (bounds_local[i][0], np.floor(val))
        if bounds1[i][0] <= bounds1[i][1]:
            branch(bounds1)

        # Branch 2: x_i >= ceil(val)
        bounds2 = bounds_local.copy()
        bounds2[i] = (np.ceil(val), bounds_local[i][1])
        if bounds2[i][0] <= bounds2[i][1]:
            branch(bounds2)

    # Initial bounds: x,y >= 0
    bounds = [(0, None), (0, None)]
    branch(bounds)
    print(f"ILP Branch and Bound solution: x={best_solution}, objective={best_obj}")

ilp_branch_and_bound()


# --- 2. Cutting Plane Algorithm (Gomory cuts illustration) ---

def cutting_plane_example():
    # Solve ILP:
    # maximize z = x + y
    # subject to:
    #   2x + y <= 4.5
    #   x + 3y <= 6
    #   x,y >= 0 integer

    c = [-1, -1]
    A = [[2,1],[1,3]]
    b = [4.5, 6]
    bounds = [(0,None),(0,None)]

    # Solve LP relaxation
    res = linprog(c, A_ub=A, b_ub=b, bounds=bounds, method='highs')
    x = res.x
    print("LP relaxation solution:", x)

    # If fractional, add Gomory cut (illustration only)
    frac_var_index = None
    for i, val in enumerate(x):
        if not np.isclose(val, round(val), atol=1e-5):
            frac_var_index = i
            break

    if frac_var_index is None:
        print("Solution integer, no cut needed.")
        return

    print(f"Fractional variable at index {frac_var_index} with value {x[frac_var_index]}")

    # Gomory cut for demonstration: Add constraint x_i <= floor(value)
    floor_val = np.floor(x[frac_var_index])
    if floor_val < x[frac_var_index]:
        print(f"Adding cut: x_{frac_var_index+1} <= {floor_val}")
        A_new = np.vstack([A, np.eye(2)[frac_var_index]])
        b_new = np.append(b, floor_val)

        # Re-solve LP with cut
        res_new = linprog(c, A_ub=A_new, b_ub=b_new, bounds=bounds, method='highs')
        print("Solution after cut:", res_new.x)
    else:
        print("No cut added")

cutting_plane_example()


# --- 3. Approximation Algorithm for TSP (Nearest Neighbor) ---

def tsp_nearest_neighbor(dist_matrix):
    n = len(dist_matrix)
    visited = [False]*n
    path = [0]  # start at node 0
    visited[0] = True

    for _ in range(n-1):
        last = path[-1]
        next_city = None
        min_dist = float('inf')
        for j in range(n):
            if not visited[j] and dist_matrix[last][j] < min_dist:
                min_dist = dist_matrix[last][j]
                next_city = j
        path.append(next_city)
        visited[next_city] = True

    path.append(0)  # return to start
    return path

def tsp_path_length(path, dist_matrix):
    length = 0
    for i in range(len(path)-1):
        length += dist_matrix[path[i]][path[i+1]]
    return length

def example_tsp():
    dist = [
        [0, 10, 15, 20],
        [10, 0, 35, 25],
        [15, 35, 0, 30],
        [20, 25, 30, 0]
    ]
    path = tsp_nearest_neighbor(dist)
    length = tsp_path_length(path, dist)
    print("TSP approximate path:", path)
    print("Total path length:", length)

example_tsp()
