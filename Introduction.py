# Unit-I Introduction: Optimization problems, neighborhoods, local and global optima, convex sets and functions, simplex method, degeneracy; duality and dual simplex algorithm, computational considerations for the simplex and dual simplex algorithms-Dantzig-Wolfe algorithms.

import numpy as np
from scipy.optimize import linprog

# --- 1. Optimization Problem, Neighborhood, Local & Global Optima ---
# Example: simple combinatorial optimization - max sum of subset under constraints

def is_local_optimum(x, neighbors, f):
    """Check if x is a local optimum compared to its neighbors for function f."""
    fx = f(x)
    return all(fx >= f(n) for n in neighbors)

def example_subset_sum_problem():
    # Given set, maximize sum subset under constraint sum <= 10
    S = [2, 3, 5, 7, 1]
    f = lambda x: sum(x)  # objective function: sum of chosen elements
    
    # Neighborhood: subsets differing by one element
    def neighbors(x):
        neighbors = []
        for i in range(len(x)):
            neighbor = x.copy()
            neighbor[i] = 1 - neighbor[i]  # flip bit
            if sum(np.array(neighbor) * np.array(S)) <= 10:
                neighbors.append(neighbor)
        return neighbors

    # Start at some solution
    x = [0, 0, 0, 0, 0]
    for _ in range(10):
        neigh = neighbors(x)
        # move to better neighbor if exists
        better = [n for n in neigh if f(np.array(n) * np.array(S)) > f(np.array(x) * np.array(S))]
        if better:
            x = better[0]
        else:
            break
    print("Local optimum subset:", x)
    print("Value:", f(np.array(x) * np.array(S)))

example_subset_sum_problem()

# --- 2. Convex Sets and Functions ---
def is_convex_function(f, domain_points):
    """Check convexity for a 1D function on a set of points."""
    for i in range(len(domain_points) - 2):
        x1, x2, x3 = domain_points[i], domain_points[i+1], domain_points[i+2]
        f1, f2, f3 = f(x1), f(x2), f(x3)
        # Convexity check: f2 <= (f1 + f3)/2
        if f2 > (f1 + f3)/2:
            return False
    return True

f_convex = lambda x: x**2  # convex function
domain = np.linspace(-10,10,100)
print("Is x^2 convex?", is_convex_function(f_convex, domain))

# --- 3. Simplex Method and Degeneracy ---
# Solve LP: maximize c^T x subject to Ax <= b, x >= 0
c = [-3, -5]  # maximize 3x + 5y = minimize -3x -5y
A = [[1, 0], [0, 2], [3, 2]]
b = [4, 12, 18]

res = linprog(c, A_ub=A, b_ub=b, method='simplex')
print("Simplex solution:", res.x)

# Degeneracy can occur when multiple constraints intersect on the same vertex.
# SciPy's linprog handles it internally.

# --- 4. Duality and Dual Simplex ---
# Dual of above problem
# Primal: max c^T x, s.t. Ax <= b
# Dual: min b^T y, s.t. A^T y >= c, y >= 0

# Using linprog on dual (just as illustration)
c_dual = b
A_dual = np.transpose(A)
b_dual = -np.array(c)
res_dual = linprog(c_dual, A_ub=-A_dual, b_ub=b_dual, method='simplex')
print("Dual solution:", res_dual.x)

# --- 5. Dantzig-Wolfe Decomposition (illustration) ---
# For large LP with block structure, decompose and solve iteratively
# Here, a mockup since full implementation is complex

def dantzig_wolfe_decomposition():
    print("\nDantzig-Wolfe decomposition illustration:")
    print("Suppose a problem with complicating constraints is decomposed into subproblems.")
    print("This is typically applied in large-scale LP, e.g., multicommodity flow.")

dantzig_wolfe_decomposition()
