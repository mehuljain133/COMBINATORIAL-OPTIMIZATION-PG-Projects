# Unit-IV Matroids: Independence Systems and Matroids, Duality, Matroid Intersection.

# Unit-IV: Matroids - Independence Systems, Duality, and Intersection

from typing import List, Set

# Independence System Base Class
class IndependenceSystem:
    def __init__(self, ground_set: Set[int]):
        self.ground_set = ground_set

    def is_independent(self, subset: Set[int]) -> bool:
        """Check if subset is independent in this system"""
        raise NotImplementedError


# Example: Uniform Matroid of rank k
class UniformMatroid(IndependenceSystem):
    def __init__(self, ground_set: Set[int], rank: int):
        super().__init__(ground_set)
        self.rank = rank

    def is_independent(self, subset: Set[int]) -> bool:
        return len(subset) <= self.rank


# Matroid Dual of a Uniform Matroid
class UniformMatroidDual(IndependenceSystem):
    def __init__(self, uniform_matroid: UniformMatroid):
        super().__init__(uniform_matroid.ground_set)
        self.original = uniform_matroid
        self.rank = len(self.ground_set) - uniform_matroid.rank

    def is_independent(self, subset: Set[int]) -> bool:
        # In dual, subset is independent if complement contains a basis of original matroid
        complement = self.ground_set - subset
        return len(complement) >= self.original.rank


# Matroid Intersection Algorithm
# Finds maximum cardinality set independent in both matroids
def matroid_intersection(matroid1: IndependenceSystem, matroid2: IndependenceSystem) -> Set[int]:
    E = matroid1.ground_set
    I = set()  # current independent set
    
    # Augmentation via BFS on exchange graph
    while True:
        # Build bipartite exchange graph edges
        # X = elements not in I
        X = E - I
        
        # Exchange graph: edges from X to I where adding/removing maintains independence
        # Edges from X -> I if I+{x}-{y} independent in both matroids
        prev = {e: None for e in E}
        dist = {e: None for e in E}
        
        # Start BFS from all x in X where I+{x} independent in both matroids
        queue = []
        for x in X:
            if matroid1.is_independent(I | {x}) and matroid2.is_independent(I | {x}):
                dist[x] = 0
                queue.append(x)

        found_augmenting = None

        while queue and found_augmenting is None:
            u = queue.pop(0)
            if u in I:
                # u in I: edges to X \ I
                for x in X:
                    if prev[x] is None and matroid1.is_independent((I - {u}) | {x}) and matroid2.is_independent((I - {u}) | {x}):
                        dist[x] = dist[u] + 1
                        prev[x] = u
                        queue.append(x)
            else:
                # u in X: if I + u independent in both matroids -> augment
                if u not in I:
                    found_augmenting = u
                    break
                else:
                    # edges to I \ I (no edges actually here, can be omitted)
                    pass

        if found_augmenting is None:
            break

        # Augment along path
        x = found_augmenting
        while x is not None:
            if x in I:
                I.remove(x)
            else:
                I.add(x)
            x = prev[x]

    return I


# Test the code:

ground_set = set(range(1, 7))
k = 3
m1 = UniformMatroid(ground_set, k)       # Independent sets size ≤ 3
m2 = UniformMatroidDual(m1)              # Dual matroid

print("Ground set:", ground_set)
print("Uniform matroid rank:", k)
print("Uniform matroid dual rank:", len(ground_set) - k)

# Test independence
test_set = {1, 2}
print("Is", test_set, "independent in m1?", m1.is_independent(test_set))
print("Is", test_set, "independent in m2?", m2.is_independent(test_set))

# Matroid intersection (maximum set independent in both)
max_indep_set = matroid_intersection(m1, m2)
print("Maximum independent set in both matroids:", max_indep_set)
print("Size of maximum independent set:", len(max_indep_set))
