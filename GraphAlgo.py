# Unit-III Graph Algorithms: Primal-Dual algorithm and its application to shortest path (Dijkstra’s algorithm, Floyd-Warshall algorithms), max-flow problem (Ford and Fulkerson labeling algorithms), matching problems (bipartite matching algorithm, non-bipartite matching algorithms, bipartite weighted matching-hungarian method for the assignment problem, non-bipartite weighted matching problem), efficient spanning tree algorithms.

import heapq
import numpy as np
from collections import deque

INF = float('inf')

# 1. Dijkstra's Algorithm (supports infinite weights)
def dijkstra(graph, start):
    n = len(graph)
    dist = [INF] * n
    dist[start] = 0
    pq = [(0, start)]
    while pq:
        cur_dist, u = heapq.heappop(pq)
        if cur_dist > dist[u]:
            continue
        for v, w in graph[u]:
            if w == INF:
                continue  # no edge
            if dist[v] > dist[u] + w:
                dist[v] = dist[u] + w
                heapq.heappush(pq, (dist[v], v))
    return dist

graph_example = [
    [(1, 4), (2, 1)],
    [(3, 1)],
    [(1, 2), (3, INF)],  # infinite weight edge
    []
]
print("Dijkstra shortest paths from node 0:", dijkstra(graph_example, 0))


# 2. Floyd-Warshall Algorithm (all pairs shortest paths with infinity)
def floyd_warshall(dist):
    n = len(dist)
    for k in range(n):
        for i in range(n):
            for j in range(n):
                if dist[i][k] == INF or dist[k][j] == INF:
                    continue
                if dist[i][j] > dist[i][k] + dist[k][j]:
                    dist[i][j] = dist[i][k] + dist[k][j]
    return dist

fw_graph = [
    [0, 3, INF, 7],
    [8, 0, 2, INF],
    [5, INF, 0, 1],
    [2, INF, INF, 0]
]
fw_result = floyd_warshall(fw_graph)
print("\nFloyd-Warshall all pairs shortest path:")
for row in fw_result:
    print(row)


# 3. Ford-Fulkerson Max Flow (with DFS), capacities can be INF
def ford_fulkerson(capacity, source, sink):
    n = len(capacity)
    flow = 0

    def dfs(u, flow_in, visited):
        if u == sink:
            return flow_in
        visited[u] = True
        for v in range(n):
            cap = capacity[u][v]
            if cap > 0 and not visited[v]:
                pushed = dfs(v, min(flow_in, cap), visited)
                if pushed > 0:
                    capacity[u][v] -= pushed
                    capacity[v][u] += pushed
                    return pushed
        return 0

    while True:
        visited = [False] * n
        pushed = dfs(source, INF, visited)
        if pushed == 0:
            break
        flow += pushed
    return flow

capacity = [
    [0, 16, 13, 0, 0, 0],
    [0, 0, 10, 12, 0, 0],
    [0, 4, 0, 0, 14, 0],
    [0, 0, 9, 0, 0, 20],
    [0, 0, 0, 7, 0, 4],
    [0, 0, 0, 0, 0, 0]
]
maxflow = ford_fulkerson([row[:] for row in capacity], 0, 5)
print("\nFord-Fulkerson max flow:", maxflow)


# 4. Hungarian Algorithm for Bipartite Weighted Matching (infinite cost means no assignment)
def hungarian_algorithm(cost_matrix):
    from scipy.optimize import linear_sum_assignment
    cost_matrix = np.array(cost_matrix)
    # Replace INF with a large number for scipy compatibility
    large_num = 10**9
    cost_matrix = np.where(cost_matrix == INF, large_num, cost_matrix)
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    total_cost = cost_matrix[row_ind, col_ind].sum()
    return list(zip(row_ind, col_ind)), total_cost

cost = [
    [4, 1, 3],
    [2, 0, INF],  # INF means no edge/assignment possible
    [3, 2, 2]
]
matching, cost_total = hungarian_algorithm(cost)
print("\nHungarian matching pairs and cost:", matching, cost_total)


# 5. Blossom Algorithm (Non-bipartite matching) - same as before (no infinite edges here)
class Blossom:
    def __init__(self, n):
        self.n = n
        self.adj = [[] for _ in range(n)]
        self.match = [-1]*n
        self.parent = [-1]*n
        self.base = list(range(n))
        self.q = []
        self.in_queue = [False]*n
        self.in_blossom = [False]*n

    def lca(self, a, b):
        visited = [False]*self.n
        while True:
            a = self.base[a]
            visited[a] = True
            if self.match[a] == -1:
                break
            a = self.parent[self.match[a]]
        while True:
            b = self.base[b]
            if visited[b]:
                return b
            b = self.parent[self.match[b]]

    def mark_blossom(self, a, b, lca):
        while self.base[a] != lca:
            self.in_blossom[self.base[a]] = self.in_blossom[self.base[self.match[a]]] = True
            self.parent[a] = b
            b = self.match[a]
            a = self.parent[b]

    def find_augmenting_path(self, root):
        self.in_queue = [False]*self.n
        self.parent = [-1]*self.n
        self.base = list(range(self.n))
        self.q = []
        self.q.append(root)
        self.in_queue[root] = True
        while self.q:
            u = self.q.pop(0)
            for v in self.adj[u]:
                if self.base[u] == self.base[v] or self.match[u] == v:
                    continue
                if v == root or (self.match[v] != -1 and self.parent[self.match[v]] != -1):
                    lca = self.lca(u, v)
                    self.in_blossom = [False]*self.n
                    self.mark_blossom(u, v, lca)
                    self.mark_blossom(v, u, lca)
                    for i in range(self.n):
                        if self.in_blossom[self.base[i]]:
                            if not self.in_queue[i]:
                                self.q.append(i)
                                self.in_queue[i] = True
                elif self.parent[v] == -1:
                    self.parent[v] = u
                    if self.match[v] == -1:
                        return v
                    v_ = self.match[v]
                    self.q.append(v_)
                    self.in_queue[v_] = True
        return -1

    def augment_path(self, start, finish):
        v = finish
        while v != -1:
            pv = self.parent[v]
            ppv = self.match[pv] if pv != -1 else -1
            self.match[v] = pv
            self.match[pv] = v
            v = ppv

    def edmonds(self):
        for i in range(self.n):
            if self.match[i] == -1:
                finish = self.find_augmenting_path(i)
                if finish != -1:
                    self.augment_path(i, finish)
        result = []
        for i in range(self.n):
            if self.match[i] != -1 and i < self.match[i]:
                result.append((i, self.match[i]))
        return result

blossom = Blossom(6)
edges = [(0,1),(0,2),(1,2),(1,3),(3,4),(4,5)]
for u,v in edges:
    blossom.adj[u].append(v)
    blossom.adj[v].append(u)
matching_nb = blossom.edmonds()
print("\nNon-bipartite matching edges (Blossom algorithm):", matching_nb)


# 6. Kruskal's MST (infinite weights = no edge)
class DisjointSet:
    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [0]*n

    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, x, y):
        rx, ry = self.find(x), self.find(y)
        if rx == ry:
            return False
        if self.rank[rx] < self.rank[ry]:
            self.parent[rx] = ry
        elif self.rank[ry] < self.rank[rx]:
            self.parent[ry] = rx
        else:
            self.parent[ry] = rx
            self.rank[rx] += 1
        return True

def kruskal(edges, n):
    ds = DisjointSet(n)
    mst = []
    edges = [e for e in edges if e[2] != INF]
    edges = sorted(edges, key=lambda x: x[2])
    for u,v,w in edges:
        if ds.union(u,v):
            mst.append((u,v,w))
    return mst

edges = [
    (0,1,10),
    (0,2,6),
    (0,3,5),
    (1,3,INF),  # infinite weight = no edge
    (2,3,4)
]
mst = kruskal(edges, 4)
print("\nKruskal MST edges:", mst)
