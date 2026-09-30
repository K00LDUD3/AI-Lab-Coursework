"""
AI Search Lab - Search and A*

Pure-output implementation.

Implements:
1. A* on the original warehouse
2. Required validation tests
3. BFS vs A*
4. Heuristic investigation:
   - Manhattan
   - Zero
   - Euclidean
   - 2 * Manhattan
5. Numerical results only

Requirements:
    Python 3

Run:
    python search_lab.py
"""

from heapq import heappush, heappop
from collections import deque
from math import sqrt


# Maps

ORIGINAL_MAP = [
    "#################",
    "#S....#.........#",
    "#.###.#.#######.#",
    "#...#.#.......#.#",
    "###.#.#######.#.#",
    "#...#.........#.#",
    "#.###########.#.#",
    "#.............#G#",
    "#################",
]

TRIVIAL_MAP = [
    "#####",
    "#SG##",
    "#####",
]

NO_SOLUTION_MAP = [
    "#######",
    "#S....#",
    "###.###",
    "#...#G#",
    "#######",
]

ALTERNATIVE_MAP = [
    "#######",
    "#S....#",
    "#.###.#",
    "#.....#",
    "#.###.#",
    "#....G#",
    "#######",
]


# Search utilities

MOVES = (
    (-1, 0),  # Up
    (1, 0),   # Down
    (0, -1),  # Left
    (0, 1),   # Right
)


def find_positions(grid):
    start = None
    goal = None

    for r, row in enumerate(grid):
        for c, cell in enumerate(row):
            if cell == "S":
                start = (r, c)
            elif cell == "G":
                goal = (r, c)

    return start, goal


def valid_neighbors(grid, state):
    r, c = state

    for dr, dc in MOVES:
        nr = r + dr
        nc = c + dc

        if (
            0 <= nr < len(grid)
            and 0 <= nc < len(grid[nr])
            and grid[nr][nc] != "#"
        ):
            yield (nr, nc)


def reconstruct_path(parent, start, goal):
    if goal not in parent and goal != start:
        return None

    path = []
    current = goal

    while current != start:
        path.append(current)
        current = parent[current]

    path.append(start)
    path.reverse()

    return path


def path_length(path):
    if path is None:
        return None
    return len(path) - 1


# Heuristics
def manhattan(state, goal):
    return abs(state[0] - goal[0]) + abs(state[1] - goal[1])


def zero_heuristic(state, goal):
    return 0


def euclidean(state, goal):
    return sqrt(
        (state[0] - goal[0]) ** 2
        + (state[1] - goal[1]) ** 2
    )


def double_manhattan(state, goal):
    return 2 * manhattan(state, goal)


# A*
def astar(grid, heuristic=manhattan):
    start, goal = find_positions(grid)

    if start is None or goal is None:
        return None, 0

    frontier = []
    counter = 0

    heappush(
        frontier,
        (heuristic(start, goal), counter, start),
    )

    g_cost = {start: 0}
    parent = {}
    closed = set()
    expanded = 0

    while frontier:
        _, _, current = heappop(frontier)

        if current in closed:
            continue

        closed.add(current)
        expanded += 1

        if current == goal:
            return reconstruct_path(parent, start, goal), expanded

        for neighbor in valid_neighbors(grid, current):
            if neighbor in closed:
                continue

            tentative_g = g_cost[current] + 1

            if (
                neighbor not in g_cost
                or tentative_g < g_cost[neighbor]
            ):
                g_cost[neighbor] = tentative_g
                parent[neighbor] = current

                counter += 1
                f_cost = (
                    tentative_g
                    + heuristic(neighbor, goal)
                )

                heappush(
                    frontier,
                    (f_cost, counter, neighbor),
                )

    return None, expanded


# BFS

def bfs(grid):
    start, goal = find_positions(grid)

    if start is None or goal is None:
        return None, 0

    frontier = deque([start])
    parent = {}
    visited = {start}
    expanded = 0

    while frontier:
        current = frontier.popleft()
        expanded += 1

        if current == goal:
            return reconstruct_path(parent, start, goal), expanded

        for neighbor in valid_neighbors(grid, current):
            if neighbor in visited:
                continue

            visited.add(neighbor)
            parent[neighbor] = current
            frontier.append(neighbor)

    return None, expanded


# Output helpers
def path_string(path):
    if path is None:
        return "None"
    return " ".join(f"({r},{c})" for r, c in path)


def print_result(name, path, expanded):
    found = path is not None
    length = path_length(path)

    print(
        f"{name} | "
        f"found={found} | "
        f"path_length={length} | "
        f"states_expanded={expanded}"
    )

    if path is not None:
        print(f"path={path_string(path)}")


# Task 2 / Task 3: Original warehouse
def run_original():
    print("ORIGINAL")

    path, expanded = astar(ORIGINAL_MAP, manhattan)
    print_result("A*", path, expanded)

    print()


# Task 3: Validation tests
def run_validation_tests():
    print("VALIDATION")

    path, expanded = astar(TRIVIAL_MAP, manhattan)
    print_result("trivial", path, expanded)

    path, expanded = astar(NO_SOLUTION_MAP, manhattan)
    print_result("no_solution", path, expanded)

    path, expanded = astar(ALTERNATIVE_MAP, manhattan)
    print_result("alternative", path, expanded)

    print()


# Task 5: BFS vs A*
def run_bfs_vs_astar():
    print("BFS_VS_ASTAR")

    bfs_path, bfs_expanded = bfs(ORIGINAL_MAP)
    astar_path, astar_expanded = astar(
        ORIGINAL_MAP,
        manhattan,
    )

    print_result(
        "BFS",
        bfs_path,
        bfs_expanded,
    )

    print_result(
        "A*",
        astar_path,
        astar_expanded,
    )

    print(
        f"same_path_length="
        f"{path_length(bfs_path) == path_length(astar_path)}"
    )

    print()


# Task 6: Heuristic investigation

def run_heuristic_investigation():
    print("HEURISTICS")

    heuristics = [
        ("manhattan", manhattan),
        ("zero", zero_heuristic),
        ("euclidean", euclidean),
        ("double_manhattan", double_manhattan),
    ]

    for name, heuristic in heuristics:
        path, expanded = astar(
            ORIGINAL_MAP,
            heuristic,
        )

        print_result(
            name,
            path,
            expanded,
        )

    print()


# Search-problem specification

def run_problem_specification():
    print("PROBLEM_SPECIFICATION")

    print("state=(row,column)")
    print("actions=up,down,left,right")
    print("transition=move_to_adjacent_non_obstacle_cell")
    print("initial_state=S")
    print("goal_state=G")
    print("step_cost=1")

    print()


# Main

def main():
    run_problem_specification()
    run_original()
    run_validation_tests()
    run_bfs_vs_astar()
    run_heuristic_investigation()


if __name__ == "__main__":
    main()

