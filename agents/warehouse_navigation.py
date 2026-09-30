from collections import deque

WAREHOUSE = [
    "#####################",
    "#S....#............G#",
    "#.##....##########..#",
    "#....##.............#",
    "#.######.###.#.###..#",
    "#........#..........#",
    "#####################",
]

MOVES = {
    "Up": (-1, 0),
    "Down": (1, 0),
    "Left": (0, -1),
    "Right": (0, 1),
}


def find_position(grid, symbol):
    for row, line in enumerate(grid):
        for col, value in enumerate(line):
            if value == symbol:
                return row, col
    raise ValueError(f"{symbol!r} not found")


def bfs(grid):
    start = find_position(grid, "S")
    goal = find_position(grid, "G")

    queue = deque([start])
    previous = {start: None}
    action_taken = {}

    while queue:
        current = queue.popleft()

        if current == goal:
            break

        for action, (dr, dc) in MOVES.items():
            next_position = (current[0] + dr, current[1] + dc)

            if next_position in previous:
                continue

            row, col = next_position

            if not (0 <= row < len(grid) and 0 <= col < len(grid[0])):
                continue

            if grid[row][col] == "#":
                continue

            previous[next_position] = current
            action_taken[next_position] = action
            queue.append(next_position)

    if goal not in previous:
        return None

    positions = []
    actions = []
    current = goal

    while current is not None:
        positions.append(current)

        if current != start:
            actions.append(action_taken[current])

        current = previous[current]

    positions.reverse()
    actions.reverse()

    return positions, actions


path_result = bfs(WAREHOUSE)

print("WAREHOUSE")
print(*WAREHOUSE, sep="\n")

print("\nSEARCH")
print("algorithm = Breadth-First Search (BFS)")

if path_result is None:
    print("path = None")
    print("path_length = None")
else:
    positions, actions = path_result

    print(f"start = {positions[0]}")
    print(f"goal = {positions[-1]}")
    print(f"path_length = {len(actions)}")
    print(f"actions = {actions}")
    print(f"positions = {positions}")

