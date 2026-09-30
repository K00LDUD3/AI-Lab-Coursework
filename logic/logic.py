"""
AI Logical Planning Lab
Pure-output implementation.

Implements the required Python planning portion:
1. Warehouse problem specification
2. State/action representation
3. Breadth-first-search planner
4. Solvable test
5. Impossible test
6. Irrelevant-action test
7. Plan/state validation
8. Logical applicability checks

No explanatory logging or interpretation messages are printed.
"""

from collections import deque
from dataclasses import dataclass
from typing import FrozenSet, Iterable, Optional, Tuple


# ---------------------------------------------------------------------------
# Logical propositions
# ---------------------------------------------------------------------------

Fact = Tuple[str, str]

State = FrozenSet[Fact]


@dataclass(frozen=True)
class Action:
    name: str
    positive_preconditions: FrozenSet[Fact]
    negative_preconditions: FrozenSet[Fact]
    positive_effects: FrozenSet[Fact]
    negative_effects: FrozenSet[Fact]

    def applicable(self, state: State) -> bool:
        return (
            self.positive_preconditions.issubset(state)
            and self.negative_preconditions.isdisjoint(state)
        )

    def apply(self, state: State) -> State:
        if not self.applicable(state):
            raise ValueError("Action is not applicable")

        new_state = set(state)
        new_state.difference_update(self.negative_effects)
        new_state.update(self.positive_effects)

        return frozenset(new_state)


# ---------------------------------------------------------------------------
# Warehouse problem
# ---------------------------------------------------------------------------

INITIAL_STATE = frozenset(
    {
        ("AtRobot", "A"),
        ("AtPackage", "A"),
    }
)

GOAL = frozenset(
    {
        ("AtPackage", "C"),
    }
)


def make_actions():
    return [
        Action(
            name="Move(A,B)",
            positive_preconditions=frozenset({("AtRobot", "A")}),
            negative_preconditions=frozenset(),
            positive_effects=frozenset({("AtRobot", "B")}),
            negative_effects=frozenset({("AtRobot", "A")}),
        ),
        Action(
            name="Move(B,A)",
            positive_preconditions=frozenset({("AtRobot", "B")}),
            negative_preconditions=frozenset(),
            positive_effects=frozenset({("AtRobot", "A")}),
            negative_effects=frozenset({("AtRobot", "B")}),
        ),
        Action(
            name="Move(B,C)",
            positive_preconditions=frozenset({("AtRobot", "B")}),
            negative_preconditions=frozenset(),
            positive_effects=frozenset({("AtRobot", "C")}),
            negative_effects=frozenset({("AtRobot", "B")}),
        ),
        Action(
            name="Move(C,B)",
            positive_preconditions=frozenset({("AtRobot", "C")}),
            negative_preconditions=frozenset(),
            positive_effects=frozenset({("AtRobot", "B")}),
            negative_effects=frozenset({("AtRobot", "C")}),
        ),
        Action(
            name="PickUp(Package,A)",
            positive_preconditions=frozenset(
                {
                    ("AtRobot", "A"),
                    ("AtPackage", "A"),
                }
            ),
            negative_preconditions=frozenset(
                {
                    ("Holding", "Package"),
                }
            ),
            positive_effects=frozenset(
                {
                    ("Holding", "Package"),
                }
            ),
            negative_effects=frozenset(
                {
                    ("AtPackage", "A"),
                }
            ),
        ),
        Action(
            name="PickUp(Package,B)",
            positive_preconditions=frozenset(
                {
                    ("AtRobot", "B"),
                    ("AtPackage", "B"),
                }
            ),
            negative_preconditions=frozenset(
                {
                    ("Holding", "Package"),
                }
            ),
            positive_effects=frozenset(
                {
                    ("Holding", "Package"),
                }
            ),
            negative_effects=frozenset(
                {
                    ("AtPackage", "B"),
                }
            ),
        ),
        Action(
            name="PickUp(Package,C)",
            positive_preconditions=frozenset(
                {
                    ("AtRobot", "C"),
                    ("AtPackage", "C"),
                }
            ),
            negative_preconditions=frozenset(
                {
                    ("Holding", "Package"),
                }
            ),
            positive_effects=frozenset(
                {
                    ("Holding", "Package"),
                }
            ),
            negative_effects=frozenset(
                {
                    ("AtPackage", "C"),
                }
            ),
        ),
        Action(
            name="Drop(Package,A)",
            positive_preconditions=frozenset(
                {
                    ("AtRobot", "A"),
                    ("Holding", "Package"),
                }
            ),
            negative_preconditions=frozenset(),
            positive_effects=frozenset(
                {
                    ("AtPackage", "A"),
                }
            ),
            negative_effects=frozenset(
                {
                    ("Holding", "Package"),
                }
            ),
        ),
        Action(
            name="Drop(Package,B)",
            positive_preconditions=frozenset(
                {
                    ("AtRobot", "B"),
                    ("Holding", "Package"),
                }
            ),
            negative_preconditions=frozenset(),
            positive_effects=frozenset(
                {
                    ("AtPackage", "B"),
                }
            ),
            negative_effects=frozenset(
                {
                    ("Holding", "Package"),
                }
            ),
        ),
        Action(
            name="Drop(Package,C)",
            positive_preconditions=frozenset(
                {
                    ("AtRobot", "C"),
                    ("Holding", "Package"),
                }
            ),
            negative_preconditions=frozenset(),
            positive_effects=frozenset(
                {
                    ("AtPackage", "C"),
                }
            ),
            negative_effects=frozenset(
                {
                    ("Holding", "Package"),
                }
            ),
        ),
    ]


# ---------------------------------------------------------------------------
# BFS planner
# ---------------------------------------------------------------------------

def goal_reached(state: State, goal: FrozenSet[Fact]) -> bool:
    return goal.issubset(state)


def applicable_actions(
    state: State,
    actions: Iterable[Action],
):
    return [action for action in actions if action.applicable(state)]


def bfs_plan(
    initial_state: State,
    actions: Iterable[Action],
    goal: FrozenSet[Fact],
):
    actions = list(actions)

    frontier = deque([initial_state])
    visited = {initial_state}

    parent = {}
    action_used = {}

    if goal_reached(initial_state, goal):
        return [], [initial_state]

    while frontier:
        state = frontier.popleft()

        for action in actions:
            if not action.applicable(state):
                continue

            successor = action.apply(state)

            if successor in visited:
                continue

            visited.add(successor)
            parent[successor] = state
            action_used[successor] = action
            frontier.append(successor)

            if goal_reached(successor, goal):
                return reconstruct_plan(
                    initial_state,
                    successor,
                    parent,
                    action_used,
                )

    return None, None


def reconstruct_plan(
    initial_state,
    goal_state,
    parent,
    action_used,
):
    actions = []
    states = [goal_state]

    current = goal_state

    while current != initial_state:
        actions.append(action_used[current])
        current = parent[current]
        states.append(current)

    actions.reverse()
    states.reverse()

    return actions, states


# ---------------------------------------------------------------------------
# Plan validation
# ---------------------------------------------------------------------------

def validate_plan(
    initial_state: State,
    actions: Iterable[Action],
    goal: FrozenSet[Fact],
):
    state = initial_state
    states = [state]

    for action in actions:
        if not action.applicable(state):
            return False, states

        state = action.apply(state)
        states.append(state)

    return goal_reached(state, goal), states


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------

def fact_key(fact):
    return (fact[0], fact[1])


def state_string(state):
    ordered = sorted(state, key=fact_key)
    return "{" + ", ".join(
        f"{predicate}({value})"
        for predicate, value in ordered
    ) + "}"


def print_state_sequence(states):
    for i, state in enumerate(states):
        print(f"S{i}={state_string(state)}")


def print_plan(actions):
    if actions is None:
        print("plan=None")
        return

    print(
        "plan=[" +
        ", ".join(action.name for action in actions) +
        "]"
    )


def print_test_result(
    name,
    initial_state,
    goal,
    actions,
    states,
    valid,
):
    print(name)
    print(f"initial={state_string(initial_state)}")
    print(f"goal={state_string(goal)}")
    print(f"found={actions is not None}")
    print_plan(actions)

    if states is not None:
        print_state_sequence(states)

    print(f"valid={valid}")
    print()


# ---------------------------------------------------------------------------
# Task 0: Problem specification
# ---------------------------------------------------------------------------

def run_problem_specification():
    print("PROBLEM_SPECIFICATION")
    print(f"initial={state_string(INITIAL_STATE)}")
    print(f"goal={state_string(GOAL)}")

    print(
        "actions=[" +
        ", ".join(action.name for action in make_actions()) +
        "]"
    )

    initially_applicable = applicable_actions(
        INITIAL_STATE,
        make_actions(),
    )

    print(
        "initially_applicable=[" +
        ", ".join(action.name for action in initially_applicable) +
        "]"
    )

    print()


# ---------------------------------------------------------------------------
# Task 1: Construct a plan
# ---------------------------------------------------------------------------

def run_original():
    actions = make_actions()

    plan, states = bfs_plan(
        INITIAL_STATE,
        actions,
        GOAL,
    )

    valid, validated_states = validate_plan(
        INITIAL_STATE,
        plan if plan is not None else [],
        GOAL,
    )

    print_test_result(
        "SOLVABLE",
        INITIAL_STATE,
        GOAL,
        plan,
        validated_states if plan is not None else None,
        valid,
    )


# ---------------------------------------------------------------------------
# Task 3A: Solvable problem
# ---------------------------------------------------------------------------

def run_solvable_test():
    actions = make_actions()

    plan, states = bfs_plan(
        INITIAL_STATE,
        actions,
        GOAL,
    )

    valid, _ = validate_plan(
        INITIAL_STATE,
        plan if plan is not None else [],
        GOAL,
    )

    print_test_result(
        "TEST_A",
        INITIAL_STATE,
        GOAL,
        plan,
        states,
        valid,
    )


# ---------------------------------------------------------------------------
# Task 3B: Impossible problem
# ---------------------------------------------------------------------------

def run_impossible_test():
    actions = [
        action
        for action in make_actions()
        if not action.name.startswith("PickUp")
    ]

    plan, states = bfs_plan(
        INITIAL_STATE,
        actions,
        GOAL,
    )

    valid = False

    if plan is not None:
        valid, _ = validate_plan(
            INITIAL_STATE,
            plan,
            GOAL,
        )

    print_test_result(
        "TEST_B",
        INITIAL_STATE,
        GOAL,
        plan,
        states,
        valid,
    )


# ---------------------------------------------------------------------------
# Task 3C: Irrelevant action
# ---------------------------------------------------------------------------

def run_irrelevant_action_test():
    actions = make_actions()

    irrelevant_action = Action(
        name="Move(A,B)_Irrelevant",
        positive_preconditions=frozenset(
            {
                ("AtRobot", "A"),
            }
        ),
        negative_preconditions=frozenset(),
        positive_effects=frozenset(
            {
                ("AtRobot", "B"),
            }
        ),
        negative_effects=frozenset(
            {
                ("AtRobot", "A"),
            }
        ),
    )

    actions_with_irrelevant = actions + [irrelevant_action]

    plan, states = bfs_plan(
        INITIAL_STATE,
        actions_with_irrelevant,
        GOAL,
    )

    valid, _ = validate_plan(
        INITIAL_STATE,
        plan if plan is not None else [],
        GOAL,
    )

    reaches_package_goal = (
        states is not None
        and goal_reached(states[-1], GOAL)
    )

    print("TEST_C")
    print(f"found={plan is not None}")
    print_plan(plan)

    if states is not None:
        print_state_sequence(states)

    print(f"valid={valid}")
    print(f"package_goal_reached={reaches_package_goal}")
    print()


# ---------------------------------------------------------------------------
# Explicit applicability checks
# ---------------------------------------------------------------------------

def run_applicability_checks():
    actions = make_actions()

    pickup_a = next(
        action
        for action in actions
        if action.name == "PickUp(Package,A)"
    )

    drop_c = next(
        action
        for action in actions
        if action.name == "Drop(Package,C)"
    )

    print("APPLICABILITY")
    print(
        f"PickUp(Package,A)="
        f"{pickup_a.applicable(INITIAL_STATE)}"
    )
    print(
        f"Drop(Package,C)="
        f"{drop_c.applicable(INITIAL_STATE)}"
    )
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    run_problem_specification()
    run_applicability_checks()
    run_original()
    run_solvable_test()
    run_impossible_test()
    run_irrelevant_action_test()


if __name__ == "__main__":
    main()
