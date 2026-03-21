import random

# Actions
actions = ["increase_angle", "increase_hold", "add_vibration"]

# Q-table
q_table = {}

alpha = 0.1
epsilon = 0.2

# -----------------------------
def get_state(leftover):
    if leftover < 5:
        return "low"
    elif leftover < 10:
        return "medium"
    else:
        return "high"

# -----------------------------
def choose_action(state):
    if random.uniform(0, 1) < epsilon:
        return random.choice(actions)

    return max(
        q_table.get(state, {}),
        key=q_table.get(state, {}).get,
        default=random.choice(actions)
    )

# -----------------------------
def update_q(state, action, reward):
    if state not in q_table:
        q_table[state] = {}

    old_value = q_table[state].get(action, 0)
    new_value = old_value + alpha * (reward - old_value)

    q_table[state][action] = new_value

# -----------------------------
# TRAIN MODEL ON IMPORT
# -----------------------------
for _ in range(1000):
    leftover = random.uniform(0, 20)
    state = get_state(leftover)
    action = choose_action(state)

    reward = -leftover
    update_q(state, action, reward)

# -----------------------------
# FINAL FUNCTION (IMPORTANT)
# -----------------------------
def get_best_action(leftover):
    state = get_state(leftover)
    actions_for_state = q_table.get(state, {})

    if not actions_for_state:
        return "No suggestion"

    return max(actions_for_state, key=actions_for_state.get)