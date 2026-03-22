import random

# -------------------------
# Actions
# -------------------------
actions = ["increase_angle", "increase_hold", "add_vibration"]

# -------------------------
# Q-table
# -------------------------
q_table = {}

alpha = 0.1
gamma = 0.9
epsilon = 0.2

# -------------------------
# State Definition
# -------------------------
def get_state(leftover):
    if leftover < 5:
        return "low"
    elif leftover < 10:
        return "medium"
    else:
        return "high"

# -------------------------
# Choose Action
# -------------------------
def choose_action(state):
    if random.uniform(0, 1) < epsilon:
        return random.choice(actions)

    if state not in q_table or not q_table[state]:
        return random.choice(actions)

    return max(q_table[state], key=q_table[state].get)

# -------------------------
# Update Q-table
# -------------------------
def update_q(state, action, reward):
    if state not in q_table:
        q_table[state] = {}

    old_value = q_table[state].get(action, 0)
    new_value = old_value + alpha * (reward - old_value)

    q_table[state][action] = new_value

# -------------------------
# TRAIN RL MODEL
# -------------------------
for _ in range(2000):  # more training
    leftover = random.uniform(0, 20)
    state = get_state(leftover)

    action = choose_action(state)

    # Reward logic (LESS leftover = better)
    reward = -leftover

    update_q(state, action, reward)

# -------------------------
# FINAL FUNCTION (used in app)
# -------------------------
def get_best_action(leftover):
    state = get_state(leftover)

    if state not in q_table or not q_table[state]:
        return "increase_angle"

    return max(q_table[state], key=q_table[state].get)