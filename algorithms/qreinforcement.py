import numpy as np 


def get_available_actions(R, state):
    current_state_row = R[state,]
    actions_with_max_reward = np.max(current_state_row)
    action_indices = np.where(actions_with_max_reward == current_state_row)[1]
    return action_indices

def update_brain(Q, current_state, action, gamma):
    # May be many indices if same value in multiple indices
    max_indices = np.where(Q[action,]== np.max(Q[action,]))[1]
    # Since each index leads to a different state randomize the selection of state
    max_index = int(np.random.choice(max_indices, size=1))
    # Q learning formula
    Q[current_state,action] = R[current_state,action] + gamma * Q[action, max_index]


def get_random_state(Q):
    return np.random.randint(0, int(Q.shape[0]))

def get_random_state_excluding_val(Q, vals = []):
    result = get_random_state(Q)
    while result not in vals:
        result = get_random_state(Q)
    return result

def train(Q, R, learning_rate=0.2, n_epochs = 500):
    brain = Q.copy()
    gamma = learning_rate
    # Training of the algorithm
    for _i in range(n_epochs):
        current_state = get_random_state(brain)
        available_actions = get_available_actions(R, current_state)
        action = int(np.random.choice(available_actions, 1))
        update_brain(brain, current_state, action, gamma)

    return brain

def predict_path(Q, current_state, end_state = 5):
    # Maximum assumes we can't have more states than number of iteration
    # therefore exit since path not found
    max_number_of_iterations = Q.shape[0]
    steps = [current_state]
    for _ in range(max_number_of_iterations):
        if current_state == end_state:
            break
        possible_next_step_indices = np.where(Q[current_state,]== np.max(Q[current_state,]))[1]
        current_state = int(np.random.choice(possible_next_step_indices, size=1))
        steps.append(current_state)

    return steps

R = np.matrix([[-1, -1, -1, -1, 0, -1],
                [-1, -1, -1, 0, -1, 100],
                [-1, -1, -1, 0, -1, -1],
                [-1, 0, 0, -1, 0, -1],
                [-1, 0, 0, -1, 0, 100],
                [-1, 0, -1, -1, 0, 100],])
Q = np.matrix(np.zeros([6,6]))
trained_brain = train(Q, R, learning_rate=0.2, n_epochs=150)
# Display the trained Q matrix
print("Original Q matrix")
print(Q)
print("Trained Q matrix")
print(trained_brain / np.max(trained_brain) * 100)

# Q Matix is optimized to lead to state 5
# Testing all paths leading to state 5
goal_state = 5
for current_state in range(6):
    print("Starting point: ", current_state)
    print("Goal: ", goal_state)
    steps = predict_path(trained_brain, current_state, end_state=goal_state)
    print("Selected path: ", steps)