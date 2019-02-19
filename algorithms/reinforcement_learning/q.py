policy = None
punishment = None
reward = None

# agent's brain/state
Q = [[0]*6] * 6
#reward matrix
# A B C D E F
# B
# C
# D
# E
# F
R = [
    [ None, None, None, None,   0,  None ],
    [ None, None, None,    0, None, 100  ],
    [ None, None, None,    0, None, None ],
    [ None,     0,   0, None,    0, None ],
    [ 0,    None, None,    0, None, 100  ],
    [ None, 0,    None, None,    0, 100  ],
]
R = [
    [ -1, -1, -1, -1,  0,  -1 ],
    [ -1, -1, -1,  0, -1, 100 ],
    [ -1, -1, -1,  0, -1,  -1 ],
    [ -1,  0,  0, -1,  0,  -1 ],
    [  0, -1, -1,  0, -1, 100 ],
    [ -1,  0, -1, -1,  0, 100 ],
]

learning_rate = 0.8
n_epochs = 5
import random
for i in range(n_epochs):
    initial_state = random.choice(Q)
    initial_state_index = Q.index(initial_state)
    for i in range(n_epochs):
        action = -1
        while action == -1:
            action_index = random.randrange(0, len(initial_state))
            action = R[initial_state_index][action_index]
        print(action_index, action)
        print(Q, Q[initial_state_index][action_index])
        Q[initial_state_index][action_index] = R[initial_state_index][action_index] + learning_rate * max(Q[action_index])

# higher learning rate allows the aget to look ahead
# learning_rate = gamma ( y ) 
# initial state = B
# Q Reinforcement learning transition formula
# Q(state,action) = R(state, action) + learning_rate.Max[Q(next_state, all_actions)]