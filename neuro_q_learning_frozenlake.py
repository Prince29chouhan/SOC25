import numpy as np
import gym
import random

# basic Q-learning params
alpha = 0.1
gamma = 0.99
epsilon = 1.0
min_epsilon = 0.01
decay = 0.005

episodes = 1000
max_steps = 100

# STDP-like parameters
A_plus = 0.01
A_minus = 0.012
tau_plus = 20
tau_minus = 20

# load environment
env = gym.make("FrozenLake-v1", is_slippery=False)
n_states = env.observation_space.n
n_actions = env.action_space.n

# Q-table with random init
q_table = np.random.uniform(low=-1, high=1, size=(n_states, n_actions))

# track last spike times
spike_times = np.full((n_states, n_actions), -np.inf)
time = 0  # global step counter

# STDP update rule
def stdp(state, action, t):
    last = spike_times[state, action]
    dt = t - last

    if dt > 0:
        change = A_plus * np.exp(-dt / tau_plus)
    else:
        change = -A_minus * np.exp(dt / tau_minus)

    q_table[state, action] += change
    spike_times[state, action] = t

# training loop
for ep in range(episodes):
    state = env.reset()[0]
    done = False
    total_reward = 0

    for step in range(max_steps):
        time += 1

        # choose action
        if random.random() < epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(q_table[state])

        # apply STDP rule
        stdp(state, action, time)

        # step in env
        new_state, reward, done, _, _ = env.step(action)
        total_reward += reward

        # update Q-value
        max_next = np.max(q_table[new_state])
        q_table[state, action] += alpha * (reward + gamma * max_next - q_table[state, action])

        state = new_state
        if done:
            break

    # update epsilon
    epsilon = max(min_epsilon, epsilon * np.exp(-decay * ep))

    if ep % 100 == 0:
        print(f"Episode {ep}, reward: {total_reward}")

# print final Q-table
print("\nFinal Q-table:")
print(q_table)
