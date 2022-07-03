from distutils.core import run_setup
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from tqdm import tqdm #This pkg is for process bar

# world height
WORLD_HEIGHT = 4

# world width
WORLD_WIDTH = 12

# probability for exploration
EPSILON = 0.1

# step size
ALPHA = 0.5

# discount factor
GAMMA = 0.9

# all possible actions
ACTION_UP = 0
ACTION_DOWN = 1
ACTION_LEFT = 2
ACTION_RIGHT = 3
ACTIONS = [ACTION_UP, ACTION_DOWN, ACTION_LEFT, ACTION_RIGHT]

# initial state action pair values
START = [WORLD_HEIGHT-1, 0]
GOAL = [WORLD_HEIGHT-1, WORLD_WIDTH-1]

def step(state, action):
    i, j = state
    if action == ACTION_UP:
        next_state = [max(i - 1, 0), j]
    elif action == ACTION_LEFT:
        next_state = [i, max(j - 1, 0)]
    elif action == ACTION_RIGHT:
        next_state = [i, min(j + 1, WORLD_WIDTH - 1)]
    elif action == ACTION_DOWN:
        next_state = [min(i + 1, WORLD_HEIGHT - 1), j]
    else:
        assert False

    reward = -1
    if (action == ACTION_DOWN and i == 2 and 1 <= j <= WORLD_WIDTH - 2) or (
        action == ACTION_RIGHT and state == START):
        reward = -100
        next_state = START

    return next_state, reward


# choose an action based on epsilon greedy algorithm
def choose_action(state, q_value):
    if np.random.binomial(1, EPSILON) == 1:
        return np.random.choice(ACTIONS)
    else:
        values_ = q_value[state[0], state[1], :]
        return np.random.choice([action_ for action_, value_ in enumerate(values_) if value_ == np.max(values_)])

# an episode with Sarsa
# @q_value: values for state action pair, will be updated
# @expected: if True, will use expected Sarsa algorithm
# @step_size: step size for updating
# @return: total rewards within this episode
def sarsa(q_value, expected=False, step_size=ALPHA, step_decrease = False):
    state = START
    action = choose_action(state, q_value)
    rewards = 0.0
    time_step = 1
    while state != GOAL:
        next_state, reward = step(state, action)
        next_action = choose_action(next_state, q_value)
        rewards += reward
        if not expected:
            target = q_value[next_state[0], next_state[1], next_action]
        else:
            # calculate the expected value of new state
            target = 0.0
            q_next = q_value[next_state[0], next_state[1], :]
            best_actions = np.argwhere(q_next == np.max(q_next))
            for action_ in ACTIONS:
                if action_ in best_actions:
                    target += ((1.0 - EPSILON) / len(best_actions) + EPSILON / len(ACTIONS)) * q_value[next_state[0], next_state[1], action_]
                else:
                    target += EPSILON / len(ACTIONS) * q_value[next_state[0], next_state[1], action_]
        target *= GAMMA
        if step_decrease == True:
            step_size = 5/(9 + time_step)
        q_value[state[0], state[1], action] += step_size * (
                reward + target - q_value[state[0], state[1], action])
        state = next_state
        action = next_action
        time_step = time_step + 1
    return rewards, time_step

# an episode with Q-Learning
# @q_value: values for state action pair, will be updated
# @step_size: step size for updating
# @return: total rewards within this episode
def q_learning(q_value, step_size=ALPHA, step_decrease = False):
    state = START
    rewards = 0.0
    time_step = 1
    while state != GOAL:
        action = choose_action(state, q_value)
        next_state, reward = step(state, action)
        rewards += reward
        # Q-Learning update
        if step_decrease == True:
            step_size = 5/(9 + time_step)
        q_value[state[0], state[1], action] += step_size * (
                reward + GAMMA * np.max(q_value[next_state[0], next_state[1], :]) -
                q_value[state[0], state[1], action])
        state = next_state
        time_step = time_step + 1
    return rewards, time_step

# print optimal policy
def print_optimal_policy(q_value):
    optimal_policy = []
    for i in range(0, WORLD_HEIGHT):
        optimal_policy.append([])
        for j in range(0, WORLD_WIDTH):
            if [i, j] == GOAL:
                optimal_policy[-1].append('G')
                continue
            bestAction = np.argmax(q_value[i, j, :])
            if bestAction == ACTION_UP:
                optimal_policy[-1].append('U')
            elif bestAction == ACTION_DOWN:
                optimal_policy[-1].append('D')
            elif bestAction == ACTION_LEFT:
                optimal_policy[-1].append('L')
            elif bestAction == ACTION_RIGHT:
                optimal_policy[-1].append('R')
    for row in optimal_policy:
        print(row)

# Use multiple runs instead of a single run and a sliding window
# With a single run I failed to present a smooth curve
# However the optimal policy converges well with a single run
# Sarsa converges to the safe path, while Q-Learning converges to the optimal path
def main(step_decrease, display_optimal_policy):
    # episodes of each run
    episodes = 500

    # perform  independent runs
    runs = 50

    rewards_sarsa = np.zeros(episodes)
    rewards_q_learning = np.zeros(episodes)
    rewards_exp_sarsa = np.zeros(episodes)
    time_step_sarsa = np.zeros(episodes)
    time_step_exp_sarsa = np.zeros(episodes)
    time_step_q_learning = np.zeros(episodes)
    for r in tqdm(range(runs)):
        q_sarsa = np.zeros((WORLD_HEIGHT, WORLD_WIDTH, 4))
        q_q_learning = np.copy(q_sarsa)
        q_exp_sarsa = np.copy(q_sarsa)
        for i in range(0, episodes):
            temp_result = sarsa(q_sarsa, step_decrease= step_decrease)
            rewards_sarsa[i] += max(temp_result[0],-100)  # cut off the value by -100 to draw the figure more elegantly
            time_step_sarsa[i] += temp_result[1]

            temp_result = sarsa(q_exp_sarsa, expected=True, step_decrease= step_decrease)
            rewards_exp_sarsa[i] += max(temp_result[0],-100)
            time_step_exp_sarsa[i] += temp_result[1]

            temp_result = q_learning(q_q_learning, step_decrease=step_decrease)
            rewards_q_learning[i] += max(temp_result[0],-100)
            time_step_q_learning[i] += temp_result[1]
            
    # averaging over independt runs
    rewards_sarsa /= runs
    rewards_q_learning /= runs
    rewards_exp_sarsa /= runs 
    time_step_sarsa /= runs
    time_step_q_learning /= runs
    time_step_exp_sarsa /=runs 


    # draw reward curves
    print(rewards_sarsa)
    plt.plot(rewards_sarsa, label='Sarsa')
    plt.plot(rewards_exp_sarsa, label='Expected Sarsa')
    plt.plot(rewards_q_learning, label='Q-Learning')
    plt.xlabel('Episodes')
    plt.ylabel('Sum of rewards during episode')
    #plt.ylim([-100, 0])
    plt.legend()
    plt.show()

    #draw step to terminate curves 
    plt.plot(time_step_sarsa, label='Sarsa')
    plt.plot(time_step_exp_sarsa, label='Expected Sarsa')
    plt.plot(time_step_q_learning, label='Q-Learning')
    plt.plot([WORLD_WIDTH+1 for i in range(episodes)], label='Optimal steps')
    plt.xlabel('Episodes')
    plt.ylabel('Time step to reach the goal')
    lim = [time_step_sarsa[0],time_step_exp_sarsa[0],time_step_exp_sarsa[0]]
    plt.ylim([0,min(lim)/8])
    plt.legend()
    plt.show()

    # display optimal policy
    if display_optimal_policy == False:
        return
    if expected == False:
        print('Sarsa Optimal Policy:')
    else:
        print('Expected Sarsa Optimal Policy:')
    print_optimal_policy(q_sarsa)
    print('Q-Learning Optimal Policy:')
    print_optimal_policy(q_q_learning)


if __name__ == '__main__':
    main(step_decrease=False, display_optimal_policy=False)
