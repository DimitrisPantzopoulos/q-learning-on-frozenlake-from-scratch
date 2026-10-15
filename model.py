"""
Q-Learning on FrozenLake from Scratch

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - init_q_table
import numpy as np

def init_q_table(num_states : int, num_actions : int) -> np.ndarray:
    """Return a zero-initialized Q-table of shape (num_states, num_actions)."""
    # 2D float64 numpy array of zeros sized by states and actions.
    
    return np.zeros((num_states, num_actions), dtype=np.float64)

# Step 2 - max_q_value
import numpy as np

def max_q_value(q_table : np.ndarray, state : int) -> np.float64:
    """Return the maximum Q value across all actions for the given state."""

    return np.max(q_table[state, :])

# Step 3 - greedy_action
import numpy as np

def greedy_action(q_table : np.ndarray, state : int) -> int:
    """Return the action index with the highest Q value at the given state."""
    return int(np.argmax(q_table[state, :]))

# Step 4 - sample_random_action
from gymnasium.spaces import Discrete

def sample_random_action(action_space : Discrete) -> int:
    """Draw a uniformly random action from the given Gymnasium action space"""
    return int(action_space.sample())

# Step 5 - should_explore
def should_explore(epsilon : float, rng : np.random.Generator) -> bool:
    """Return True with probability epsilon using the provided numpy Generator."""
    
    if rng.random() < epsilon:
        return True
    return False

# Step 6 - epsilon_greedy_action
from gymnasium.spaces import Discrete

import numpy as np

def multiple_maximum_total_reward(
        state_vector   : np.ndarray, 
        maximum_reward : np.int64,
    ) -> bool:

    return np.sum(maximum_reward == state_vector) > 1

def epsilon_greedy_action(
        q_table      : np.ndarray, 
        state        : int, 
        epsilon      : float, 
        action_space : Discrete, 
        rng          : np.random.Generator
    ) -> int:
    """Return an epsilon-greedy action for the given state."""

    # Should the agent explore this step?
    does_explore : bool = should_explore(epsilon, rng)

    # Exploration
    if does_explore:
        return sample_random_action(action_space)
    
    # Exploitation
    state_vector : np.ndarray = q_table[state, :]

    # Check if multiple actions share the same maximum total area.
    maximum_reward : np.int64 = np.max(state_vector)

    multiple_max : bool = multiple_maximum_total_reward(
        state_vector,
        maximum_reward
    )

    if multiple_max:
        return int(rng.choice(
            np.flatnonzero(maximum_reward == state_vector).tolist()
        ))

    return int(np.argmax(state_vector))

# Step 7 - decay_epsilon
def decay_epsilon(
        epsilon     : float, 
        decay_rate  : float, 
        min_epsilon : float
    ):
    # Decay epsilon using max(min_epsilon, epsilon * decay_rate)
    """
    Decay epsilon.
    epsilon_t+1 = max(min_epsilon, epsilon_t * decay_rate).
    """

    return max(min_epsilon, epsilon * decay_rate)

# Step 8 - td_target
def td_target(
        reward     : float, 
        gamma      : float, 
        q_table    : np.ndarray, 
        next_state : int, 
        done       : bool
    ) -> float:
    """
    Compute the bootstrap target used to update a state-action value.
    """

    future_return : np.float64 = (
        gamma * max_q_value(q_table, next_state) if not done else 0.0
    )

    return reward + future_return

# Step 9 - td_error
def td_error(
        target  : float, 
        q_table : np.ndarray, 
        state   : int, 
        action  : int
    ) -> float:
    """
    Return the Temporal Difference Error.

    delta = target - Q(s', a')

    where:
        target = reward + gamma * Q(s', a')
    """
    return target - q_table[state, action]

# Step 10 - q_learning_update
import numpy as np

def q_learning_update(
        q_table     : np.ndarray, 
        state       : int, 
        action      : int, 
        reward      : float, 
        next_state  : int, 
        done        : bool, 
        alpha       : float, 
        gamma       : float
    ) -> float:

    """
    Compute the new updated value of Q(s, a).

    Q(s, a)_t+1 = Q(s, a)_t + alpha * TD

    where:
        TD = y - Q(s, a), The Temporal Distance, the signal used to train update the Q Table.
    """
    
    temporal_difference : float = td_target(
        reward, gamma, q_table, next_state, done
    )

    # Compute the gradient descent step
    q_table[state, action] = q_table[state, action] + (
        alpha * (temporal_difference - q_table[state, action])
    )

    return q_table[state, action]

# Step 11 - interaction_step
from typing import Tuple

import gymnasium as gym

def interaction_step(
        env     : gym.Env, 
        q_table : np.ndarray, 
        state   : int, 
        epsilon : float, 
        alpha   : float, 
        gamma   : float, 
        rng     : np.random.Generator
    ) -> Tuple[int, float, bool]:
    """
    Select Epsilon-greed action, step env and 
    then apply Q-learning update.

    return (next_state, reward, done)
    """
    
    # Pick Epsilon Greedy action
    action : int = epsilon_greedy_action(
        q_table, state, epsilon, env.action_space, rng 
    )

    # Make the step in the env    
    observation, reward, terminated, truncated, _ = env.step(action)

    done : bool = terminated or truncated

    # Update the Q-Learning Table
    q_learning_update(
        q_table, state, action, reward, observation, done, alpha, gamma
    )

    return int(observation), float(reward), bool(done)

# Step 12 - run_training_episode
def run_training_episode(
        env       : gym.Env, 
        q_table   : np.ndarray, 
        epsilon   : float, 
        alpha     : float, 
        gamma     : float, 
        rng       : np.random.Generator, 
        max_steps : int = 200
    ) -> float:
    """
    Simulate a training episode,
    Repeatedly call interaction_step until done or 
    time limit has been reached.
    """
    observation, _ = env.reset()

    # Training Progress
    total_reward : float = 0.0

    # Terminating Condition
    done          : bool = False
    current_steps : int = 0

    while not done and max_steps >= current_steps:
        observation, reward, done = interaction_step(
            env, q_table, observation, epsilon, alpha, gamma, rng
        )

        total_reward  += reward
        current_steps += 1
    return total_reward

# Step 13 - train_q_learning
import numpy as np

def train_q_learning(
        env           : gym.Env, 
        num_episodes  : int, 
        alpha         : float = 0.8, 
        gamma         : float = 0.95, 
        epsilon_start : float = 1.0, 
        epsilon_min   : float = 0.01, 
        epsilon_decay : float = 0.99, 
        seed          : int   = 0, 
        max_steps     : int   = 200
    ) -> (np.ndarray, list[float]):
    
    # Seeding
    rng : np.random.Generator = np.random.default_rng(seed)

    env.action_space.seed(seed)

    # Training Script
    q_table : np.ndarray = init_q_table(env.observation_space.n, env.action_space.n)
    
    # Hyper Parameters
    epsilon : float = epsilon_start

    episode_returns : list[float] = []

    for epoch in range(num_episodes):
        env.reset(seed=seed)
        
        reward : float = run_training_episode(
            env, q_table, epsilon, alpha, gamma, rng, max_steps
        )

        epsilon = decay_epsilon(epsilon, epsilon_decay, epsilon_min)

        episode_returns.append(reward)

    return q_table, episode_returns

# Step 14 - extract_greedy_policy
import numpy as np

def extract_greedy_policy(q_table : np.ndarray) -> np.ndarray:
    return np.argmax(q_table, axis=1)

# Step 15 - run_greedy_episode
import gymnasium as gym
import numpy as np

def run_greedy_episode(
        env       : gym.Env, 
        policy    : np.ndarray, 
        seed      : int | None = None, 
        max_steps : int        = 200
    ) -> bool:
    """Run one greedy episode and return True if the goal was reached."""
    successful : bool = False
    
    observation, _ = env.reset(seed=seed)

    for greedy_index in policy:
        observation, reward, terminated, truncated, _ = env.step(greedy_index)

        if terminated or truncated or greedy_index >= max_steps:
            successful = True
            break

    return successful

# Step 16 - evaluate_success_rate
import gymnasium as gym
import numpy as np

def evaluate_success_rate(
        env          : gym.Env, 
        policy       : np.ndarray, 
        num_episodes : int, 
        seed         : int = 0, 
        max_steps    : int = 200
    ) -> float:

    total_runs     : int = num_episodes
    completed_runs : int = 0

    for i in range(num_episodes):
        successful : bool = run_greedy_episode(
            env, policy, seed + i, max_steps
        )

        if successful:
            completed_runs += 1

    return completed_runs / total_runs

