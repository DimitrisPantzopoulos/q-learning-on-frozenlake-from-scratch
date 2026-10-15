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

# Step 10 - q_learning_update (not yet solved)
# TODO: implement

# Step 11 - interaction_step (not yet solved)
# TODO: implement

# Step 12 - run_training_episode (not yet solved)
# TODO: implement

# Step 13 - train_q_learning (not yet solved)
# TODO: implement

# Step 14 - extract_greedy_policy (not yet solved)
# TODO: implement

# Step 15 - run_greedy_episode (not yet solved)
# TODO: implement

# Step 16 - evaluate_success_rate (not yet solved)
# TODO: implement

