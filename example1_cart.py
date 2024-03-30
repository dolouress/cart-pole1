import time

import numpy as np
import gym
import matplotlib.pyplot as plt

# Create CartPole environment
env = gym.make('CartPole-v1')

# Reset the environment to get initial state
initial_state = env.reset()

# Check the shape of the initial state
print("Shape of initial state:", len(initial_state))

# Assuming the state has only two elements, we'll need to adjust our approach
if len(initial_state) == 2:
    # For environments where the state has only two elements
    initial_theta = initial_state[0]  # Assuming the first element represents the pole angle
    initial_omega = 0  # Assuming zero initial angular velocity
else:
    # For environments where the state has four elements
    initial_theta = initial_state[2]  # Pole angle (theta)
    initial_omega = initial_state[3]  # Pole angular velocity (omega)

# Apply random force to the pole
# Assuming random force between -100 and 100 Newtons
random_force = np.random.uniform(-10, 10)

# Apply the force to the environment
# Converting the action to integer format (0 or 1)
action = 0 if random_force < 0 else 1
new_state, _, _, _, _ = env.step(action)

# Wait for 20 milliseconds
time.sleep(0.2)

print("Shape of new state:", len(new_state))
# Pole position and velocity after applying force
if len(new_state) == 2:
    new_theta = new_state[0]  # Assuming the first element represents the pole angle
    new_omega = 0  # Assuming zero initial angular velocity
else:
    new_theta = new_state[2]  # Pole angle (theta)
    new_omega = new_state[3]  # Pole angular velocity (omega)

# Compare pole position before and after applying force
print("Initial Pole Position:", initial_theta[2])
print("New Pole Position:", new_theta)


