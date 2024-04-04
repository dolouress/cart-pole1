<<<<<<< HEAD
import numpy as np
import pade
from sklearn import tree
import matplotlib.pyplot as plt
import gym

# Step 1: Generate Data
env = gym.make('CartPole-v1')
samples = 100
data = []
target = []
def basic_policy(obs1):
    #print(obs1)
    if obs1[1] == {}:
        angle = obs1[0][2]
    else:
        angle = obs1[2]
    #print(angle)
    if angle < 0:
        return 0
    else:
        return 1


for _ in range(samples):
    obs_before = env.reset()
    action = basic_policy(obs_before)  # Applying some force to the pole, for example
    obs_after, _, _, _, _ = env.step(action)
    data.append(obs_before[0])  # Extracting only the NumPy array from the tuple
    target.append(obs_after)

# Convert lists to NumPy arrays
data = np.array(data)
target = np.array(target)


#  data before passing it to pade.pade() function
# print("Data shape:", data.shape)
# print("Target shape:", target.shape)
# print("Data array before passing to pade.pade():", data)

# Compute Pade values.
q_table = pade.pade(data, target)

# Translate Pade values to the readable Q-labels.
q_labels = pade.create_q_labels(q_table, ['horizontal_position', 'velocity', 'angle', 'angular_velocity'])

# Print Pade results.
for (point, q_label) in zip(data, q_labels):
    print(point, q_label)


# Enumerate Q-labels so they can be used with scikit-learn.
classes, class_names = pade.enumerate_q_labels(q_labels)

# Build a qualitative model in the form of a decision tree.
classifier = tree.DecisionTreeClassifier(min_samples_split=40, min_impurity_decrease=0.01)  # Adjust min_samples_split and min_impurity_decrease parameters
model = classifier.fit(data, classes)

# Visualize the learned model.
plt.figure(figsize=(12, 8))
tree.plot_tree(model, feature_names=['horizontal_position', 'velocity', 'angle', 'angular_velocity'], class_names=class_names, filled=True)
plt.show()
=======
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


>>>>>>> 7f0729bbbc1a63943882d6a740df2ba71c31763c
