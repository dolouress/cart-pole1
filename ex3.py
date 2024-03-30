import numpy as np
import gym
import matplotlib.pyplot as plt
import warnings

# Suppressing runtime warnings
#warnings.filterwarnings("ignore", category=RuntimeWarning)

# Importing functions from pade.py
from pade import pade, create_q_labels, enumerate_q_labels

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

# Generate random samples of the cart pole function
samples = 20
data = np.random.uniform(-1, 1, (samples, 2))
target = np.random.uniform(-1, 1, samples)  # Random target values for demonstration

# Compute Pade values
q_table = pade(data, target)

# Translate Pade values to the readable Q-labels
q_labels = create_q_labels(q_table, ['x', 'y'])

# Print Pade results
for (point, q_label) in zip(data, q_labels):
    print(point, q_label)

# Enumerate Q-labels so they can be used with scikit-learn
classes, class_names = enumerate_q_labels(q_labels)

# Build a qualitative model in the form of a decision tree (for demonstration purpose)
from sklearn import tree

classifier = tree.DecisionTreeClassifier(min_impurity_decrease=0.1)
model = classifier.fit(data, classes)

# Visualize the learned model (for demonstration purpose)
tree.plot_tree(model, feature_names=['x', 'y'], class_names=class_names, filled=True)
plt.show()
