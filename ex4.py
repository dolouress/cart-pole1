import numpy as np
import gym
import matplotlib.pyplot as plt
import warnings

# Suppressing runtime warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

# Importing functions from pade.py
from pade import pade, create_q_labels, enumerate_q_labels

# Create CartPole environment
env = gym.make('CartPole-v1')

# Reset the environment to get initial state
initial_state = env.reset()

# Check the shape of the initial state
print("Shape of initial state:", len(initial_state))

# Assuming the state has only four elements
initial_theta = initial_state[2]  # Pole angle (theta)
initial_omega = initial_state[3]  # Pole angular velocity (omega)

# Apply random force to the pole
# Assuming random force between -100 and 100 Newtons
random_force = np.random.uniform(-100, 100)

# Apply the force to the environment
# Converting the action to integer format (0 or 1)
action = 0 if random_force < 0 else 1
new_state, _, _, _ = env.step(action)

# Pole position and velocity after applying force
new_theta = new_state[2]  # Pole angle (theta)
new_omega = new_state[3]  # Pole angular velocity (omega)

# Collecting samples before and after applying force
data = np.array([[initial_theta, initial_omega], [new_theta, new_omega]])
target = np.random.uniform(-1, 1, 2)  # Random target values for demonstration

# Compute Pade values
q_table = pade(data, target)

# Translate Pade values to the readable Q-labels
q_labels = create_q_labels(q_table, ['theta', 'omega'])

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
tree.plot_tree(model, feature_names=['theta', 'omega'], class_names=class_names, filled=True)
plt.show()
