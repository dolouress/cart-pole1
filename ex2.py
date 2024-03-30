import numpy as np
import gym
import matplotlib.pyplot as plt
import pade
from sklearn import tree

# Create CartPole environment
env = gym.make('CartPole-v1')

# Generate random samples of the cart pole function
samples = 1000
data = np.array([env.observation_space.sample() for _ in range(samples)])

# Compute Pade values.
# For the cart pole environment, the target can be any value since we are not predicting anything.
# So, we'll just create dummy target values.
target = np.random.rand(samples)

q_table = pade.pade(data, target)

# Translate Pade values to the readable Q-labels.
# Since the cart pole environment has four attributes, we'll label them as ['position', 'velocity', 'angle', 'angular velocity']
attribute_names = ['position', 'velocity', 'angle', 'angular velocity']
q_labels = pade.create_q_labels(q_table, attribute_names)

# Print Pade results.
for (point, q_label) in zip(data, q_labels):
    print(point, q_label)

# Enumerate Q-labels so they can be used with scikit-learn.
classes, class_names = pade.enumerate_q_labels(q_labels)

# Build a qualitative model in the form of a decision tree.
classifier = tree.DecisionTreeClassifier(min_impurity_decrease=0.1)
model = classifier.fit(data, classes)

# Visualize the learned model.
plt.figure(figsize=(12, 6))
tree.plot_tree(model, feature_names=attribute_names, class_names=class_names, filled=True)
plt.show()
