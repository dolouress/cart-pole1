import numpy as np
import math
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.tree import plot_tree


# Define a function to simulate CartPole dynamics
def simulate_cartpole(cart_position, cart_velocity, pole_angle, pole_angular_velocity, action):
    gravity = 9.8
    masscart = 1.0
    masspole = 0.1
    total_mass = (masspole + masscart)
    length = 0.5  # actually half the pole's length
    polemass_length = (masspole * length)
    force_mag = 10.0
    tau = 0.02  # seconds between state updates

    if action == 1:
        force = force_mag
    else:
        force = -force_mag

    costheta = math.cos(pole_angle)
    sintheta = math.sin(pole_angle)

    temp = (force + polemass_length * pole_angular_velocity ** 2 * sintheta) / total_mass
    thetaacc = (gravity * sintheta - costheta * temp) / (length * (4.0 / 3.0 - masspole * costheta ** 2 / total_mass))
    xacc = temp - polemass_length * thetaacc * costheta / total_mass

    # Update the state
    cart_position += tau * cart_velocity
    cart_velocity += tau * xacc
    pole_angle += tau * pole_angular_velocity
    pole_angular_velocity += tau * thetaacc

    return cart_position, cart_velocity, pole_angle, pole_angular_velocity

# Define function to compute tube neighbors
def _get_tube_neighbours(data, ref_idx, tube_dimension, n):
    # Get the data of the reference sample.
    ref = data[ref_idx]

    # Get the tube dimension values for all samples
    tube_values = data[:, tube_dimension]

    # Compute the differences between the reference sample and all samples along the tube dimension
    differences = tube_values - ref[tube_dimension]

    # Compute the distances
    distances = np.sqrt(np.sum(differences ** 2, axis=0))

    # Sort distances and return the indices of n closest neighbours
    nearest_indices = np.argsort(distances)[:n]
    return nearest_indices


# Define function to perform PADe regression
def pade(data, target, nNeighbours=10):
    # Get the data dimension.
    (_, nAttributes) = data.shape

    # Initialize the Q-table with the same dimension as data.
    q_table = np.zeros(data.shape)

    # Make a tube regression along each dimension.
    for dim in range(nAttributes):
        # Process all samples along the current dimension.
        for idx, sample in enumerate(data):
            # Get the sample value.
            x0 = sample[dim]

            # Get the indices of the nearest neighbours within the tube.
            neighbours = _get_tube_neighbours(data, idx, dim, nNeighbours)

            # If not enough neighbours, skip this sample.
            if len(neighbours) < nNeighbours:
                continue

            # Take the target values of the returned neighbours.
            values = np.take(target, neighbours, axis=0)

            # Get the x values of the nearest neighbours.
            neighbours_x = np.take(data, neighbours, axis=0)[:, dim]

            # Compute the distance of the farthest neighbour.
            max_distance = max([abs(x - x0) for x in neighbours_x])

            # Compute the sigma parameter so that the weight of the farthest sample is 0.001.
            if max_distance < 1e-10:
                sg = math.log(.001)
            else:
                sg = math.log(.001) / max_distance ** 2

            # Compute the weighted univariate linear regression.
            Sx = Sy = Sxx = Syy = Sxy = n = 0.0
            for (x, y) in zip(neighbours_x, values):
                w = math.exp(sg * (x - x0) ** 2)
                Sx += w * x
                Sy += w * y
                Sxx += w * x ** 2
                Syy += w * y ** 2
                Sxy += w * x * y
                n += w
            div = n * Sxx - Sx ** 2
            if div != 0:
                b = (Sxy * n - Sx * Sy) / div
            else:
                b = 0

            # Store the sign of the partial derivative to the Q-table.
            q_table[idx][dim] = np.sign(b)

    # Return the Q-table.
    return q_table

# Define function to translate signs to Q-notation
def create_q_labels(q_table, attribute_names):
    labels = []

    for sample in q_table:
        label = 'Q('
        for name, value in zip(attribute_names, sample):
            if value > 0:
                if label != 'Q(':
                    label += ', '
                label += '+{}'.format(name)
            elif value < 0:
                if label != 'Q(':
                    label += ', '
                label += '-{}'.format(name)
        label += ')'
        labels.append(label)

    return np.array(labels)

# Define function to enumerate Q-labels
def enumerate_q_labels(q_labels):
    # The unique class names.
    class_names = np.unique(q_labels)

    # Enumerate classes.
    classes = np.array([np.where(class_names == q_label)[0][0] for q_label in q_labels])

    return classes, class_names

# Generate random samples of the CartPole environment
samples = 100
data = []
target = []

# Define maximum number of steps per episode
max_steps = 200

for i in range(samples):
    cart_position, cart_velocity, pole_angle, pole_angular_velocity = np.random.uniform(-1, 1, size=4)
    done = False
    step = 0
    while not done and step < max_steps:
        data.append([cart_position, cart_velocity, pole_angle, pole_angular_velocity])
        # Choose a random action
        action = np.random.choice([0, 1])
        # Simulate CartPole dynamics
        cart_position, cart_velocity, pole_angle, pole_angular_velocity = simulate_cartpole(cart_position, cart_velocity, pole_angle, pole_angular_velocity, action)
        # Target value based on the angle of the pole (negative for leftward tilt, positive for rightward tilt)
        target.append(-pole_angle)
        # Increment step count
        step += 1
        # Check if the pole has fallen (termination condition)
        done = abs(pole_angle) >= math.pi / 2

    if step == max_steps:
        print(f"Episode {i + 1}: Maximum steps reached without termination.")

# Convert data and target to numpy arrays
data = np.array(data)
target = np.array(target)

# Compute Pade values
q_table = pade(data, target)

# Translate Pade values to the readable Q-labels
q_labels = create_q_labels(q_table, ['Cart Position', 'Cart Velocity', 'Pole Angle', 'Pole Angular Velocity'])

# Enumerate Q-labels so they can be used with scikit-learn's algorithms
classes, class_names = enumerate_q_labels(q_labels)

# Train a decision tree classifier on the Q-labels
clf = tree.DecisionTreeClassifier()
clf = clf.fit(data, classes)

classifier = tree.DecisionTreeClassifier(min_samples_split=20, min_samples_leaf=10, min_impurity_decrease=0.1)

# Build a qualitative model in the form of a decision tree.
#classifier = tree.DecisionTreeClassifier(min_impurity_decrease=0.1)
model = classifier.fit(data, classes)

# Visualize the learned model
plt.figure(figsize=(8, 6))
plot_tree(model, feature_names=['Cart Position', 'Cart Velocity', 'Pole Angle', 'Pole Angular Velocity'],
          class_names=model.classes_, filled=True)
plt.show()

