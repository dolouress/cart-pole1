import random
import numpy as np
from example_cart import CartPole
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.tree import export_text
import pade

errors = []

for itr in range(1):
    # print(itr)

    dt = 0.02  # 50 Hz
    samples = 900

    cart = CartPole()
    cart.reset()

    plot_x = np.zeros(samples)
    plot_y = np.zeros(samples)
    plot_z = np.zeros(samples)

    data = np.zeros((samples, 3))
    target_theta = np.zeros(samples)
    target_dtheta = np.zeros(samples)

    for i in range(samples):
        force = random.random() * 200.0 - 100.0

        duration = random.random() * 0.5

        cart.tag_state()

        first = True
        t = duration
        while t > 0:
            cart.step(force)
            t -= dt

            if first:
                (x0, dx0, theta0, dtheta0) = cart.get_tagged_state()
                (x, dx, theta, dtheta) = cart.state
                (delta_x, delta_dx, delta_theta, delta_dtheta) = cart.get_tag_diff()

                plot_x[i] = np.degrees(theta)  # Convert to degrees
                plot_y[i] = np.degrees(dtheta)  # Convert to degrees
                plot_z[i] = force

                data[i, 0] = theta0
                data[i, 1] = dtheta0
                data[i, 2] = force
                target_theta[i] = delta_theta
                target_dtheta[i] = delta_dtheta

                first = False

    # print(plot_x)

    scatter = plt.scatter(plot_x, plot_y, c=plot_z, cmap='viridis', vmin=-100, vmax=100)
    plt.xticks([-180, -90, 0, 90, 180])
    plt.yticks([-360, -180, 0, 180, 360])

    plt.colorbar(ticks=[-100, -50, 0, 50, 100])
    plt.show()

    q_table = pade.pade(data, target_dtheta, nNeighbours=10)
    q_labels = pade.create_q_labels(q_table[:, 2:3], ['force'])

    classes, class_names = pade.enumerate_q_labels(q_labels)

    classifier = tree.DecisionTreeClassifier(min_impurity_decrease=0.05)
    model = classifier.fit(data[:, 0:2], classes)

    tree_rules = export_text(model, feature_names=['theta', 'dtheta'])

    # for thr in model.tree_.threshold:
    #    if abs(thr) > 50:
    #        errors.append(abs(abs(thr) - 90))

    # print("theta dtheta class")
    # for ([x, y], q) in zip(data[:,0:2], q_labels):
    #    print(x, y, q)

    # Convert threshold values from radians to degrees
    for i, threshold in enumerate(model.tree_.threshold):
        if threshold != tree._tree.TREE_UNDEFINED:
            model.tree_.threshold[i] = np.degrees(threshold)

    tree.plot_tree(model, feature_names=['theta', 'dtheta'], class_names=class_names, filled=True)
    plt.show()

    # print(errors)
