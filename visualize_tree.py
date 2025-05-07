# visualize_tree.py
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree

def visualize_tree(model, feature_names, class_names):
    one_tree = model.estimators_[0]
    plt.figure(figsize=(20, 10))
    plot_tree(
        one_tree,
        feature_names=feature_names,
        class_names=class_names,
        filled=True,
        rounded=True,
        max_depth=3
    )
    plt.title("Random Forest - One Decision Tree")
    plt.show()
