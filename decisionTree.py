import numpy as np

np.random.seed(42)

chi_table = {0.01: 6.635,
             0.005: 7.879,
             0.001: 10.828,
             0.0005: 12.116,
             0.0001: 15.140,
             0.00001: 19.511,
             1: 0}

PRINT_TEMPLATE = """\
{indentation}[X{feature} <= {threshold}],
{left}
{right}"""


def counts_classes(data):
    label_to_amount_of_instances = {}  # creating a dictionry of type {label -> amount_of_instance_with_label}
    for instance in data:
        label = instance[-1]
        if label not in label_to_amount_of_instances:
            label_to_amount_of_instances[label] = 0
        label_to_amount_of_instances[label] += 1
    return label_to_amount_of_instances


def calc_gini(data):
    """
    Calculate gini impurity measure of a dataset.

    Input:
    - data: any dataset where the last column holds the labels.

    Returns the gini impurity of the dataset.
    """

    label_to_amount_of_instances = counts_classes(data)

    # Calculating gini impurity
    gini = 1
    for label in label_to_amount_of_instances:
        amount_of_instances = label_to_amount_of_instances[label]
        label_probability = amount_of_instances / float(len(data))
        gini -= label_probability ** 2

    return gini


def calc_entropy(data):
    """
    Calculate the entropy of a dataset.

    Input:
    - data: any dataset where the last column holds the labels.

    Returns the entropy of the dataset.
    """
    entropy = 0.0

    label_to_amount_of_instances = counts_classes(data)

    # Calculating entropy impurity

    for label in label_to_amount_of_instances:
        amount_of_instances = label_to_amount_of_instances[label]
        p = amount_of_instances / float(len(data))  # maybe need to divide with some thnig else? x[0]?
        entropy -= p * np.log2(p)

    return entropy


def info_gain(father_node_impurity, left_branch, right_branch, impurity):
    weighted_average = float(len(left_branch)) / (len(left_branch) + len(right_branch))  # TODO:not true
    info_gain = father_node_impurity - weighted_average * impurity(left_branch) - (
            (1 - weighted_average) * impurity(right_branch))
    return info_gain


class DecisionNode:

    # This class will hold everything you require to construct a decision tree.
    # The structure of this class is up to you. However, you need to support basic
    # functionality as described in the notebook. It is highly recommended that you
    # first read and understand the entire exercise before diving into this class.

    def __init__(self, feature=None, threshold=None, data=None, parent=None):
        self.feature = feature  # column index of criteria being tested
        self.threshold = threshold  # value necessary to get a true result
        self.parent = parent
        if data is not None:
            self.predictions = counts_classes(data)

        self.left_branch = None  # left child's data, will be set post construction
        self.right_branch = None  # right child's data, will be set post construction

    def set_children(self, left_branch, right_branch):
        if left_branch is not None:
            left_branch.parent = self
        self.left_branch = left_branch

        if right_branch is not None:
            right_branch.parent = self
        self.right_branch = right_branch

        self.predictions = None

    def convert_to_leaf(self, data):
        left_branch = self.left_branch
        right_branch = self.right_branch

        self.left_branch.parent = None
        self.right_branch.parent = None
        self.set_children(None, None)

        self.predictions = counts_classes(data)

        return left_branch, right_branch

    def is_leaf(self):
        return self.left_branch is None and self.right_branch is None and self.predictions is not None

    def to_string(self, indentation=""):
        if self.is_leaf():
            return "{indentation}leaf: [{predictions}]".format(
                indentation=indentation, predictions=self.predictions)

        child_indentation = indentation + "  "
        return PRINT_TEMPLATE.format(
            indentation=indentation, feature=self.feature, threshold=self.threshold,
            left=self.left_branch.to_string(child_indentation),
            right=self.right_branch.to_string(child_indentation))

def print_tree(node):
    print(node.to_string())

    
def thresholds_avg(feature):
    threshold_avg = []
    thresholds = np.sort(list(set(feature)))
    for i in range(len(thresholds) - 1):
        threshold_avg.append(float(thresholds[i] + thresholds[i + 1]) / 2)
    return threshold_avg


def single_data_split(feature, threshold, data):
    """
   Splits the data based on a threshold and an attribute.

    Input:
    - feature: the attribute on which to split.
    - threshold: the value on which the function will make the split.
    - data: the training dataset.

    Output: the two lists after split.
    """
    left_data = []
    right_data = []
    for instance in range(len(data)):
        if data[instance][feature] >= threshold: 
            right_data.append(data[instance])
        else:
            left_data.append(data[instance])
    return np.array(left_data), np.array(right_data)


def find_best_split(data, impurity):
    max_impurity_gain = 0
    feature_to_split = 0
    threshold_to_split = 0
    father_impurity = impurity(data)
    num_of_features = len(data[0]) - 1

    for feature in range(num_of_features):
        threshold_avg = thresholds_avg(data[:, feature])

        for threshold in threshold_avg:
            left_branch, right_branch = single_data_split(feature, threshold, data)
            if len(left_branch) == 0 or len(right_branch) == 0:
                continue  # Don't split if no division of the data will be made.

            impurity_gain = info_gain(father_impurity, left_branch, right_branch, impurity)
            if impurity_gain > max_impurity_gain:
                max_impurity_gain = impurity_gain
                feature_to_split = feature
                threshold_to_split = threshold

    return max_impurity_gain, feature_to_split, threshold_to_split


def build_tree(data, impurity, chi_value=1, parent=None):
    """
    Build a tree using the given impurity measure and training dataset.
    You are required to fully grow the tree until all leaves are pure.

    Input:
    - data: the training dataset.
    - impurity: the chosen impurity measure. Notice that you can send a function
                as an argument in python.

    Output: the root node of the tree.
    """

    impurity_gain, feature_to_split, threshold_to_split = find_best_split(data, impurity)
    if impurity_gain == 0:
        return DecisionNode(data=data, parent=parent)  # If there isn't any impurity gain, create a leaf.

    left_instances, right_instances = single_data_split(feature_to_split, threshold_to_split, data)
    chi_sq = calc_Chi_Square(data, left_instances, right_instances)
    if chi_sq < chi_table[chi_value]:
        return DecisionNode(data=data, parent=parent)  # If chi-sq is smaller than the dictionary one, make it a leaf.

    tree = DecisionNode(feature=feature_to_split, threshold=threshold_to_split, parent=parent)
    tree.set_children(build_tree(left_instances, impurity, chi_value, tree),
                      build_tree(right_instances, impurity, chi_value, tree))
    return tree


def predict(node, instance):
    if node.is_leaf():
        prediction = node.predictions
        return next(iter(prediction))
    if instance[node.feature] >= node.threshold:
        return predict(node.right_branch, instance)
    return predict(node.left_branch, instance)


def calc_accuracy(node, data):
    """
    calculate the accuracy starting from some node of the decision tree using
    the given dataset.
    Input:
    - node: a node in the decision tree.
    - dataset: the dataset on which the accuracy is evaluated
    Output: the accuracy of the decision tree on the given dataset (%).
    """
    true_prediction = 0.0
    false_prediction = 0.0
    for instance in range(len(data)):
        if predict(node, data[instance]) == data[instance][-1]:
            true_prediction += 1
        else:
            false_prediction += 1
    return (float(true_prediction) / (true_prediction + false_prediction)) * 100.0  # len(data))


def calc_Chi_Square(data, left_instances, right_instances):
    data = np.array(data)
    py_one = np.sum(data[:, -1])
    py_zero = data.shape[0] - py_one
    py_one = (py_one / data.shape[0])
    py_zero = (py_zero / data.shape[0])

    l_data = np.array(left_instances)
    pf = l_data[l_data[:, -1] == 0].shape[0]
    nf = l_data[l_data[:, -1] == 1].shape[0]
    e0 = l_data.shape[0] * py_zero
    e1 = l_data.shape[0] * py_one
    acc = ((((pf - e0) ** 2) / e0) + (((nf - e1) ** 2) / e1))

    r_data = np.array(right_instances)
    pf = r_data[r_data[:, -1] == 0].shape[0]
    nf = r_data[r_data[:, -1] == 1].shape[0]
    e0 = r_data.shape[0] * py_zero
    e1 = r_data.shape[0] * py_one
    acc += ((((pf - e0) ** 2) / e0) + (((nf - e1) ** 2) / e1))
    return acc


def rec_post_pruning(root, node, data):
    if node.is_leaf():
        parent = node.parent  # Parent is guaranteed due to the calling function loop condition.
        left_branch, right_branch = parent.convert_to_leaf(data)
        accuracy = calc_accuracy(root, data)
        parent.set_children(left_branch, right_branch)
        return 0, accuracy, parent

    left_nodes_counter, left_max_accuracy, left_parent = rec_post_pruning(root, node.left_branch, data)
    right_nodes_counter, right_max_accuracy, right_parent = rec_post_pruning(root, node.right_branch, data)

    internal_nodes_counter = left_nodes_counter + right_nodes_counter + 1
    max_accuracy = max(left_max_accuracy, right_max_accuracy)
    node_to_convert_to_leaf = left_parent if max_accuracy == left_max_accuracy else right_parent

    return internal_nodes_counter, max_accuracy, node_to_convert_to_leaf


def post_pruning(root, data):
    return rec_post_pruning(root, root, data)


# Receiving a tree root and cut every time a different parent until we left with only the root.
def iterating_tree(root, data):
    arr = []
    while not root.is_leaf():
        internal_nodes_counter, max_accuracy, node = post_pruning(root, data)
        arr.append((internal_nodes_counter, max_accuracy))  # A tuple of (internal nodes, accuracy) for the graph
        node.convert_to_leaf(data)
    return arr
