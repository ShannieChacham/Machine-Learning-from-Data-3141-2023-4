import numpy as np
import matplotlib.pyplot as plt

### Chi square table values ###
# The first key is the degree of freedom 
# The second key is the p-value cut-off
# The values are the chi-statistic that you need to use in the pruning

chi_table = {1: {0.5 : 0.45,
             0.25 : 1.32,
             0.1 : 2.71,
             0.05 : 3.84,
             0.0001 : 100000},
         2: {0.5 : 1.39,
             0.25 : 2.77,
             0.1 : 4.60,
             0.05 : 5.99,
             0.0001 : 100000},
         3: {0.5 : 2.37,
             0.25 : 4.11,
             0.1 : 6.25,
             0.05 : 7.82,
             0.0001 : 100000},
         4: {0.5 : 3.36,
             0.25 : 5.38,
             0.1 : 7.78,
             0.05 : 9.49,
             0.0001 : 100000},
         5: {0.5 : 4.35,
             0.25 : 6.63,
             0.1 : 9.24,
             0.05 : 11.07,
             0.0001 : 100000},
         6: {0.5 : 5.35,
             0.25 : 7.84,
             0.1 : 10.64,
             0.05 : 12.59,
             0.0001 : 100000},
         7: {0.5 : 6.35,
             0.25 : 9.04,
             0.1 : 12.01,
             0.05 : 14.07,
             0.0001 : 100000},
         8: {0.5 : 7.34,
             0.25 : 10.22,
             0.1 : 13.36,
             0.05 : 15.51,
             0.0001 : 100000},
         9: {0.5 : 8.34,
             0.25 : 11.39,
             0.1 : 14.68,
             0.05 : 16.92,
             0.0001 : 100000},
         10: {0.5 : 9.34,
              0.25 : 12.55,
              0.1 : 15.99,
              0.05 : 18.31,
              0.0001 : 100000},
         11: {0.5 : 10.34,
              0.25 : 13.7,
              0.1 : 17.27,
              0.05 : 19.68,
              0.0001 : 100000}}

def calc_gini(data):
    """
    Calculate gini impurity measure of a dataset.
 
    Input:
    - data: any dataset where the last column holds the labels.
 
    Returns:
    - gini: The gini impurity value.
    """
    gini = 0.0
    ###########################################################################
    # TODO: Implement the function.                                           #
    ########################################################################### 
    num_elements= len(data)
    labels, counts = np.unique(data[:, -1], return_counts=True)
    probabilities = counts / num_elements
    gini = 1 - np.sum(probabilities ** 2)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return gini

def calc_entropy(data):
    """
    Calculate the entropy of a dataset.

    Input:
    - data: any dataset where the last column holds the labels.

    Returns:
    - entropy: The entropy value.
    """
    entropy = 0.0
    ###########################################################################
    # TODO: Implement the function.                                           #
    ###########################################################################
    num_elements = len(data)
    labels, counts = np.unique(data[:, -1], return_counts=True)
    probabilities = counts / num_elements
    probabilitiesMultiLog = np.log2(probabilities)
    entropy = (-1)*(np.sum(probabilities * probabilitiesMultiLog))
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return entropy

class DecisionNode:
    def __init__(self, data, impurity_func, feature=-1,depth=0, chi=1, max_depth=1000, gain_ratio=False):
        
        self.data = data # the relevant data for the node
        self.feature = feature # column index of criteria being tested
        self.pred = self.calc_node_pred() # the prediction of the node
        self.depth = depth # the current depth of the node
        self.children = [] # array that holds this nodes children
        self.children_values = []
        self.terminal = False # determines if the node is a leaf
        self.chi = chi 
        self.max_depth = max_depth # the maximum allowed depth of the tree
        self.impurity_func = impurity_func
        self.gain_ratio = gain_ratio
        self.feature_importance = 0
        self.__number_of_samples: int = self.data.shape[0]
    
    def calc_node_pred(self):
        """
        Calculate the node prediction.

        Returns:
        - pred: the prediction of the node
        """
        pred = None
        ###########################################################################
        # TODO: Implement the function.                                           #
        ###########################################################################

        unique_labels, counts = np.unique(self.data[:, -1], return_counts=True)
        # Get the label with the highest count
        pred = unique_labels[np.argmax(counts)]

        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################
        return pred
        
    def add_child(self, node, val):
        """
        Adds a child node to self.children and updates self.children_values

        This function has no return value
        """
        ###########################################################################
        # TODO: Implement the function.                                           #
        ###########################################################################
        self.children.append(node)
        self.children_values.append(val)
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################
        
    def calc_feature_importance(self, n_total_sample):
        """
        Calculate the selected feature importance.
        
        Input:
        - n_total_sample: the number of samples in the dataset.

        This function has no return value - it stores the feature importance in 
        self.feature_importance
        """
        ###########################################################################
        # TODO: Implement the function.                                           #
        ###########################################################################
        if self.feature != -1:
            impurity_before_split = self.impurity_func(self.data)
            feature_values = np.unique(self.data[:, self.feature])
            total_importance = 0
            
            for value in feature_values:
                data_subset = self.data[self.data[:, self.feature] == value]
                weight = len(data_subset) / n_total_sample
                impurity_after_split = self.impurity_func(data_subset)
                impurity_reduction = impurity_before_split - impurity_after_split
                total_importance += weight * impurity_reduction
                
            self.feature_importance = total_importance
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################

    """ From here we have functions for the goodness_of_split""" 

    def goodness_of_split(self, feature):
        """
        Calculate the goodness of split of a dataset given a feature and impurity function.

        Input:
        - feature: the feature index the split is being evaluated according to.

        Returns:
        - goodness: the goodness of split
        - groups: a dictionary holding the data after splitting 
                  according to the feature values.
        """
        goodness = 0
        groups = {} 
        ###########################################################################
        # TODO: Implement the function.                                           #
        ###########################################################################
        
        # Determine the impurity function based on whether gain ratio is used.
        if self.gain_ratio:
            impurity_measure = calc_entropy
        else:
            impurity_measure = self.impurity_func
    
        # Splitting the data into groups based on the feature values.
        unique_values = np.unique(self.data[:, feature])
        groups = {val: self.data[self.data[:, feature] == val] for val in unique_values}
        
        # Calculate total impurity after the split.
        total_weighted_impurity = sum((len(group) / len(self.data)) * impurity_measure(group) 
                                      for group in groups.values())
    
        # Calculate the impurity reduction from the split.
        initial_impurity = impurity_measure(self.data)
        impurity_reduction = initial_impurity - total_weighted_impurity
        goodness = impurity_reduction
    
        # Adjust the goodness measure by the split information if gain ratio is true.
        if self.gain_ratio:
            split_info = self.calculate_split_info(groups)
            if split_info > 0:
                goodness /= split_info
    
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################
        return goodness, groups
        
    
    def calculate_split_info(self, groups):
        """
        Calculate the split information, a measure of how well the feature splits the data.
    
        Input:
        - groups: Dictionary of data subsets split by feature values.
    
        Returns:
        - The split information value.
        """
        total_instances = len(self.data)
        split_info = 0
        for group in groups.values():
            proportion = len(group) / total_instances
            if proportion > 0:
                split_info -= proportion * np.log2(proportion)
        return split_info

    """ From here we have functions for the split""" 

    def check_purity(self):
        """Return True if all outputs in this node are the same."""
        return len(set(self.data[:, -1])) == 1

    def calculate_chi_square_statistic(self, splits):
        """Compute the chi-square statistic for the splits."""
        labels, labels_sizes = np.unique(self.data[:, -1], return_counts=True)
        total_count = self.data.shape[0]
        expected_probabilities = labels_sizes / total_count
        chi_stat = 0
    
        for subset in splits.values():
            subset_labels, subset_sizes = np.unique(subset[:, -1], return_counts=True)
            subset_label_probabilities = dict(zip(subset_labels, subset_sizes))
    
            for label, prob in zip(labels, expected_probabilities):
                expected_count = prob * len(subset)
                observed_count = subset_label_probabilities.get(label, 0)
                chi_stat += ((observed_count - expected_count) ** 2) / expected_count if expected_count != 0 else 0
    
        return chi_stat

    def perform_chi_square_test(self, splits):
        """Check if the split is significant using the chi-square test."""
        degrees_of_freedom = (len(np.unique(self.data[:, -1])) - 1) * (len(splits) - 1)
        chi_stat = self.calculate_chi_square_statistic(splits)
        chi_threshold = chi_table[degrees_of_freedom][self.chi]
        return chi_stat < chi_threshold

    def split(self):
        """
        Splits the current node according to the self.impurity_func. This function finds
        the best feature to split according to and create the corresponding children.
        This function should support pruning according to self.chi and self.max_depth.

        This function has no return value
        """
        ###########################################################################
        # TODO: Implement the function.                                           #
        ###########################################################################
       
        if self.check_purity() or self.depth >= self.max_depth:
            self.terminal = True
            return

        optimal_feature, highest_goodness, optimal_groups = None, float('-inf'), None
        for index in range(self.data.shape[1] - 1):
            current_goodness, current_groups = self.goodness_of_split(index)
            if current_goodness > highest_goodness:
                optimal_feature, highest_goodness, optimal_groups = index, current_goodness, current_groups

        if not optimal_groups or highest_goodness <= 0 or (self.chi != 1 and self.perform_chi_square_test(optimal_groups)):
            self.terminal = True
            return

        self.feature = optimal_feature
        self.calc_feature_importance(self.__number_of_samples)
        for group_key, group_data in optimal_groups.items():
            new_child = DecisionNode(data=group_data, impurity_func=self.impurity_func,
                                         depth=self.depth + 1, chi=self.chi,
                                         max_depth=self.max_depth, gain_ratio=self.gain_ratio)
            self.add_child(new_child, group_key)
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################
    
    def node_max_depth(self):
        if self.terminal is True:
            return self.depth
        
        return max([child.node_max_depth() for child in self.children])


class DecisionTree:
    def __init__(self, data, impurity_func, feature=-1, chi=1, max_depth=1000, gain_ratio=False):
        self.data = data # the relevant data for the tree
        self.impurity_func = impurity_func # the impurity function to be used in the tree
        self.chi = chi 
        self.max_depth = max_depth # the maximum allowed depth of the tree
        self.gain_ratio = gain_ratio #
        self.root = None # the root node of the tree
        
    def build_tree(self):
        """
        Build a tree using the given impurity measure and training dataset. 
        You are required to fully grow the tree until all leaves are pure 
        or the goodness of split is 0.

        This function has no return value
        """
        self.root = None
        ###########################################################################
        # TODO: Implement the function.                                           #
        ###########################################################################
        self.root = DecisionNode(data=self.data, impurity_func=self.impurity_func, max_depth=self.max_depth,
                                 chi=self.chi, gain_ratio=self.gain_ratio)
        queue = [self.root]
        while queue:
            node = queue.pop(0)
            if node.depth >= self.max_depth or node.data.shape[0] <= 1:
                continue

            # Perform split
            node.split()

            for child in node.children:
                # Check if node is pure
                if len(np.unique(child.data[:, -1])) == 1:
                    child.terminal = True
                    continue

                # Check if node's impurity is already 0
                impurity_before_split = child.impurity_func(child.data)
                if impurity_before_split == 0:
                    child.terminal = True
                    continue

                # Check if the goodness of split is 0
                goodness, _ = child.goodness_of_split(child.feature)
                if goodness == 0:
                    child.terminal = True
                    continue

                queue.append(child)       
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################

    def predict(self, instance):
        """
        Predict a given instance
     
        Input:
        - instance: an row vector from the dataset. Note that the last element 
                    of this vector is the label of the instance.
     
        Output: the prediction of the instance.
        """
        pred = None
        ###########################################################################
        # TODO: Implement the function.                                           #
        ###########################################################################
        node = self.root  
        while not node.terminal:  # Continue until a leaf node is reached
            if instance[node.feature] not in node.children_values:
                break  
            index = node.children_values.index(instance[node.feature])  
            node = node.children[index]  
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################
        return node.pred

    def calc_accuracy(self, dataset):
        """
        Predict a given dataset 
     
        Input:
        - dataset: the dataset on which the accuracy is evaluated
     
        Output: the accuracy of the decision tree on the given dataset (%).
        """
        accuracy = 0
        ###########################################################################
        # TODO: Implement the function.                                           #
        ###########################################################################
        correct_predictions = 0
        total_instances = dataset.shape[0]
        
        for instance in dataset:
            prediction = self.predict(instance[:-1])  # Exclude the label from the instance for prediction
            if prediction == instance[-1]:
                correct_predictions += 1
    
        accuracy = (correct_predictions / total_instances) 

        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################
        return accuracy
        
    def depth(self):
        return self.root.node_max_depth()
        
def depth_pruning(X_train, X_validation):
    """
    Calculate the training and validation accuracies for different depths
    using the best impurity function and the gain_ratio flag you got
    previously. On a single plot, draw the training and testing accuracy 
    as a function of the max_depth. 

    Input:
    - X_train: the training data where the last column holds the labels
    - X_validation: the validation data where the last column holds the labels
 
    Output: the training and validation accuracies per max depth
    """
    training = []
    validation  = []
    root = None

    best_impurity_func = calc_entropy  # based on earlier test results
    best_gain_ratio = True  # Set based on earlier test results

    for max_depth in range(1, 11):
        ###########################################################################
        # TODO: Implement the function.                                           #
        ###########################################################################
        
        tree = DecisionTree(data=X_train, impurity_func=best_impurity_func,
                            max_depth=max_depth, gain_ratio=best_gain_ratio)
        tree.build_tree()

        train_acc = tree.calc_accuracy(X_train)
        training.append(train_acc)

        val_acc = tree.calc_accuracy(X_validation)
        validation.append(val_acc)

        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################
        return training, validation

def depth_pruning(X_train, X_validation):
    """
    Calculate the training and validation accuracies for different depths
    using the best impurity function and the gain_ratio flag you got
    previously. On a single plot, draw the training and testing accuracy 
    as a function of the max_depth. 

    Input:
    - X_train: the training data where the last column holds the labels
    - X_validation: the validation data where the last column holds the labels
 
    Output: the training and validation accuracies per max depth
    """
    impurity_func = calc_entropy
    gain_ratio = True

    training = []
    validation  = []
    for max_depth in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
        tree = DecisionTree(data=X_train, impurity_func=impurity_func, feature=-1, max_depth=max_depth, gain_ratio=gain_ratio)
        tree.build_tree()
        training.append(tree.calc_accuracy(X_train))
        validation.append(tree.calc_accuracy(X_validation))
    return training, validation

def chi_pruning(X_train, X_test):

    """
    Calculate the training and validation accuracies for different chi values
    using the best impurity function and the gain_ratio flag you got
    previously. 

    Input:
    - X_train: the training data where the last column holds the labels
    - X_validation: the validation data where the last column holds the labels
 
    Output:
    - chi_training_acc: the training accuracy per chi value
    - chi_validation_acc: the validation accuracy per chi value
    - depth: the tree depth for each chi value
    """
    chi_training_acc = []
    chi_testing_acc = []
    depth = []

    ###########################################################################
    # TODO: Implement the function.                                           #
    ###########################################################################
    chi_values = [1, 0.5, 0.25, 0.1, 0.05, 0.0001]
    for chi_threshold in chi_values:
        decision_tree = DecisionTree(data=X_train, impurity_func=calc_entropy, feature=-1, chi=chi_threshold, gain_ratio=True)
        decision_tree.build_tree()

        train_accuracy = decision_tree.calc_accuracy(X_train)
        chi_training_acc.append(train_accuracy)

        validation_accuracy = decision_tree.calc_accuracy(X_test)
        chi_testing_acc.append(validation_accuracy)

        tree_depth = decision_tree.depth()
        depth.append(tree_depth)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
        
    return chi_training_acc, chi_testing_acc, depth


def count_nodes(node):
    """
    Count the number of node in a given tree
 
    Input:
    - node: a node in the decision tree.
 
    Output: the number of node in the tree.
    """
    ###########################################################################
    # TODO: Implement the function.                                           #
    ###########################################################################
    if node is None:
        return 0
    
    n_nodes= 1 + np.sum(list(map(count_nodes, node.children)))
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return n_nodes






