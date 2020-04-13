from math import sqrt
import numpy as np
import pandas as pd

from decision_tree import decision_tree_model, tree_predictor


def bootstrapping(train_df, n_bootstrap):
    # Sample a Random Fraction of the Train Set for each Decision Tree
    bootstrap_indices = np.random.randint(low=0, high=len(train_df), size=n_bootstrap)
    # Returning the Bootstrapped Part of the DataFrame
    return train_df.iloc[bootstrap_indices]


def random_forest_model(train_df, n_trees, n_bootstrap, n_features, dt_max_depth):
    forest = []
    # Construct a Forest as a List of Decision Trees
    for i in range(n_trees):
        # Randomly Bootstrap the Samples to use in Each Tree
        df_bootstrapped = bootstrapping(train_df, n_bootstrap)
        # Append each Decision Tree
        tree = decision_tree_model(df_bootstrapped, max_depth=dt_max_depth, random_subspace=n_features)
        forest.append(tree)

    return forest


def euclidean_distance(row1, row2):
    distance = 0.0
    # len(row)-2 to omit the label column at the end
    for i in range(len(row1)-2):
        # Calculate the Distance By All Features
        distance += (row1[i] - row2[i])**2
    return sqrt(distance)


# Locate the most similar neighbors
def get_neighbors(majority_set, row, num_neighbors):
    # find the distance of all in the other class
    distances = list()
    for majority_class_row in majority_set:
        distances.append((majority_class_row, euclidean_distance(row, majority_class_row)))

    # sort neighbors by distance and limit how many
    distances.sort(key=lambda tup: tup[1])

    neighbors = list()
    for i in range(num_neighbors):
        neighbors.append(distances[i][0])
    # get num_neighbors neighbors
    return neighbors


def make_critical_set(train_df, K):
    # distinguishing classes by sets
    critical_set = train_df.loc[train_df['label'] == 1, :].values.tolist()
    majority_set = train_df.loc[train_df['label'] == 0, :].values.tolist()

    # for each minority class observation
    for minority_row in critical_set:
        # add K similar majority observations to the formerly minority-only set
        for neighbor in get_neighbors(majority_set, minority_row, K):
            if neighbor not in critical_set:
                critical_set.append(neighbor)

    # convert from list of lists to df
    return pd.DataFrame(critical_set, columns=train_df.columns)


def biased_random_forest_model(train_df, K, p, s, n_bootstrap, n_features, dt_max_depth):
    # fit a regular random forest
    regular_rf = random_forest_model(train_df, s - int(s * p), n_bootstrap, n_features, dt_max_depth)

    # fit a difficult-areas random forest
    critical_set = make_critical_set(train_df, K)
    critical_rf = random_forest_model(critical_set, int(s * p), n_bootstrap, n_features, dt_max_depth)

    # combine the two forests
    return regular_rf + critical_rf


def random_forest_predictions(test_df, forest):
    predictions = {}

    # Get the Vote of Each Tree of the Forest
    for i in range(len(forest)):
        predictions["tree_{}".format(i)] = test_df.apply(tree_predictor, args=(forest[i],), axis=1)

    # Most Common Vote
    predictions = pd.DataFrame(predictions)
    ensemble_vote = predictions.mode(axis=1)[0]

    # Compute the Probabilities of Each Class in Each Observation via Tree Vote Counts
    predictions['proba_zero'] = predictions.apply(lambda row: 1 - (sum(row) / len(row)), axis=1)
    predictions['proba_one'] = predictions.apply(lambda row: (sum(row) / len(row)), axis=1)

    # Returning Class Predictions (Most Common Vote of Trees by Observation), Class Probabilities
    return ensemble_vote, predictions[['proba_zero', 'proba_one']]
