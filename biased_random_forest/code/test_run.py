import random
import time

from prediction_functions import cross_validation, predict_test_set
from preprocessing import preprocess_and_split


# Setting our Data Path
filepath = "C:/Users/ericn/Desktop/assignment_eric_nechayev/data/diabetes.csv"


# Importing, Pre-processing and Splitting our Dataset
test_size = 0.2
random.seed(42)
train_df, test_df = preprocess_and_split(filepath=filepath, test_size=test_size)
start = time.time()


# Initializing Default Parameters
subsample = 0.8
feature_fraction = 0.8
max_depth = 3
K, p, s = 10, 0.5, 100
plots = 'no'


# Regular Random Forest Baseline
print('Regular Random Forest Baseline:')
predict_test_set(train_df, test_df, model_type='RF', s=s, subsample=subsample,
                 feature_fraction=feature_fraction, max_depth=max_depth, plots=plots)


# Using 10-Fold Cross Validation to Tune Parameters
max_auroc = 0
for K in [10, 5, 3, 2, 1]:
    for p in [0.5, 0.3, 0.7, 0.4, 0.6]:
        for s in [100, 50, 150, 200]:
            # Score the Current Parameter Combo Based on Mean AUROC of Folds
            print("Biased Random Forest: K={}, p={}, s={}:".format(K,p,s))
            current_mean_auroc = cross_validation(train_df, model_type='BRAF', K=K, p=p, s=s, subsample=subsample,
                                                  feature_fraction=feature_fraction, max_depth=max_depth, plots=plots)
            # If Better, Update the Best Parameters
            if current_mean_auroc > max_auroc:
                best_k, best_p, best_s = K, p, s


# Best Parameters
K, p, s = best_k, best_p, best_s
plots = 'yes'


# # Cross-Validation with Optimal Parameters
cross_validation(train_df, model_type='BRAF', K=K, p=p, s=s, subsample=subsample,
                 feature_fraction=feature_fraction, max_depth=max_depth, plots=plots)


# Predicting the Test Set with Optimal Parameters
predict_test_set(train_df, test_df, model_type='BRAF', K=K, p=p, s=s, subsample=subsample,
                 feature_fraction=feature_fraction, max_depth=max_depth, plots=plots)

print("--- %s seconds ---" % (time.time() - start))