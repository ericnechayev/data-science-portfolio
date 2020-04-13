import numpy as np
import pandas as pd

from biased_random_forest import random_forest_model, random_forest_predictions, biased_random_forest_model
from metric_functions import calculate_precision, calculate_recall, calculate_auprc, calculate_auroc, create_prc_curve, create_roc_curve


def cross_validation(train_df, model_type='braf', K=10, p=0.5, s=10, subsample=0.67, feature_fraction=1, max_depth=3, plots='no'):
    # Split the Train Set into 10 Folds
    n_folds = 10
    folds = np.array_split(train_df, n_folds)
    aurocs = []
    auprcs = []
    precisions = []
    recalls = []

    # Training on 9 Folds and Testing on the 10th Each Time
    for fold in range(n_folds):
        train = folds.copy() # work on a copy of the array
        val = folds[fold]
        del train[fold]
        train = pd.concat(train)

        train_set = train.copy()
        val_set = val.copy()

        # Fitting the Model
        if model_type == 'RF':
            model = random_forest_model(train_set, n_trees=s, n_bootstrap=int(subsample * len(train_set)),
                                            n_features=int(feature_fraction * train_df.shape[1]), dt_max_depth=max_depth)
        else:
            model = biased_random_forest_model(train_set, K=K, p=p, s=s, n_bootstrap=int(subsample * len(train_set)),
                                                   n_features=int(feature_fraction * train_df.shape[1]), dt_max_depth=max_depth)
        # Predicting the Validation Fold
        predictions, probabilities = random_forest_predictions(val_set, model)

        aurocs.append(calculate_auroc(val_set['label'].values, probabilities.iloc[:, 1].values))
        auprcs.append(calculate_auprc(val_set['label'].values, probabilities.iloc[:, 1].values))
        precisions.append(calculate_precision(predictions.values, val_set.label.values))
        recalls.append(calculate_recall(predictions.values, val_set.label.values))

        # Plotting AUROC and AUPRC
        if plots == 'yes':
            create_roc_curve(val_set['label'].values, probabilities.iloc[:, 1].values, 'cv_fold_{}_roc'.format(str(fold)))
            create_prc_curve(val_set['label'].values, probabilities.iloc[:, 1].values, 'cv_fold_{}_prc'.format(str(fold)))

    # Evaluating the Performance
    print("CV {}".format(model_type)+" Precision = {}".format(np.mean(precisions)))
    print("CV {}".format(model_type)+" Recall = {}".format(np.mean(recalls)))
    print("CV {}".format(model_type)+" AUPRC = {}".format(np.mean(auprcs)))
    print("CV {}".format(model_type)+" AUROC = {}".format(np.mean(aurocs))+'\n')

    # Grading the Chosen Parameter Set
    return np.mean(aurocs)


def predict_test_set(train_df, test_df, model_type='BRAF', K=10, p=0.5, s=10, subsample=1, feature_fraction=1, max_depth=3, plots='no'):
    # Fitting the Model
    if model_type == 'RF':
        model = random_forest_model(train_df, n_trees=s, n_bootstrap=int(subsample * len(train_df)),
                                        n_features=train_df.shape[1], dt_max_depth=max_depth)
    else:
        model = biased_random_forest_model(train_df, K=K, p=p, s=s, n_bootstrap=int(feature_fraction * len(train_df)),
                                               n_features=train_df.shape[1], dt_max_depth=max_depth)

    # Predicting the Test Set
    predictions, probabilities = random_forest_predictions(test_df, model)

    precision = calculate_precision(predictions.values, test_df.label.values)
    recall = calculate_recall(predictions.values, test_df.label.values)
    auprc = calculate_auprc(test_df['label'].values, probabilities.iloc[:, 1].values)
    auroc = calculate_auroc(test_df['label'].values, probabilities.iloc[:, 1].values)

    # Evaluating the Performance
    print("Test {}".format(model_type) + " Precision = {}".format(precision))
    print("Test {}".format(model_type) + " Recall = {}".format(recall))
    print("Test {}".format(model_type) + " AUPRC = {}".format(auprc))
    print("Test {}".format(model_type) + " AUROC = {}".format(auroc)+'\n')

    # Plotting AUROC and AUPRC
    if plots == 'yes':
        create_roc_curve(test_df['label'].values, probabilities.iloc[:, 1].values, 'test_roc')
        create_prc_curve(test_df['label'].values, probabilities.iloc[:, 1].values, 'test_prc')

    return

