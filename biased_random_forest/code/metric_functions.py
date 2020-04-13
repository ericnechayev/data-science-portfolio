import matplotlib.pyplot as plt
import numpy as np
import collections


def calculate_recall(predictions, labels):
    # Count up the True Positives and False Negatives while Verifying Predictions
    tp, fn = 0, 0
    for i in range(len(predictions)):
        if labels[i] == predictions[i] == 1:
            tp += 1
        if predictions[i] == 0 and labels[i] != predictions[i]:
            fn += 1
    return (tp/(tp+fn))


def calculate_precision(predictions, labels):
    # Count up the True Positives and False Positives while Verifying Predictions
    tp, fp = 0, 0
    for i in range(len(predictions)):
        if labels[i] == predictions[i] == 1:
            tp += 1
        if predictions[i] == 1 and labels[i] != predictions[i]:
            fp += 1
    return (tp/(tp+fp))


# https://www.daniweb.com/programming/computer-science/tutorials/520084/understanding-roc-curves-from-scratch
def calc_ConfusionMatrix(actuals, scores, threshold=0.5, positive_label=1.0):

    ConfusionMatrix = collections.namedtuple('conf', ['tp', 'fp', 'tn', 'fn'])

    tp=fp=tn=fn=0
    bool_actuals = [act == positive_label for act in actuals]

    for truth, score in zip(bool_actuals, scores):
        if score > threshold:                      # predicted positive
            if truth:                              # actually positive
                tp += 1
            else:                                  # actually negative
                fp += 1
        else:                                      # predicted negative
            if not truth:                          # actually negative
                tn += 1
            else:                                  # actually positive
                fn += 1
    return ConfusionMatrix(tp, fp, tn, fn)


# https://www.daniweb.com/programming/computer-science/tutorials/520084/understanding-roc-curves-from-scratch
def get_auroc_or_auprc(actuals, scores):

    # generate thresholds over score domain
    low = min(scores)
    high = max(scores)
    step = (abs(low) + abs(high)) / 1000
    thresholds = np.arange(low-step, high+step, step)

    # calculate confusion matrices for all thresholds
    confusionMatrices = []
    for threshold in thresholds:
        confusionMatrices.append(calc_ConfusionMatrix(actuals, scores, threshold))

    # metric functions
    def FPR(conf_mtrx):
        return conf_mtrx.fp / (conf_mtrx.fp + conf_mtrx.tn) if (conf_mtrx.fp + conf_mtrx.tn) != 0 else 0

    def TPR(conf_mtrx):
        return conf_mtrx.tp / (conf_mtrx.tp + conf_mtrx.fn) if (conf_mtrx.tp + conf_mtrx.fn) != 0 else 0

    def PRECISION(conf_mtrx):
        return conf_mtrx.tp / (conf_mtrx.tp + conf_mtrx.fp) if (conf_mtrx.tp + conf_mtrx.fp) != 0 else 0

    def RECALL(conf_mtrx):
        return conf_mtrx.tp / (conf_mtrx.tp + conf_mtrx.fn) if (conf_mtrx.tp + conf_mtrx.fn) != 0 else 0

    # apply functions to all confusion matrices
    results = {}
    results["FPR"] = list(map(FPR, confusionMatrices))
    results["TPR"] = list(map(TPR, confusionMatrices))
    results["RECALL"] = list(map(RECALL, confusionMatrices))
    results["PRECISION"] = list(map(PRECISION, confusionMatrices))

    return results


def calculate_auroc(labels, probabilities):
    # Get Metrics from Confusion Matrices using Probabilities
    roc_dict = get_auroc_or_auprc(labels, probabilities)
    fpr, tpr = roc_dict['FPR'], roc_dict['TPR']
    # Use the Trapezoidal Area Summation Method to Compute Area Under the Sampled Curve
    return np.abs(np.trapz(y=tpr, x=fpr))


def calculate_auprc(labels, probabilities):
    # Get Metrics from Confusion Matrices using Probabilities
    metric_lists = get_auroc_or_auprc(labels, probabilities)
    recall, precision = metric_lists['RECALL'], metric_lists['PRECISION']
    # Use the Trapezoidal Area Summation Method to Compute Area Under the Sampled Curve
    return np.abs(np.trapz(y=precision, x=recall))


def create_roc_curve(labels, probabilities, filename):
    # Get Metrics from Confusion Matrices using Probabilities
    metric_lists = get_auroc_or_auprc(labels, probabilities)
    fpr, tpr = metric_lists['FPR'], metric_lists['TPR']

    # plot roc curve
    plt.plot(fpr, tpr)
    plt.title("ROC Curve")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")

    # save curve as file
    plt.savefig(fname='../plot_images/'+filename+'.png')
    plt.clf()

    return


def create_prc_curve(labels, probabilities, filename):
    # Get Metrics from Confusion Matrices using Probabilities
    metric_lists = get_auroc_or_auprc(labels, probabilities)
    recall, precision = metric_lists['RECALL'], metric_lists['PRECISION']

    # plot prc curve
    plt.plot(recall, precision)
    plt.title("Precision-Recall Curve")
    plt.xlabel("Recall")
    plt.ylabel("Precision")

    # save curve as file
    plt.savefig(fname='../plot_images/'+filename+'.png')
    plt.clf()

    return

