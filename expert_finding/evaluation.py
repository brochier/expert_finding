import numpy as np
import expert_finding.metrics
import logging
import matplotlib.pyplot as plt
from numpy import interp
fig_dim = (15.0,12.0)

logger = logging.getLogger()


def run(model, A_da, A_dd, T, L_d, L_d_mask, L_a, L_a_mask, tags, path=None):
    eval_batches = run_all_evaluations(model, A_da, A_dd, T, L_d, L_d_mask, L_a, L_a_mask)
    merged_eval = merge_evaluations(eval_batches, tags)
    plot_evaluation(merged_eval, path=path)
    return eval_batches, merged_eval


def get_empty_eval():
    return {
        "metrics": {
            "AP": 0,
            "RR": 0,
            "P@5": 0,
            "P@10": 0,
            "P@50": 0,
            "P@100": 0,
            "P@200": 0,
            "ROC AUC": 0
        },
        "std": {
            "AP": 0,
            "RR": 0,
            "P@5": 0,
            "P@10": 0,
            "P@50": 0,
            "P@100": 0,
            "P@200": 0,
            "ROC AUC": 0
        },
        "curves": {
            "precision": list(),
            "recall": list(),
            "thresholds_pr": list(),
            "fpr": list(),
            "tpr": list(),
            "thresholds_roc": list()
        },
        "info": {

        }
    }

def run_all_evaluations(model, A_da, A_dd, T, L_d, L_d_mask, L_a, L_a_mask):
    eval_batches = list()
    model.fit(A_da, A_dd, T)
    for i, d in enumerate(L_d_mask):
        y_pred = model.predict(d, mask = L_a_mask)
        y_true = np.squeeze(L_d[i].dot(L_a).astype(np.bool).A)
        eval = expert_finding.metrics.get_all_scores(y_true, y_pred)
        eval_batches.append(eval)
    return eval_batches

def merge_evaluations(eval_batches, tags):
    all_eval = get_empty_eval()
    all_count = 0
    std_lists = {
        "AP": [],
        "RR": [],
        "P@5": [],
        "P@10": [],
        "P@50": [],
        "P@100": [],
        "P@200": [],
        "ROC AUC": []
    }
    for eval in eval_batches:
        all_count += 1
        for key, value in eval["metrics"].items():
            std_lists[key].append(value)
        for key, value in eval["curves"].items():
            all_eval["curves"][key].append(value)
    for key, value in all_eval["metrics"].items():
        all_eval["metrics"][key] = np.mean(std_lists[key])
        all_eval["std"][key] = np.std(std_lists[key])
    return all_eval


def plot_evaluation(eval, path=None):
    precision, recall, thresholds_pr = eval["curves"]["precision"], eval["curves"]["recall"], eval["curves"][
        "thresholds_pr"]
    fpr, tpr, thresholds_roc = eval["curves"]["fpr"], eval["curves"]["tpr"], eval["curves"]["thresholds_roc"]

    f, axarr = plt.subplots(1, 2, figsize=fig_dim)

    ###############
    # PRECISION CURVE
    ###############

    mean_rec = np.linspace(0, 1, 100)
    pres = []
    for pre, rec in zip(precision, recall):
        pres.append(interp(mean_rec, rec[::-1], pre[::-1]))
        pres[-1][0] = 1.0
        # axarr[0].plot(rec, pre, lw=1, alpha=0.05)

    mean_pre = np.mean(pres, axis=0)
    mean_pre[-1] = 0.0
    axarr[0].plot(mean_rec, mean_pre, color='b', label=r'Mean PR')

    std_pre = np.std(pres, axis=0)
    pre_upper = np.minimum(mean_pre + std_pre, 1)
    pre_lower = np.maximum(mean_pre - std_pre, 0)
    axarr[0].fill_between(mean_rec, pre_lower, pre_upper, color='grey', alpha=.2, label=r'$\pm$ 1 std. dev.')

    axarr[0].set_xlabel('Recall')
    axarr[0].set_ylabel('Precision')
    axarr[0].set_ylim([-0.05, 1.05])
    axarr[0].set_xlim([-0.05, 1.05])
    axarr[0].set_title("Precision Recall Curve")
    axarr[0].legend(loc="lower right")

    ###############
    # ROC CURVE
    ###############

    mean_fpr = np.linspace(0, 1, 100)
    tprs = []
    for fpr, tpr in zip(fpr, tpr):
        tprs.append(interp(mean_fpr, fpr, tpr))
        tprs[-1][0] = 0.0
        # axarr[1].plot(fpr, tpr, lw=1, alpha=0.05)

    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    axarr[1].plot(mean_fpr, mean_tpr, color='b', label=r'Mean ROC')

    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    axarr[1].fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2, label=r'$\pm$ 1 std. dev.')

    axarr[1].set_xlim([-0.05, 1.05])
    axarr[1].set_ylim([-0.05, 1.05])
    axarr[1].set_xlabel('False Positive Rate')
    axarr[1].set_ylabel('True Positive Rate')
    axarr[1].set_title('Receiver operating characteristic')
    axarr[1].legend(loc="lower right")
    lw = 2
    axarr[1].plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')

    title = " - ".join([i + ": " + str(j) for i, j in eval["info"].items()]) + "\n"
    title += " - ".join([i + "={0:.3f}".format(j) for i, j in eval["metrics"].items()])
    plt.suptitle(title)
    if path is None:
        plt.show()
    else:
        plt.savefig(path)
