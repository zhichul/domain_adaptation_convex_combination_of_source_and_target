import json
import os
import random
from itertools import product

import matplotlib.pyplot as plt

from dataset_loading import load_dataset, encode_dataset, load_vocab, \
    load_directory
from sklearn.linear_model import LogisticRegression
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import numpy as np

colormap = {0:"tab:red",1:"tab:blue"}
zeta = {
    ("books","dvd"): 2*(0.9391682184978274),
    ("dvd","books"): 2*(0.9391682184978274),
    ("books", "electronics"): 2*(0.9507389162561576),
    ("electronics", "books"): 2*(0.9507389162561576),
    ("books", "kitchen"): 2*(0.9399615754082613),
    ("kitchen", "books"): 2*(0.9399615754082613),
    ("dvd", "electronics"): 2*(0.894878706199461),
    ("electronics", "dvd"): 2*(0.894878706199461),
    ("dvd","kitchen"): 2*(0.8904037755637126),
    ("kitchen","dvd"): 2*(0.8904037755637126),
    ("electronics", "kitchen"): 2*(0.8594153052450559),
    ("kitchen", "electronics"): 2*(0.8594153052450559),
}
step = 0.1
proportions = [0.1, 0.2, 0.4, 0.8]
seeds = list(range(123,123+5))

def shuffled(seed, features, labels):
    random.seed(seed)
    inds = list(range(len(features)))
    random.shuffle(inds)
    features = [features[i] for i in inds]
    labels = [labels[i] for i in inds]
    return features, labels

def bound_exp1(root, vocab, fA, lA, fB, lB, A, B):
    # A is the source B is the target
    assert len(fA) == len(lA) and len(fB) == len(lB)
    lines = []
    for proportion in proportions:
        lines.append([])
        cutB = int(len(fB) * proportion)
        for alpha in np.arange(0.0,1.0001,step):
            beta = cutB/(cutB + len(fA))
            err = (1601/(len(fA) + cutB) * (alpha**2/beta + (1-alpha)**2/(1-beta)))**0.5 + (1-alpha) * zeta[(A,B)]
            lines[-1].append((alpha, err))
    for line, proportion in zip(lines, proportions):
        alpha = [pair[0] for pair in line]
        err = [pair[1] for pair in line]
        plt.plot(alpha, err, label=f"mT={int(proportion * len(fB))}")
        plt.legend()
    plt.savefig(os.path.join(root,"exp1_bound.png"))
    plt.close()

def bound_exp2(root, vocab, fA, lA, fB, lB, A, B):
    # A is the source B is the target
    assert len(fA) == len(lA) and len(fB) == len(lB)
    lines = []
    for proportion in proportions:
        lines.append([])
        cutA = int(len(fA) * proportion)
        cutB = int(len(fB) * 0.8)
        for alpha in np.arange(0.0,1.0001,step):
            beta = cutB/(cutB + cutA)
            err = (1601/(len(fA) + cutB) * (alpha**2/beta + (1-alpha)**2/(1-beta)))**0.5 + (1-alpha) * zeta[(A,B)]
            lines[-1].append((alpha, err))
    for line, proportion in zip(lines, proportions):
        alpha = [pair[0] for pair in line]
        err = [pair[1] for pair in line]
        plt.plot(alpha, err, label=f"mS={int(proportion * len(fB))}")
        plt.legend()
    plt.savefig(os.path.join(root,"exp2_bound.png"))
    plt.close()

def exp1(root, vocab, fA, lA, fB, lB):
    # A is the source B is the target
    assert len(fA) == len(lA) and len(fB) == len(lB)
    lines = []
    for seed in seeds:
        lines.append([])
        fA, lA = shuffled(seed, fA, lA)
        fB, lB = shuffled(seed, fB, lB)
        for proportion in proportions:
            lines[-1].append([])
            cutB = int(len(fB) * proportion)
            features = fA + fB[:cutB]
            labels = lA + lB[:cutB]
            test_features = fB[cutB:]
            test_labels = lB[cutB:]
            X, y = encode_dataset(vocab, features, labels)
            X_test, y_test = encode_dataset(vocab, test_features, test_labels)
            print(X.shape, y.shape)
            for alpha in np.arange(0.0,1.0001,step):
                beta = cutB/(cutB + len(fA))
                sample_weight = np.array([(1-alpha)/(1-beta)] * len(fA) + [alpha/beta] * cutB)
                model = LogisticRegression(penalty='l2',solver="lbfgs", max_iter=100000)
                model.fit(X, y, sample_weight=sample_weight)
                acc = model.score(X_test, y_test)
                lines[-1][-1].append((alpha, 1-acc))
                save_feature_importance_graph(model, [x[0] for x in vocab], f"{root}/feature_importance_{alpha}_{seed}.png")
    with open(os.path.join(root, "metrics.json"),"wt") as f:
        json.dump({"proportions":proportions, "lines":lines, "seeds":seeds},f,indent=4,sort_keys=True)
    lines = np.array(lines) #[seeds, proportions, alphas, 2]
    lines_mean = lines.mean(axis=0)
    lines_std = lines.std(axis=0)
    for lineavg, linetd, proportion in zip(lines_mean, lines_std, proportions):
        alpha = [pair[0] for pair in lineavg]
        err = [pair[1] for pair in lineavg]
        errstd = [pair[1] for pair in linetd]
        plt.errorbar(alpha, err, yerr=errstd, label=f"mT={int(proportion * len(fB))}")
        plt.legend()
    plt.savefig(os.path.join(root,"exp1.png"))
    plt.close()

def exp2(root, vocab, fA, lA, fB, lB):
    # A is the source B is the target
    assert len(fA) == len(lA) and len(fB) == len(lB)
    lines = []
    for seed in seeds:
        lines.append([])
        fA, lA = shuffled(seed, fA, lA)
        fB, lB = shuffled(seed, fB, lB)
        for proportion in proportions:
            lines[-1].append([])
            cutA = int(len(fA) * proportion)
            cutB = int(len(fB) * 0.8)
            features = fA[:cutA] + fB[:cutB]
            labels = lA[:cutA] + lB[:cutB]
            test_features = fB[cutB:]
            test_labels = lB[cutB:]
            X, y = encode_dataset(vocab, features, labels)
            X_test, y_test = encode_dataset(vocab, test_features, test_labels)
            print(X.shape, y.shape)
            for alpha in np.arange(0.0,1.0001,step):
                beta = cutB/(cutB + cutA)
                sample_weight = np.array([(1-alpha)/(1-beta)] * cutA + [alpha/beta] * cutB)
                model = LogisticRegression(penalty='l2',solver="lbfgs", max_iter=100000)
                model.fit(X, y, sample_weight=sample_weight)
                acc = model.score(X_test, y_test)
                lines[-1][-1].append((alpha, 1-acc))
                save_feature_importance_graph(model, [x[0] for x in vocab], f"{root}/2-feature_importance_{alpha}_{seed}.png")
    with open(os.path.join(root, "2-metrics.json"),"wt") as f:
        json.dump({"proportions":proportions, "lines":lines, "seeds":seeds},f,indent=4,sort_keys=True)
    lines = np.array(lines) #[seeds, proportions, alphas, 2]
    lines_mean = lines.mean(axis=0)
    lines_std = lines.std(axis=0)
    for lineavg, linetd, proportion in zip(lines_mean, lines_std, proportions):
        alpha = [pair[0] for pair in lineavg]
        err = [pair[1] for pair in lineavg]
        errstd = [pair[1] for pair in linetd]
        plt.errorbar(alpha, err, yerr=errstd, label=f"mS={int(proportion * len(fA))}")
        plt.legend()
    plt.savefig(os.path.join(root,"exp2.png"))
    plt.close()

def main():
    directories = ["books","dvd","electronics","kitchen"]
    vocab = load_vocab("vocab1600.txt")
    for A, B in product(directories, directories):
            if A == B: continue
            root = f"results/{A}-{B}"
            os.makedirs(root, exist_ok=True)
            pathA = os.path.join("data", A)
            pathB = os.path.join("data", B)
            fA, lA = load_directory(pathA, mode="l")
            fB, lB = load_directory(pathB, mode="l")
            # exp1(root, vocab, fA, lA, fB, lB)
            # exp2(root, vocab, fA, lA, fB, lB)
            bound_exp1(root, vocab, fA, lA, fB, lB, A, B)
            bound_exp2(root, vocab, fA, lA, fB, lB, A, B)

def save_feature_importance_graph(model, feature_names, filename, limit=20):
    if isinstance(model, LogisticRegression):
        imp = model.coef_[0,:]
    else:
        return
    coefficients_abs = list(np.abs(imp))
    sign = [1 if val > 0 else -1 for val in list(imp)]
    trios = list(zip(coefficients_abs, feature_names, sign))
    sorted_trios = sorted(trios,key=lambda x:x[0])
    x = [name +(" [+]" if sign == 1 else " [-]") for (_, name, sign) in sorted_trios]
    y = [abs_coeff for (abs_coeff, _, _) in sorted_trios]
    plt.barh(x[-limit:], y[-limit:], color='blue')
    plt.savefig(f'{filename}-l2-feature-importance.png', bbox_inches='tight', dpi=100)
    plt.close()
    # for xx,yy in zip(reversed(x), reversed(y)):
    #     print("%70s %-4s" % (xx, str(yy)))

if __name__ == "__main__":
    main()
