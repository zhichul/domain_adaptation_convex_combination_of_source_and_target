import os
import matplotlib.pyplot as plt

from dataset_loading import load_dataset, encode_dataset, load_vocab
from sklearn.linear_model import LogisticRegression
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import numpy as np

colormap = {0:"tab:red",1:"tab:blue"}
def main():
    directories = ["data/books","data/dvd","data/electronics","data/kitchen"]
    vocab = load_vocab("vocab1600.txt")
    for i in range(len(directories)):
        for j in range(i+1, len(directories)):
            # if i == j:
            #     continue
            A = directories[i]
            B = directories[j]
            file = "unlabeled.review"
            pathA = os.path.join(A, file)
            pathB = os.path.join(B, file)
            fA, _ = load_dataset(pathA)
            fB, _ = load_dataset(pathB)
            features = fA[:int(0.8*len(fA))] + fB[:int(0.8*len(fB))]
            labels = [1] * len(fA[:int(0.8*len(fA))]) + [0] * len(fB[:int(0.8*len(fB))])
            features_val = fA[int(0.8*len(fA)):] + fB[int(0.8*len(fB)):]
            labels_val = [1] * len(fA[int(0.8*len(fA)):]) + [0] * len(fB[int(0.8*len(fB)):])
            assert len(features) == len(labels)
            X, y = encode_dataset(vocab, features, labels)
            X_val, y_val = encode_dataset(vocab, features_val, labels_val)
            model = LogisticRegression(penalty='l2',solver="lbfgs", max_iter=100000)
            model.fit(X, y)
            zeta = model.score(X_val, y_val)
            print(A, B, zeta)
            save_feature_importance_graph(model, [x[0] for x in vocab], f"{A}-{B}".replace("/","-"))
            # X_embedded = PCA(n_components=2).fit_transform(X)
            # for x, label in zip(X_embedded, y):
            #     plt.scatter(x[0],x[1],color=colormap[label])
            # plt.savefig(f"{A}-{B}.png".replace("/","-"))
            # plt.close()

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
