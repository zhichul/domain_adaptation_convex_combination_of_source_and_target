import os
from collections import defaultdict
import numpy as np
import random

def load_dataset(path):
    features = []
    labels = []
    with open(path, "rt") as f:
        for line in f:
            line = line.strip()
            cells = line.split()
            label_cell = cells[-1]
            feature_cells = cells[:-1]
            if "positive" in label_cell:
                label = 1
            elif "negative" in label_cell:
                label = 0
            else:
                assert False
            feature_names = []
            feature_vals = []
            for cell in feature_cells:
                feature = cell.split(":")
                assert len(feature) == 2
                feature_name = feature[0]
                feature_val = int(feature[1])
                feature_names.append(feature_name)
                feature_vals.append(feature_val)
            features.append((feature_names,feature_vals))
            labels.append(label)
    assert len(labels) == len(features)
    return features, labels


def load_directory(dir, mode="l"):
    """

    :param dir:
    :param mode: l for labeled, u for unlabeled, and b for both
    :return:
    """
    assert os.path.isdir(dir)
    random.seed(123)
    features = []
    labels = []
    for file in os.listdir(dir):
        if not file.endswith("review"):
            continue
        if mode == "l" and file.startswith("unlabeled"):
            continue
        if mode == "u" and not file.startswith("unlabeled"):
            continue
        path = os.path.join(dir, file)
        f, l = load_dataset(path)
        features.extend(f)
        labels.extend(l)
    assert len(labels) == len(features)
    return features, labels


def load_directories(*dirs, mode="l"):
    features = []
    labels = []
    for dir in dirs:
        f, l = load_directory(dir, mode=mode)
        features.extend(f)
        labels.extend(l)
    assert len(labels) == len(features)
    return features, labels


def gen_sorted_feature_count_pairs(features, limit=None):
    vocab = defaultdict(int)
    for feature_names, feature_vals in features:
        for feature_name, feature_val in zip(feature_names,feature_vals):
            vocab[feature_name] += feature_val
    if limit is None:
        return sorted([(k,v) for k,v in vocab.items()],key=lambda x:x[1],reverse=True)
    else:
        return sorted([(k,v) for k,v in vocab.items()],key=lambda x:x[1],reverse=True)[:limit]

def gen_vocab_by_feature_count_pairs(feature_count_pairs):
    vocab = []
    for i, (feature, _) in enumerate(feature_count_pairs):
        vocab.append((feature,i))
    return vocab

def save_vocab(vocab, file):
    with open(file, "wt") as f:
        for feature, ind in vocab:
            print(f"{feature}:{ind}", file=f)

def load_vocab(file):
    vocab = []
    with open(file, "rt") as f:
        for line in f:
            item = line.strip().split(":")
            vocab.append((item[0], int(item[1])))
    return vocab

def encode_dataset(vocab, features, labels):
    quickmap = dict(vocab)
    X = []
    y = []
    for ((feature_names, feature_vals), label) in zip(features, labels):
        x = np.zeros((len(vocab),))
        for feature_name, feature_val in zip(feature_names, feature_vals):
            if feature_name in quickmap:
                x[quickmap[feature_name]] += feature_val
            else:
                pass
        if x.sum() == 0:
            print("warn: keeping example with all zero feature")
        else:
            x /= np.linalg.norm(x,1)
        X.append(x)
        y.append(label)

    X = np.array(X)
    y = np.array(y)
    assert X.shape[0] == y.shape[0]
    return X, y


if __name__ == "__main__":
    features, labels = load_directories("data/books","data/dvd","data/electronics","data/kitchen", mode="b")
    feature_count_pairs = gen_sorted_feature_count_pairs(features)
    vocab = gen_vocab_by_feature_count_pairs(feature_count_pairs)
    save_vocab(feature_count_pairs, "feature_count_pairs.txt")
    save_vocab(vocab, "vocab.txt")
    feature_count_pairs1600 = gen_sorted_feature_count_pairs(features, limit=1600)
    vocab1600 = gen_vocab_by_feature_count_pairs(feature_count_pairs1600)
    save_vocab(feature_count_pairs1600, "feature_count_pairs1600.txt")
    save_vocab(vocab1600, "vocab1600.txt")
    X, y = encode_dataset(vocab1600, features, labels)
    print(X, y)
