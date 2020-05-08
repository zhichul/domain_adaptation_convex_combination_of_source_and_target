import os

from dataset_loading import load_dataset

if __name__ == "__main__":
    directories = ["data/books","data/dvd","data/electronics","data/kitchen"]
    for directory in directories:
        print(directory)
        for file in os.listdir(directory):
            path = os.path.join(directory, file)
            features,labels = load_dataset(path)
            print(path, len(features))