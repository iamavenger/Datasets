from MNIST_dataloadercsv import load_mnistcsv
import numpy as np
from collections import Counter

class knn():
    def __init__(self, k = 3):
        self.k = k
    
    def fit(self, train_data, train_labels):
        self.train_data = train_data
        self.train_labels = train_labels
    
    def predict(self, test_data):
        predictions = []
        for test_sample in test_data:
            distances = np.linalg.norm(self.train_data - test_sample, axis=1)
            nearest_neighbours = np.argsort(distances)[:self.k]
            nearest_labels = self.train_labels[nearest_neighbours]
            
            most_common_labels = Counter(nearest_labels).most_common(1)[0][0]
            predictions.append(most_common_labels)
        return np.array(predictions)
    
train_data, train_labels = load_mnistcsv(r"MNIST_DATASET_CSV/mnist_train.csv")
test_data, test_labels = load_mnistcsv(r"MNIST_DATASET_CSV/mnist_test.csv")
print(test_data.shape)
print(train_data.shape)


train_data = train_data.reshape(train_data.shape[0], -1)
print(train_data.shape)
test_data = test_data.reshape(test_data.shape[0], -1)
print(test_data.shape)

KNN = knn(k = 3)
KNN.fit(train_data[:3000], train_labels[:3000])

accuracy = np.mean(KNN.predict(test_data[:500]) == test_labels[:500])
print(f"Accuracy: {accuracy * 100:.2f}%")

