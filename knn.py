import numpy as np
from numpy.linalg import norm
import pandas as pd
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier


class Knn:
    # The default number of neighbours is 5
    def __init__(self, k=5):
        self.k = k

    # Load the training data and labels
    def fit(self, train_data, train_label):
        self.train_data = train_data.values
        self.train_lable = train_label.values

    def predict(self, test_data):
        # Predict the class label for a single sample p with k neighbours
        def single_predict(p, k):
            # Calculate the Euclidean distance between p and training data
            distance = norm(self.train_data - p, axis=1)
            distance_label = np.c_[distance.reshape(-1, 1), self.train_lable]
            # Sort the distance matrix in ascending order
            sorted_distance_label = distance_label[distance_label[:, 0].argsort()]
            # Sort class labels by the number of occurrences from the nearest k neighbours
            unique_count = np.asarray(np.unique(sorted_distance_label[:k][:, 1], return_counts=True))
            sorted_unique_count = unique_count[:, unique_count[1].argsort()[::-1]]
            i = 0
            for x in range(1, sorted_unique_count.shape[1]):
                # Find all classes labels that ended in a tie
                if sorted_unique_count[1][x] == sorted_unique_count[1][0]:
                    i = x
            if i:
                # Randomly choose a class label among those classes to break a tie
                vote = np.random.choice(sorted_unique_count[0][:i + 1], 1)[0]
            else:
                # Otherwise, stick with the top frequent class
                vote = sorted_unique_count[0][0]
            return vote

        # Generate the predictions for all samples in test data
        pred = [single_predict(p, self.k) for p in test_data.values]
        return pred


# Merge multiple csv file into one data and one label data frames. Optionally, we can exclude certain files
def merge_train_files(num_of_files, skip=None):
    df_train_data = pd.DataFrame()
    df_train_label = pd.DataFrame()
    for k in range(num_of_files):
        if k == skip:
            continue
        data = pd.read_csv('knn-dataset/trainData' + str(k + 1) + '.csv', header=None)
        df_train_data = df_train_data.append(data, ignore_index=True)
        label = pd.read_csv('knn-dataset/trainLabels' + str(k + 1) + '.csv', header=None)
        df_train_label = df_train_label.append(label, ignore_index=True)
    return df_train_data, df_train_label


def cross_validation(k_fold=10, max_neighbours=30):
    accuracies = []
    for i in range(1, max_neighbours + 1):
        accuracy = []
        for j in range(k_fold):
            df_validate_data = pd.read_csv('knn-dataset/trainData' + str(j + 1) + '.csv', header=None)
            df_validate_label = pd.read_csv('knn-dataset/trainLabels' + str(j + 1) + '.csv', header=None)
            df_train_data, df_train_label = merge_train_files(k_fold, skip=j)
            # Create a Knn classifier
            clf = Knn(k=i)
            clf.fit(df_train_data, df_train_label)
            pred = clf.predict(df_validate_data)
            accuracy.append(accuracy_score(df_validate_label, pred))
            # At the end of each k-fold cv, calculate the average accuracy
            if j == k_fold - 1:
                avg_accuracy = np.mean(np.array(accuracy)) * 100
                accuracies.append(avg_accuracy)
                print('Number of neighbours = {:2}, accuracy = {:4.1f}%'.format(i, avg_accuracy))
    # Find the index of the max accuracy so we can get k by adding one
    optimal_k = np.argmax(np.array(accuracies)) + 1
    print('The best k = ', optimal_k)
    return optimal_k, accuracies


optimal_k, y = cross_validation()

# Find accuracy for the test set
df_train_data, df_train_label = merge_train_files(10)
df_test_data = pd.read_csv('knn-dataset/testData.csv', header=None)
df_test_label = pd.read_csv('knn-dataset/testLabels.csv', header=None)
clf = Knn(k=optimal_k)
clf.fit(df_train_data, df_train_label)
pred = clf.predict(df_test_data)
print('The accuracy for the test set = {:4.1f}%'.format(100 * accuracy_score(df_test_label, pred)))
# Try Knn from sklearn package
clf_ = KNeighborsClassifier(n_neighbors=optimal_k)
clf_.fit(df_train_data, df_train_label.values.reshape((-1,)))
pred_ = clf_.predict(df_test_data)
print('sklearn: The accuracy for the test set = {:4.1f}%'.format(100 * accuracy_score(df_test_label, pred)))

# Plot the relationship between k and accuracy
x = list(range(1, 31))
plt.plot(x, y)
plt.xlabel('Number of Neighbours', fontsize=14)
plt.ylabel('10-Fold Cross Validation Accuracy', fontsize=14)
plt.title('Number of Neighbours vs 10-Fold Cross Validation Accuracy', fontsize=18)
plt.show()
