from tqdm import tqdm
from sklearn.tree import DecisionTreeClassifier
from dataUtils import load_dataset
import matplotlib.pyplot as plt
import numpy as np


class boosTran:
    def __init__(self, num_learner: int = 30, bootstrap_ratio = 0.9) -> None:
        self.num_learner = num_learner
        self.tree_kwargs = dict(criterion="entropy", class_weight="balanced", max_depth=1)
        self.bootstrap_ratio = bootstrap_ratio
    
    
    def fit(self, X, y):
        # Variable initialization
        length_sample = len(X)
        self.tree = [DecisionTreeClassifier(**self.tree_kwargs) for _ in range(self.num_learner)]
        P = np.empty((length_sample, self.num_learner))
        sample_index = np.arange(length_sample)
        bootstrap_length = int(self.bootstrap_ratio * length_sample)
        weight = np.ones(bootstrap_length) / bootstrap_length
        # Main loop
        for i in range(self.num_learner):
            np.random.shuffle(sample_index)
            # weight = np.random.rand(length_sample)
            # weight /= sum(weight)
            tmp_x = X[sample_index[:bootstrap_length], :]
            tmp_y = y[sample_index[:bootstrap_length]]
            tmp_tree = self.tree[i]
            tmp_tree.fit(tmp_x, tmp_y, sample_weight= weight)
            y_hat = tmp_tree.predict(X)
            weight *= np.exp((y != y_hat)[sample_index[:bootstrap_length]])
            weight /= np.sum(weight)
            P[:, i] = y_hat
        # np.savetxt('matrixTemplate.csv', P, delimiter=',')
        rank = np.linalg.matrix_rank(P)

        # basis
        self.basis = np.linalg.pinv(P) @ y
        self.basis /= np.sum(self.basis)
    
    def prediction(self, X):
        length_sample = len(X)
        tmp = np.zeros((length_sample, self.num_learner))
        for i, tree in enumerate(self.tree):
            tmp[:, i] = tree.predict(X)
        result =  tmp @ self.basis
        result = (np.sign(result - 0.5) + 1) / 2
        return result.astype(int)


def demo():
    # Prepare dataset
    X_train, X_test, y_train, y_test = load_dataset("df_arabica_clean.csv")
    # Initialize 
    bs = boosTran()
    bs.fit(X_train, y_train)
    print(bs.basis)
    yhat = bs.prediction(X_test)
    asd = yhat == y_test
    print(np.sum(asd) / len(yhat))

def main():
    # Prepare dataset
    X_train, X_test, y_train, y_test = load_dataset("df_arabica_clean.csv")
    average = 10

    plt.figure()
    x = range(1, 80, 5)
    ratio = [.1, .2, .3, .4, .5, .6, .7, .8, .9, 1.]
    for r in tqdm(ratio):
        result = np.zeros(len(x))
        # averaging result
        for _ in range(average):
            for idx, i in enumerate(x):
                bs = boosTran(num_learner=i, bootstrap_ratio=r)
                bs.fit(X_train, y_train)
                yhat = bs.prediction(X_test)
                asd = yhat == y_test
                result[idx] += np.sum(asd) / len(yhat)
        result /= average
        plt.plot(x, result)
    plt.legend(ratio)
    plt.show()


if __name__ == "__main__":
    main()