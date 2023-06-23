from sklearn.tree import DecisionTreeClassifier
from dataUtils import load_dataset
import numpy as np


class boosTran:
    def __init__(self, num_learner: int = 30) -> None:
        self.num_learner = num_learner
        self.tree_kwargs = dict(criterion="entropy", class_weight="balanced", max_depth=1)
        self
    
    
    def fit(self, X, y):
        length_sample = len(X)
        weight = np.ones(length_sample) / length_sample
        self.tree = [DecisionTreeClassifier(**self.tree_kwargs) for _ in range(self.num_learner)]
        P = np.empty((length_sample, self.num_learner))
        for i in range(self.num_learner):
            # weight = np.random.rand(length_sample)
            # weight /= sum(weight)
            tmp_tree = self.tree[i]
            tmp_tree.fit(X, y, sample_weight= weight)
            y_hat = tmp_tree.predict(X)
            weight *= np.exp((y != y_hat))
            weight /= np.sum(weight)
            P[:, i] = y_hat
        # np.savetxt('matrixTemplate.csv', P, delimiter=',')
        rank = np.linalg.matrix_rank(P)
        print('rank= ' , rank)

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


def main():
    # Prepare dataset
    X_train, X_test, y_train, y_test = load_dataset("df_arabica_clean.csv")
    # Initialize 
    bs = boosTran(20)
    bs.fit(X_train, y_train)
    print(bs.basis)
    yhat = bs.prediction(X_test)
    asd = yhat == y_test
    print(np.sum(asd) / len(yhat))


if __name__ == "__main__":
    main()