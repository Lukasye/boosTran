from tqdm import tqdm
from sklearn.tree import DecisionTreeClassifier
from dataUtils import load_dataset
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize

# For linear Regression
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression


# For optimizzation
def objective(a, P, y):
    return sum((P.dot(a) - y)**2)

class boosTran:
    def __init__(self, bootstrap_ratio = 0.9, method: str = 'pinv') -> None:
        self.tree_kwargs = dict(criterion="entropy", class_weight="balanced", max_depth=2)
        self.bootstrap_ratio = bootstrap_ratio
        self.method = method

    
    def fit(self, X, y):
        # Variable initialization
        self.num_learner = X.shape[1]
        self.basis = np.zeros(self.num_learner)
        self.constraints = [{'type': 'ineq', 'fun': lambda a: a[i]} for i in range(len(self.basis))]
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
        if self.method == 'pinv':
            self.basis = np.linalg.pinv(P) @ y
        elif self.method == 'svd':
            _, S, Vh = np.linalg.svd(P)
            # print("Smallest Eigen value: ", S[rank - 1])
            # print(Vh[rank - 1, :])
            self.basis = Vh[rank - 1, :]
        elif self.method == 'opt':
            result = minimize(objective, self.basis, args=(P, y), constraints=self.constraints)
            self.basis = result.x
        elif self.method == 'proj':
            for i in range(self.num_learner):
                self.basis[i] = y.dot(P[:, i])
        self.basis /= np.sum(self.basis)

    
    def prediction(self, X):
        length_sample = len(X)
        tmp = np.zeros((length_sample, self.num_learner))
        for i, tree in enumerate(self.tree):
            tmp[:, i] = tree.predict(X)
        result =  tmp @ self.basis
        result = (np.sign(result - 0.5) + 1) / 2
        return result.astype(int)

def benchmark(bootstrap_ratio, max_depth, X_train, y_train, X_test, y_test, 
              epoche: int = 10,
              method: str = 'pinv') -> float:
    result = []
    for _ in range(epoche):
        bs = boosTran(bootstrap_ratio, method=method)
        bs.tree_kwargs['max_depth'] = max_depth
        bs.fit(X_train, y_train)
        yhat = bs.prediction(X_test)
        asd = yhat == y_test
        result.append(np.sum(asd) / len(yhat))
    return np.mean(result)

def demo():
    # Prepare dataset
    X_train, X_test, y_train, y_test = load_dataset("df_arabica_clean.csv")
    # Initialize 
    bs = boosTran(method='proj')
    bs.fit(X_train, y_train)
    print(bs.basis)
    yhat = bs.prediction(X_test)
    asd = yhat == y_test
    print(np.sum(asd) / len(yhat))

def compare_ratio_numlearners():
    # Prepare dataset
    X_train, X_test, y_train, y_test = load_dataset("df_arabica_clean.csv")
    average = 10

    plt.figure()
    x = range(1, 80, 5)
    ratio = [.1, .2, .3, .4, .5, .6, .7, .8, .9, 1.]
    for r in tqdm(ratio):
        result = []
        # averaging result
        for idx, i in enumerate(x):
            result.append(benchmark(r, X_train, y_train, X_test, y_test, 10))
        plt.plot(x, result)
    plt.legend(ratio)
    plt.show()

def main():
    # Prepare dataset
    X_train, X_test, y_train, y_test = load_dataset("df_arabica_clean.csv")
    average = 10

    plt.figure()
    x = range(1, 7)
    ratio = [.1, .2, .3, .4, .5, .6, .7, .8, .9, 1.]
    for i in x:
        result = []
        for r in tqdm(ratio):
        # averaging result
            result.append(benchmark(r, i, X_train, y_train, X_test, y_test, 10, method='opt'))
        plt.plot(ratio, result)
    plt.legend(x)
    plt.show()


def linear_regression():
    # Prepare dataset
    X_train, X_test, y_train, y_test = load_dataset("df_arabica_clean.csv")
    model=LinearRegression()
    model.fit(X_train,y_train)
    y_pred=model.predict(X_test)
    y_pred = (np.sign(y_pred - 0.5) + 1)/2
    y_pred.astype(int)
    result = y_pred == y_test
    print(np.sum(result) / len(result))


def compare_methods():
    # Prepare dataset
    X_train, X_test, y_train, y_test = load_dataset("df_arabica_clean.csv")
    methods = ['proj', 'svd', 'pinv', 'opt']
    ratio = [.1, .2, .3, .4, .5, .6, .7, .8, .9, 1.]
    
    plt.figure()
    for m in methods:
        result = []
        for r in tqdm(ratio):
            result.append(benchmark(r, 4, X_train, y_train, X_test, y_test, 10, method=m))
        plt.plot(ratio, result)
    plt.legend(methods)
    plt.show()




if __name__ == "__main__":
    # main()
    # demo()
    # linear_regression()
    compare_methods()