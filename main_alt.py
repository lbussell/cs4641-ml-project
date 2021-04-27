from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import validation_curve
from sklearn.svm import SVC
from sklearn.model_selection import RandomizedSearchCV

class RandomForest(object):
    def __init__(self, n_estimators = 10, max_depth = 3, max_features = 0.8, bootstrap = True, oob_score = True):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.oob_score = oob_score
        self.model = None
        self.train = False
    
    def training(self, X, Y, **kwargs):
        self.model  = RandomForestClassifier(n_estimators = self.n_estimators, max_depth = self.max_depth, max_features = self.max_features, bootstrap = self.bootstrap, oob_score = self.oob_score, **kwargs)
        self.model.fit(X, Y)
        self.train = True
        
    def testing(self, X):
        if(self.train is False):
            raise TestException
        return self.model.predict(X)
    
    def testingAccuracy(self, X, Y):
        if(self.train is False):
            raise TestException
        return self.model.score(X, Y)
    
    def run(self, X, Y, tests_size = 0.05):
        x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = tests_size, shuffle = False)
        self.training(x_train, y_train)
        return self.testingAccuracy(x_test, y_test)

class SVM(object):
    def __init__(self, kernel = 'rbf', regularization = 1.0, gamma = 'scale'):
        self.C = regularization
        self.kernel = kernel
        self.gamma = gamma
        self.model = None
        self.train = False
        
    def training(self, X, Y, **kwargs):
        self.model = SVC(C = self.C, kernel = self.kernel, gamma = self.gamma, **kwargs)
        self.model.fit(X, Y)
        self.train = True
    
    def testing(self, X):
        if(self.train is False):
            raise TestException
        return self.model.predict(X)
    
    def testingAccuracy(self, X, Y):
        if(self.train is False):
            raise TestException
        return self.model.score(X, Y)
    
    def run(self, X, Y, tests_size = 0.05):
        x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = tests_size, shuffle = False)
        self.training(x_train, y_train)
        return self.testingAccuracy(x_test, y_test)

class TestingException(Exception):
    def __init__(self, message = "Model must be trained first"):
        self.message = message
        super().__init__(self.message)

def optimizeRF(x_val, y_val):
    rf_n_estimators = [int(x) for x in np.linspace(5, 1000, 5)]

    rf_max_depth = [int(x) for x in np.linspace(5, 50, 5)]

    rf_max_features = [x in np.linspace(0.5, 1.0, 0.1)]

    rf_max_features.append('sqrt')
    rf_max_features.append('log2')
    
    # Method of selecting samples for training each tree
    rf_bootstrap = [True, False]

    # Create the grid
    rf_grid = {'n_estimators': rf_n_estimators,
                   'max_depth': rf_max_depth,
                   'max_features': rf_max_features,
                   'bootstrap': rf_bootstrap}
    
    rf_base = RandomForestClassifier()

    rf_random = RandomizedSearchCV(estimator = rf_base, param_distributions = rf_grid, 
                               n_iter = 200, cv = 4, verbose = 0, random_state = 42, 
                               n_jobs = -1)

    # Fit the random search model
    rf_random.fit(x_val, y_val)

    # View the best parameters from the random search
    return rf_random.best_params_
    
def plotValidationCurve(x_train, y_train, params_range, params_name):
    train_scores, test_scores = validation_curve(
                                    RandomForestClassifier(),
                                    X = x_train, y = y_train, 
                                    param_name = params_name, 
                                    param_range = params_range, cv = 3)
    param_range = params_range
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    plt.title("Validation Curve")
    plt.xlabel(r"$\gamma$")
    plt.ylabel("Score")
    plt.ylim(0.0, 1.1)
    lw = 2
    plt.semilogx(param_range, train_scores_mean, label="Training score",
                 color="darkorange", lw=lw)
    plt.fill_between(param_range, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.2,
                     color="darkorange", lw=lw)
    plt.semilogx(param_range, test_scores_mean, label="Cross-validation score",
                 color="navy", lw=lw)
    plt.fill_between(param_range, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.2,
                     color="navy", lw=lw)
    plt.legend(loc="best")
    plt.show()