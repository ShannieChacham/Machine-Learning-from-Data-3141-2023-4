import numpy as np
import pandas as pd


def pearson_correlation( x, y):
    """
    Calculate the Pearson correlation coefficient for two given columns of data.

    Inputs:
    - x: An array containing a column of m numeric values.
    - y: An array containing a column of m numeric values. 

    Returns:
    - The Pearson correlation coefficient between the two columns.    
    """
    r = 0.0

    x = np.array(x, dtype=float)
    y = np.array(y, dtype=float)
    
    mean_x = np.mean(x)
    mean_y = np.mean(y)
    
    numerator = np.sum((x - mean_x) * (y - mean_y))
    denominator = np.sqrt(np.sum((x - mean_x)**2) * np.sum((y - mean_y)**2))
    
    r = numerator / denominator

    return r

def feature_selection(X, y, n_features=5):
    """
    Select the best features using pearson correlation.

    Input:
    - X: Input data (m instances over n features).
    - y: True labels (m instances).

    Returns:
    - best_features: list of best features (names - list of strings).  
    """
    best_features = []

    if isinstance(X, np.ndarray):
        feature_names = [f"feature_{i}" for i in range(X.shape[1])]
    else:
        feature_names = X.columns.tolist()
        X = X.values

    y = np.array(y, dtype=float)

    # Calculate Pearson correlation coefficient for each feature with the target
    correlations = []
    for i in range(X.shape[1]):
        corr = pearson_correlation(X[:, i], y)
        correlations.append((abs(corr), feature_names[i]))

    correlations.sort(reverse=True, key=lambda x: x[0])

    best_features = [feature for _, feature in correlations[:n_features]]

    return best_features

class LogisticRegressionGD(object):
    """
    Logistic Regression Classifier using gradient descent.

    Parameters
    ------------
    eta : float
      Learning rate (between 0.0 and 1.0)
    n_iter : int
      Passes over the training dataset.
    eps : float
      minimal change in the cost to declare convergence
    random_state : int
      Random number generator seed for random weight
      initialization.
    """

    def __init__(self, eta=0.00005, n_iter=10000, eps=0.000001, random_state=1):
        self.eta = eta
        self.n_iter = n_iter
        self.eps = eps
        self.random_state = random_state

        # model parameters
        self.theta = None

        # iterations history
        self.Js = []
        self.thetas = []
       
        np.random.seed(random_state)
        self.theta = np.random.random(size=3)
        
    def fit(self, X, y):
        """
        Fit training data (the learning phase).
        Update the theta vector in each iteration using gradient descent.
        Store the theta vector in self.thetas.
        Stop the function when the difference between the previous cost and the current is less than eps
        or when you reach n_iter.
        The learned parameters must be saved in self.theta.
        This function has no return value.

        Parameters
        ----------
        X : {array-like}, shape = [n_examples, n_features]
          Training vectors, where n_examples is the number of examples and
          n_features is the number of features.
        y : array-like, shape = [n_examples]
          Target values.

        """
        # set random seed
        np.random.seed(self.random_state)

        X_normalized = self.normalize_features(X)

        a = np.ones((len(X_normalized), 1))
        X_normalized = np.append(a, X_normalized, axis=1)
        m = len(y)
        cost = self.compute_cost(X_normalized, y)

        for i in range(self.n_iter):
            self.Js.append(cost)
            h = self.hypothesis(X_normalized.T)
            self.theta = self.theta - self.eta * np.dot(h - y, X_normalized)
            self.thetas.append(self.theta)
            cost = self.compute_cost(X_normalized, y)
            if self.Js[-1] - cost < self.eps:
                break
                
    def compute_cost(self, X, y):
        numOfInstances = len(y)
        hypot = self.hypothesis(X.T)
        ylnhypot = -y * np.log(hypot)
        oneMinusylnhypot = -(1 - y) * np.log(1 - hypot)

        result = ylnhypot + oneMinusylnhypot
        result = np.sum(result)
        result = result / numOfInstances

        return result
   
    def hypothesis(self, x):
        return 1 / (1 + np.exp(-np.dot(self.theta, x)))


    def normalize_features(self, X):
        mean = np.mean(X, axis=0)
        std = np.std(X, axis=0)
        X_normalized = (X - mean) / std
        return X_normalized
            
    def predict(self, X):
        """
        Return the predicted class labels for a given instance.
        Parameters
        ----------
        X : {array-like}, shape = [n_examples, n_features]
        """
        preds = None

        t = np.ones((len(X), 1))
        X_normalized = np.append(t, self.normalize_features(X), axis=1)
        preds = np.where(self.hypothesis(X_normalized.T) > 0.5, 1, 0)

        return preds

    def accuracy(self, X, y):
        length = len(y)
        predictions = self.predict(X)
        count = np.sum(predictions == y)
        return count / length
        
def cross_validation(X, y, folds, algo, random_state):
    """
    This function performs cross validation as seen in class.

    1. shuffle the data and creates folds
    2. train the model on each fold
    3. calculate aggregated metrics

    Parameters
    ----------
    X : {array-like}, shape = [n_examples, n_features]
      Training vectors, where n_examples is the number of examples and
      n_features is the number of features.
    y : array-like, shape = [n_examples]
      Target values.
    folds : number of folds (int)
    algo : an object of the classification algorithm
    random_state : int
      Random number generator seed for random weight
      initialization.

    Returns the cross validation accuracy.
    """

    cv_accuracy = 0.0

    # set random seed
    np.random.seed(random_state)

    data = np.column_stack((X, y))
    np.random.shuffle(data)

    folds_data = np.array_split(data, folds)

    for i in range(folds):
        test = folds_data[i]
        train = np.concatenate(folds_data[:i] + folds_data[i + 1:], axis=0)
        trainX = train[:, :-1]
        trainY = train[:, -1]
        algo.fit(trainX, trainY)
        testX = test[:, :-1]
        testY = test[:, -1]
        accuracy = algo.accuracy(testX, testY)
        cv_accuracy += accuracy

    cv_accuracy /= folds
    return cv_accuracy

def norm_pdf(data, mu, sigma):
    """
    Calculate normal desnity function for a given data,
    mean and standrad deviation.
 
    Input:
    - x: A value we want to compute the distribution for.
    - mu: The mean value of the distribution.
    - sigma:  The standard deviation of the distribution.
 
    Returns the normal distribution pdf according to the given mu and sigma for the given x.    
    """
    p = None

    p = np.exp(-0.5*(((data-mu)/sigma)**2))/(sigma*np.sqrt(2*np.pi))

    return p

class EM(object):
    """
    Naive Bayes Classifier using Gauusian Mixture Model (EM) for calculating the likelihood.

    Parameters
    ------------
    k : int
      Number of gaussians in each dimension
    n_iter : int
      Passes over the training dataset in the EM proccess
    eps: float
      minimal change in the cost to declare convergence
    random_state : int
      Random number generator seed for random params initialization.
    """

    def __init__(self, k=1, n_iter=1000, eps=0.01, random_state=1991):
        self.k = k
        self.n_iter = n_iter
        self.eps = eps
        self.random_state = random_state

        np.random.seed(self.random_state)

        self.responsibilities = None
        self.weights = None
        self.mus = None
        self.sigmas = None
        self.costs = None

    # initial guesses for parameters
    def init_params(self, data):
        """
        Initialize distribution params
        """
        self.weights = np.ones(self.k) / self.k
        self.mus = np.random.randn(self.k)
        self.sigmas = np.ones(self.k)
        self.costs = None

    def expectation(self, data):
        """
        E step - This function should calculate and update the responsibilities
        """
        a = 0 
        b = 0

        samples = data.shape[0]
        self.responsibilities = np.zeros((samples, self.k))


        for j in range(self.k):
            b += self.weights[j]*norm_pdf(data, self.mus[j], self.sigmas[j]).flatten()

        for i in range(self.k):
            a = self.weights[i]*norm_pdf(data, self.mus[i], self.sigmas[i]).flatten()
            self.responsibilities[:, i] = (a / b)

    def maximization(self, data):
        """
        M step - This function should calculate and update the distribution params
        """
        for j in range(self.k):
            sum_r = np.sum(self.responsibilities[:, j])
            self.weights[j] = sum_r / len(data)

        for j in range(self.k):
            sum_rd = np.sum(self.responsibilities[:, j] * data.reshape(-1))
            self.mus[j] = sum_rd / (self.weights[j] * len(data))

        for j in range(self.k):
            sum_rd = np.sum(self.responsibilities[:, j] * ((data - self.mus[j]) ** 2).reshape(-1))
            self.sigmas[j] = np.sqrt(sum_rd / (self.weights[j] * len(data)))
        self.mus = self.mus.flatten()
        self.sigmas = self.sigmas.flatten()

    def fit(self, data):
        """
        Fit training data (the learning phase).
        Use init_params and then expectation and maximization function in order to find params
        for the distribution.
        Store the params in attributes of the EM object.
        Stop the function when the difference between the previous cost and the current is less than eps
        or when you reach n_iter.
        """
        np.random.seed(self.random_state)  
        self.init_params(data)
        self.costs = np.zeros(self.n_iter) 
        self.costs[0] = self.cost(data) 

        for i in range(1,self.n_iter): 
            self.expectation(data)
            self.maximization(data)
            self.costs[i] = self.cost(data)
            if np.abs(self.costs[i-1] - self.costs[i]) < self.eps: 
                break 

    def get_dist_params(self):
        return self.weights, self.mus, self.sigmas

    def cost(self,data):    
        Cost = 0
        for i in range(self.k):
            Cost += self.weights[i] * norm_pdf(data, self.mus[i], self.sigmas[i])
        Cost = np.sum(-np.log(Cost))
        return Cost


def gmm_pdf(data, weights, mus, sigmas):
    """
    Calculate gmm desnity function for a given data,
    mean and standrad deviation.
 
    Input:
    - data: A value we want to compute the distribution for.
    - weights: The weights for the GMM
    - mus: The mean values of the GMM.
    - sigmas:  The standard deviation of the GMM.
 
    Returns the GMM distribution pdf according to the given mus, sigmas and weights
    for the given data.    
    """
    pdf = None

    gaussian_densities = [
        weight * norm_pdf(data, mu, sigma)
        for weight, mu, sigma in zip(weights, mus, sigmas)
    ]
    pdf = np.sum(gaussian_densities, axis=0)

    return pdf

class NaiveBayesGaussian(object):
    """
    Naive Bayes Classifier using Gaussian Mixture Model (EM) for calculating the likelihood.

    Parameters
    ------------
    k : int
      Number of gaussians in each dimension
    random_state : int
      Random number generator seed for random params initialization.
    """

    def __init__(self, k=1, random_state=1991):
        self.k = k
        self.random_state = random_state
        self.prior = None
        self.ems = []
        
    def calculate_prior(self, data):
        _, counts = np.unique(data[:, -1], return_counts=True)
        self.prior = [count / data.shape[0] for count in counts]

    def init_params(self, data):
        self.calculate_prior(data)
        n_features = data.shape[1] - 1  
        self.ems = [[EM(k=self.k) for _ in range(n_features)] for _ in range(2)]
            
    def fit(self, X, y):
        """
        Fit training data.

        Parameters
        ----------
        X : array-like, shape = [n_examples, n_features]
          Training vectors, where n_examples is the number of examples and
          n_features is the number of features.
        y : array-like, shape = [n_examples]
          Target values.
        """
        data = np.column_stack((X, y))
        self.init_params(data)

        for class_val in range(2):
            class_data = data[data[:, -1] == class_val]
            for feature_index in range(X.shape[1]):
                self.ems[class_val][feature_index].fit(class_data[:, feature_index])


    def predict(self, X):
        """
        Return the predicted class labels for a given instance.
        Parameters
        ----------
        X : {array-like}, shape = [n_examples, n_features]
        """
        preds = None

        likelihood_l = np.zeros((len(X), 2))
        for class_val in range(2):
            class_li = self.prior[class_val]
            for i, em in enumerate(self.ems[class_val]):
                em_li = em.get_dist_params()
                class_li *= gmm_pdf(X[:, i], *em_li)
            likelihood_l[:, class_val] = class_li
        preds = np.argmax(likelihood_l, axis=1)
        preds= np.array(preds).reshape(-1,1)

        return preds.reshape(-1,1)

        
def accuracy(X, y, classifier):
    preds = classifier.predict(X)
    correct_preds = np.sum(preds == y)
    total_preds = len(y)
    return correct_preds / total_preds
    
def model_evaluation(x_train, y_train, x_test, y_test, k, best_eta, best_eps):
    ''' 
    Read the full description of this function in the notebook.

    You should use visualization for self debugging using the provided
    visualization functions in the notebook.
    Make sure you return the accuracies according to the return dict.

    Parameters
    ----------
    x_train : array-like, shape = [n_train_examples, n_features]
      Training vectors, where n_examples is the number of examples and
      n_features is the number of features.
    y_train : array-like, shape = [n_train_examples]
      Target values.
    x_test : array-like, shape = [n_test_examples, n_features]
      Training vectors, where n_examples is the number of examples and
      n_features is the number of features.
    y_test : array-like, shape = [n_test_examples]
      Target values.
    k : Number of gaussians in each dimension
    best_eta : best eta from cv
    best_eps : best eta from cv
    ''' 

    lor_train_acc = None
    lor_test_acc = None
    bayes_train_acc = None
    bayes_test_acc = None

    log_train = LogisticRegressionGD(eta=best_eta,eps=best_eps)
    log_train.fit(x_train,y_train)
    lor_train_acc= accuracy(x_train,y_train,log_train)
    lor_test_acc = accuracy(x_test,y_test,log_train)
    
    bayes_train = NaiveBayesGaussian(k=k)
    bayes_train.fit(x_train,y_train)
    bayes_train_acc=accuracy(x_train,y_train,bayes_train)
    bayes_test_acc = accuracy(x_test,y_test,bayes_train)

    return {'lor_train_acc': lor_train_acc,
            'lor_test_acc': lor_test_acc,
            'bayes_train_acc': bayes_train_acc,
            'bayes_test_acc': bayes_test_acc}

def generate_datasets():
    from scipy.stats import multivariate_normal
    '''
    This function should have no input.
    It should generate the two dataset as described in the jupyter notebook,
    and return them according to the provided return dict.
    '''
    dataset_a_features = []
    dataset_a_labels = []
    dataset_b_features = []
    dataset_b_labels = []
    
    mean_a1 = [1, 1, 1]
    cov_a1 = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    class_a1 = multivariate_normal(mean_a1, cov_a1)

    mean_a2 = [-1, -1, -1]
    cov_a2 = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    class_a2 = multivariate_normal(mean_a2, cov_a2)

    for _ in range(1000):
        features = class_a1.rvs()
        dataset_a_features.append(features)
        dataset_a_labels.append(0)

        features = class_a2.rvs()
        dataset_a_features.append(features)
        dataset_a_labels.append(1)

    mean_b1 = [0, 0, 0]
    cov_b1 = [[2, 0.5, 0.3], [0.5, 2, 0.1], [0.3, 0.1, 0.5]]
    class_b1 = multivariate_normal(mean_b1, cov_b1)

    mean_b2 = [0, 0, 0]
    cov_b2 = [[2, 0.1, 0.4], [0.1, 2, 0.2], [0.4, 0.2, 0.5]]
    class_b2 = multivariate_normal(mean_b2, cov_b2)

    for _ in range(1000):
        features = class_b1.rvs()
        dataset_b_features.append(features)
        dataset_b_labels.append(0)

        features = class_b2.rvs()
        dataset_b_features.append(features)
        dataset_b_labels.append(1)

    dataset_a_features = np.array(dataset_a_features)
    dataset_a_labels = np.array(dataset_a_labels)
    dataset_b_features = np.array(dataset_b_features)
    dataset_b_labels = np.array(dataset_b_labels)

    return{'dataset_a_features': dataset_a_features,
           'dataset_a_labels': dataset_a_labels,
           'dataset_b_features': dataset_b_features,
           'dataset_b_labels': dataset_b_labels
           }