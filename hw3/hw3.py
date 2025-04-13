import numpy as np

class conditional_independence():

    def __init__(self):

        # You need to fill the None value with *valid* probabilities
        self.X = {0: 0.3, 1: 0.7}  # P(X=x)
        self.Y = {0: 0.3, 1: 0.7}  # P(Y=y)
        self.C = {0: 0.5, 1: 0.5}  # P(C=c)

        self.X_Y = {
            (0, 0): 0.05,
            (0, 1): 0.25,
            (1, 0): 0.25,
            (1, 1): 0.45
        }  # P(X=x, Y=y)

        self.X_C = {
            (0, 0): 0.15,
            (0, 1): 0.15,
            (1, 0): 0.35,
            (1, 1): 0.35
        }  # P(X=x, C=y)

        self.Y_C = {
            (0, 0): 0.15,
            (0, 1): 0.15,
            (1, 0): 0.35,
            (1, 1): 0.35
        }  # P(Y=y, C=c)

        self.X_Y_C = {
            (0, 0, 0): 0.045,
            (0, 0, 1): 0.045,
            (0, 1, 0): 0.105,
            (0, 1, 1): 0.105,
            (1, 0, 0): 0.105,
            (1, 0, 1): 0.105,
            (1, 1, 0): 0.245,
            (1, 1, 1): 0.245,
        }  # P(X=x, Y=y, C=c)

    def is_X_Y_dependent(self):
        """
        return True iff X and Y are depndendent
        """
        X = self.X
        Y = self.Y
        X_Y = self.X_Y
        for (n,m) in X_Y:
            if not np.isclose(X[n]*Y[m], X_Y[(n,m)]):
                return True
        
        return False

    def is_X_Y_given_C_independent(self):
        """
        return True iff X_given_C and Y_given_C are indepndendent
        """
        X = self.X
        Y = self.Y
        C = self.C
        X_C = self.X_C
        Y_C = self.Y_C
        X_Y_C = self.X_Y_C

        for (x,y,c) in X_Y_C:
            if not np.isclose(X_Y_C[(x,y,c)],X_C[x,c]*Y_C[y,c]/C[c]):
                return False
    
        return True

def poisson_log_pmf(k, rate):
    """
    k: A discrete instance
    rate: poisson rate parameter (lambda)

    return the log pmf value for instance k given the rate
    """
    log_p = None
    lamda = rate
    probability = (np.power(lamda, k) * np.power(np.math.e,-lamda))/np.math.factorial(k)
    log_p = np.log(probability)
    return log_p

def get_poisson_log_likelihoods(samples, rates):
    """
    samples: set of univariate discrete observations
    rates: an iterable of rates to calculate log-likelihood by.

    return: 1d numpy array, where each value represent that log-likelihood value of rates[i]
    """
    likelihoods = None
    likelihoods = np.empty(len(rates))
    i=0
    for rate in rates: 
        log_probs = np.array([poisson_log_pmf(k, rate) for k in samples])
        likelihoods[i] = np.sum(log_probs)
        i=i+1;

    return likelihoods

def possion_iterative_mle(samples, rates):
    """
    samples: set of univariate discrete observations
    rate: a rate to calculate log-likelihood by.

    return: the rate that maximizes the likelihood 
    """
    rate = 0.0
    likelihoods = get_poisson_log_likelihoods(samples, rates)
    index_rate = np.argmax(likelihoods)
    rate = rates[index_rate]
    return rate

def possion_analytic_mle(samples):
    """
    samples: set of univariate discrete observations

    return: the rate that maximizes the likelihood
    """
    mean = None
    sumSamples = np.sum(samples)
    mean = sumSamples/samples.size
    return mean

def normal_pdf(x, mean, std):
    """
    Calculate normal desnity function for a given x, mean and standrad deviation.
 
    Input:
    - x: A value we want to compute the distribution for.
    - mean: The mean value of the distribution.
    - std:  The standard deviation of the distribution.
 
    Returns the normal distribution pdf according to the given mean and std for the given x.    
    """
    p = None
    d = (x - mean)/ std
    powerOfPi = np.power(d, 2) * (-0.5)
    numerator = np.power(np.math.e, d) 
    denominator = np.sqrt(np.power(std, 2)* 2 * mean)
    p = numerator/ denominator
    return p

class NaiveNormalClassDistribution():
    def __init__(self, dataset, class_value):
        """
        A class which encapsulates the relevant parameters(mean, std) for a class conditinoal normal distribution.
        The mean and std are computed from a given data set.
        
        Input
        - dataset: The dataset as a 2d numpy array, assuming the class label is the last column
        - class_value : The class to calculate the parameters for.
        """
        self.dataset = dataset
        self.dataset_val = dataset[dataset[:,2]==class_value]
        self.class_value = class_value
        self.mean = [np.mean(self.dataset_val[:,0]),np.mean(self.dataset_val[:,1])]
        self.std = [np.std(self.dataset_val[:,0]), np.std(self.dataset_val[:,1])]
    
    def get_prior(self):
        """
        Returns the prior porbability of the class according to the dataset distribution.
        """
        prior = None                                      
        prior = len(self.dataset_val)/len(self.dataset) 
        return prior
    
    def get_instance_likelihood(self, x):
        """
        Returns the likelihhod porbability of the instance under the class according to the dataset distribution.
        """
        likelihood = None
        tempP = normal_pdf(x[0], self.mean[0], self.std[0])
        humidityP = normal_pdf(x[1], self.mean[1], self.std[1])
        likelihood = tempP * humidityP
        return likelihood
    
    def get_instance_posterior(self, x):
        """
        Returns the posterior porbability of the instance under the class according to the dataset distribution.
        * Ignoring p(x)
        """
        posterior = None
        posterior = self.get_instance_likelihood(x)*self.get_prior()
        return posterior

class MAPClassifier():
    def __init__(self, ccd0 , ccd1):
        """
        A Maximum a posteriori classifier. 
        This class will hold 2 class distributions. 
        One for class 0 and one for class 1, and will predict an instance
        using the class that outputs the highest posterior probability 
        for the given instance.
    
        Input
            - ccd0 : An object contating the relevant parameters and methods 
                     for the distribution of class 0.
            - ccd1 : An object contating the relevant parameters and methods 
                     for the distribution of class 1.
        """
        self.ccd0 = ccd0
        self.ccd1 = ccd1
    
    def predict(self, x):
        """
        Predicts the instance class using the 2 distribution objects given in the object constructor.
    
        Input
            - An instance to predict.
        Output
            - 0 if the posterior probability of class 0 is higher and 1 otherwise.
        """
        pred = None
        ccd0_Posterior = self.ccd0.get_instance_posterior(x)
        ccd1_Posterior = self.ccd1.get_instance_posterior(x)
        if ccd0_Posterior > ccd1_Posterior:
            pred = 0
        else: 
            pred =1
        return pred

def compute_accuracy(test_set, map_classifier):
    """
    Compute the accuracy of a given a test_set using a MAP classifier object.
    
    Input
        - test_set: The test_set for which to compute the accuracy (Numpy array). where the class label is the last column
        - map_classifier : A MAPClassifier object capable of prediciting the class for each instance in the testset.
        
    Ouput
        - Accuracy = #Correctly Classified / test_set size
    """
    acc = None
    num_correct = 0
    test_set_size = len(test_set)

    for instance in test_set:
        true_class = instance[-1]
        predicted_class = map_classifier.predict(instance[:-1])
        if predicted_class == true_class:
            num_correct += 1

    acc = num_correct / test_set_size
    return acc

def multi_normal_pdf(x, mean, cov):
    """
    Calculate multi variable normal desnity function for a given x, mean and covarince matrix.
 
    Input:
    - x: A value we want to compute the distribution for.
    - mean: The mean vector of the distribution.
    - cov:  The covariance matrix of the distribution.
 
    Returns the normal distribution pdf according to the given mean and var for the given x.    
    """
    pdf = None
    x = np.array([x[0],x[1]])
    determinant = np.linalg.det(cov)
    dimension = len(x)
    transpose = np.subtract(x,mean).T
    covInverse = np.linalg.inv(cov)
    
    leftPart = np.power(2*np.pi,-dimension/2)*np.power(determinant,-0.5)
    rightExponent = np.dot(covInverse,np.subtract(x,mean))
    rightPart = np.exp(np.dot(-0.5 * transpose, rightExponent))
    
    pdf = leftPart * rightPart
    return pdf

class MultiNormalClassDistribution():

    def __init__(self, dataset, class_value):
        """
        A class which encapsulate the relevant parameters(mean, cov matrix) for a class conditinoal multi normal distribution.
        The mean and cov matrix (You can use np.cov for this!) will be computed from a given data set.
        
        Input
        - dataset: The dataset as a numpy array
        - class_value : The class to calculate the parameters for.
        """
        self.dataset = dataset
        self.dataset_val = dataset[dataset[:,2]==class_value]
        self.class_value = class_value
        self.mean = np.array([np.mean(self.dataset_val[:,0]),np.mean(self.dataset_val[:,1])])
        self.covMatrix = np.cov(self.dataset_val[:,0],self.dataset_val[:,1])
        
    def get_prior(self):
        """
        Returns the prior porbability of the class according to the dataset distribution.
        """
        prior = None
        prior = len(self.dataset_val)/len(self.dataset) 
        return prior
    
    def get_instance_likelihood(self, x):
        """
        Returns the likelihood of the instance under the class according to the dataset distribution.
        """
        likelihood = None
        likelihood = multi_normal_pdf(x, self.mean, self.covMatrix)
        return likelihood
    
    def get_instance_posterior(self, x):
        """
        Returns the posterior porbability of the instance under the class according to the dataset distribution.
        * Ignoring p(x)
        """
        posterior = None
        posterior = self.get_instance_likelihood(x)*self.get_prior()
        return posterior

class MaxPrior():
    def __init__(self, ccd0 , ccd1):
        """
        A Maximum prior classifier. 
        This class will hold 2 class distributions, one for class 0 and one for class 1, and will predicit an instance
        by the class that outputs the highest prior probability for the given instance.
    
        Input
            - ccd0 : An object contating the relevant parameters and methods for the distribution of class 0.
            - ccd1 : An object contating the relevant parameters and methods for the distribution of class 1.
        """
        self.ccd0 = ccd0
        self.ccd1 = ccd1
    
    def predict(self, x):
        """
        Predicts the instance class using the 2 distribution objects given in the object constructor.
    
        Input
            - An instance to predict.
        Output
            - 0 if the posterior probability of class 0 is higher and 1 otherwise.
        """
        pred = None
        ccd0_Prior = self.ccd0.get_prior()
        ccd1_Prior = self.ccd1.get_prior()
        if ccd0_Prior > ccd1_Prior:
            pred = 0
        else:
            pred = 1
        return pred

class MaxLikelihood():
    def __init__(self, ccd0 , ccd1):
        """
        A Maximum Likelihood classifier. 
        This class will hold 2 class distributions, one for class 0 and one for class 1, and will predicit an instance
        by the class that outputs the highest likelihood probability for the given instance.
    
        Input
            - ccd0 : An object contating the relevant parameters and methods for the distribution of class 0.
            - ccd1 : An object contating the relevant parameters and methods for the distribution of class 1.
        """
        self.ccd0 = ccd0
        self.ccd1 = ccd1
    
    def predict(self, x):
        """
        Predicts the instance class using the 2 distribution objects given in the object constructor.
    
        Input
            - An instance to predict.
        Output
            - 0 if the posterior probability of class 0 is higher and 1 otherwise.
        """
        pred = None
        ccd0_Likelihood = self.ccd0.get_instance_likelihood(x)
        ccd1_Likelihood = self.ccd1.get_instance_likelihood(x)
        if ccd0_Likelihood > ccd1_Likelihood:
            pred = 0
        else:
            pred = 1
        return pred

EPSILLON = 1e-6 # if a certain value only occurs in the test set, the probability for that value will be EPSILLON.

class DiscreteNBClassDistribution():
    def __init__(self, dataset, class_value):
        """
        A class which computes and encapsulate the relevant probabilites for a discrete naive bayes 
        distribution for a specific class. The probabilites are computed with laplace smoothing.
        
        Input
        - dataset: The dataset as a numpy array.
        - class_value: Compute the relevant parameters only for instances from the given class.
        """
        self.dataset = dataset
        self.dataset_val = dataset[dataset[:,-1]==class_value]
        self.class_value = class_value
    
    def get_prior(self):
        """
        Returns the prior porbability of the class 
        according to the dataset distribution.
        """
        prior = None
        prior = len(self.dataset_val)/len(self.dataset)
        return prior
    
    def get_instance_likelihood(self, x):
        """
        Returns the likelihood of the instance under 
        the class according to the dataset distribution.
        """
        likelihood = None
        likelihood=1
        ni = len(self.dataset_val)
        for i in range(len(self.dataset[0])-1):
            Vj=len(np.unique(self.dataset[self.dataset[:,i]]))
            nj= len(self.dataset_val[self.dataset_val[:,i] == x[i]])
            if(nj==0):
                likelihood*=EPSILLON
            else:
                likelihood*= (nj+1)/(ni+Vj)
        return likelihood
        
    def get_instance_posterior(self, x):
        """
        Returns the posterior porbability of the instance 
        under the class according to the dataset distribution.
        * Ignoring p(x)
        """
        posterior = None
        posterior = self.get_instance_likelihood(x) * self.get_prior()
        return posterior


class MAPClassifier_DNB():
    def __init__(self, ccd0 , ccd1):
        """
        A Maximum a posteriori classifier. 
        This class will hold 2 class distributions, one for class 0 and one for class 1, and will predict an instance
        by the class that outputs the highest posterior probability for the given instance.
    
        Input
            - ccd0 : An object contating the relevant parameters and methods for the distribution of class 0.
            - ccd1 : An object contating the relevant parameters and methods for the distribution of class 1.
        """
        self.ccd0=ccd0
        self.ccd1=ccd1

    def predict(self, x):
        """
        Predicts the instance class using the 2 distribution objects given in the object constructor.
    
        Input
            - An instance to predict.
        Output
            - 0 if the posterior probability of class 0 is higher and 1 otherwise.
        """
        pred = None
        ccd0_Posterior = self.ccd0.get_instance_posterior(x)
        ccd1_Posterior = self.ccd1.get_instance_posterior(x)
        if ccd0_Posterior >= ccd1_Posterior:
            pred = 0
        else: 
            pred =1
        return pred

    def compute_accuracy(self, test_set):
        """
        Compute the accuracy of a given a testset using a MAP classifier object.

        Input
            - test_set: The test_set for which to compute the accuracy (Numpy array).
        Ouput
            - Accuracy = #Correctly Classified / #test_set size
        """
        acc = None
        samplesCorrect = 0
        samplesSize = 0

        for sample in test_set:
            if self.predict(sample) == sample[-1]:
                samplesCorrect += 1
            samplesSize += 1

        acc = samplesCorrect/samplesSize
        return acc


