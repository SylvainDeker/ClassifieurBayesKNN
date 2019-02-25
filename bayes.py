import numpy as np
from typing import Union


class GaussianBayes(object):
    """ Classification by normal law by Bayesian approach
    """
    def __init__(self, priors:np.ndarray,optimisation:bool=False) -> None:
        self.priors = priors    # a priori probabilities of classes
                                # (n_classes,)
        self.optimisation = optimisation
        self.mu = None          #  mean of each feature per class
                                # (n_classes, n_features)
        self.sigma = None       # covariance of each feature per class
                                # (n_classes, n_features, n_features)

    def predict(self, X:np.ndarray) -> np.ndarray:
        """
        X shape = [n_samples, n_features]
        maximum log-likelihood
        """
        n_obs = X.shape[0]
        n_classes = self.mu.shape[0]
        n_features = self.mu.shape[1]

        # initalize the output vector
        y = np.empty(n_obs)
        cte = -np.log(np.math.pi*2)

        for i in range(X.shape[0]): # pour chaque coordonnée
            rates = np.zeros(n_classes)
            likelihood = np.zeros(n_classes)
            prio = np.zeros(n_classes)
            for clss in range(n_classes):  #pourchaque class
                # D'après la formule suivante
                # -(1/2)*ln(2*pi) -(1/2)*log(det(sigma[i])) -(1/2)*((X[i]-mu[class])tr * inverse(sigma[class]) * (X[i]-mu[class])
                a = -np.log(np.linalg.det(self.sigma[clss]))
                b = -np.matmul(np.transpose(X[i] - self.mu[clss]), np.matmul(np.linalg.inv(self.sigma[clss]), X[i] - self.mu[clss]))

                # Calcul de la vraissemblance
                likelihood[clss] = (cte + a + b)/2

                # proba à priori
                if self.priors[clss]==0: # Pour eviter les divisions par 0 dans le log()
                    prio[clss] = -np.math.inf
                else:
                    prio[clss] = np.log(self.priors[clss])

            #Calcul de l'évidence
            evidence = 0
            for clss in range(n_classes):  #pourchaque class
                evidence += np.exp(likelihood[clss])*self.priors[clss]
            #Calcul des poids de chaque classe
            rates = likelihood+prio

            y[i] = np.argmax(rates)
        return y


    def fit(self, X:np.ndarray, y:np.ndarray) -> None:
        """Learning of parameters
        X : shape (n_data, n_features)
        y : shape (n_data)
        """
        # number of random variables and classes
        n_features = X.shape[1]
        n_classes = len(np.unique(y))
        # initialization of parameters
        self.mu = np.zeros((n_classes, n_features))
        self.sigma = np.zeros((n_classes, n_features, n_features))


        if self.optimisation:
            # learning
            for i in range(n_classes): #classe par classe
                #Calcul du centroide (Moyenne des coordonnées par x et y)
                self.mu[i] = np.mean(X[y==i],axis=0)
                # Calcul de la matrice de variance-covariance avec optimisation
                self.sigma[i] = np.cov(X[y==i],rowvar=False)*np.eye(n_features,n_features)
        else:
             # learning
            for i in range(n_classes): #classe par classe
                #Calcul du centroide (Moyenne des coordonnées par x et y)
                self.mu[i] = np.mean(X[y==i],axis=0)
                # Calcul de la matrice de variance-covariance sans optimisation
                self.sigma[i] = np.cov(X[y==i],rowvar=False)







    def score(self, X:np.ndarray, y:np.ndarray) -> float:
        """Compute the precision
        X : shape (n_data, n_features)
        y : shape (n_data)
        """
        return np.sum(y == self.predict(X)) / len(X)
