import numpy as np
import random
import time

from bayes import GaussianBayes
from utils import load_dataset, plot_scatter_hist
from sklearn import neighbors, datasets
from sklearn.metrics import confusion_matrix

# nombre de voisin définis
n_neighbours = 10

def matriceConf(test_labels, y):
    #Instanciation matrice de confusion pour les données labelisé et leur prediction
    m = confusion_matrix(test_labels,y)
    
    precision = 0
    
    score = np.sum(m)
    
    for i in range(m.shape[0]):
        precision += m[i][i]
    
    print(m)
    print("Precision : {:.2f}".format(precision/score))
    

def bayes(train_data, train_labels, test_data, test_labels):

    # affichage
                                                            # TODO fix histogramm
    # plot_scatter_hist(train_data,train_labels)
    # input("[enter]")
    # plot_scatter_hist(test_data,test_labels)
    
    #Demmarage du timer 
    start = time.time()
    
    # Instanciation de la classe GaussianB
    g = GaussianBayes((1,)*10)

    # Apprentissage
    g.fit(train_data, train_labels)

    # Score
    y = g.predict(test_data)
    
    #Arrêt timer
    stop = time.time()
    
    print("Bayes time : ", stop - start)
    
    print("Matrice Confusion Bayes :")
    matriceConf(test_labels,y)
    
def k_NN(train_data, train_labels, test_data, test_labels):
    #Debut timer
    start = time.time()
    
    #Instanciation de la classe KNeighborsClassifier
    n = neighbors.KNeighborsClassifier(n_neighbours, weights = 'distance')
    
    #Apprentissage
    n.fit(train_data, train_labels)

    #Score
    y = n.predict(test_data)
    
    #Arrêt timer
    stop = time.time()
    
    print("K-NN time : ", stop - start)
    
    print("Matrice Confusion K-NN :")
    matriceConf(test_labels,y)

def main():
#    Récupération des données de tests et d'apprentissage
    data, labels = load_dataset("./data/data2.csv")
    tmp = list(zip(data, labels))
    random.shuffle(tmp)
    data, labels = zip(*tmp)
    data   = np.array(data)
    labels = np.array(labels)


    train_data = data[:int(len(data) * .8)]
    train_labels = labels[:int(len(labels) * .8)]

    test_data  = data[int(len(data) * .8):]
    test_labels  = labels[int(len(labels) * .8):]

#Bayesienne avec gaussienne
    bayes(train_data,train_labels,test_data,test_labels)
    
#K-NN
    k_NN(train_data,train_labels,test_data,test_labels)
if __name__ == "__main__":
    main()
