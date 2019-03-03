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
    # print(m)
    # print("Precision : {:.2f}".format(precision/score))
    return ((m,precision/score))


def bayes(train_data, train_labels, test_data, test_labels):

    # affichage
                                                            # TODO fix histogramm
    # plot_scatter_hist(train_data,train_labels)
    # input("[enter]")
    # plot_scatter_hist(test_data,test_labels)

    #Demmarage du timer
    start = time.time()

    # Instanciation de la classe GaussianB
    g = GaussianBayes((1,)*len(np.unique(train_labels)))

    # Apprentissage
    g.fit(train_data, train_labels)

    # Score
    y = g.predict(test_data)

    #Arrêt timer
    stop = time.time()

    # print("Bayes time : ", stop - start)
    # print("Matrice Confusion Bayes :")
    mconf = matriceConf(test_labels,y)
    return ((stop-start,mconf[0],mconf[1]))

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

    # print("K-NN time : ", stop - start)
    # print("Matrice Confusion K-NN :")
    mconf = matriceConf(test_labels,y)
    return ((stop-start,mconf[0],mconf[1]))

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
    res = bayes(train_data,train_labels,test_data,test_labels)
    print("Bayes time : ", res[0])
    print("Matrice Confusion Bayes :\n",res[1])
    print("Precision : ",res[2])
#K-NN
    res = k_NN(train_data,train_labels,test_data,test_labels)
    print("KNN time : ", res[0])
    print("Matrice Confusion KNN :\n",res[1])
    print("Precision : ",res[2])

def test(file:str):
    data, labels = load_dataset(file)
    tmp = list(zip(data, labels))
    random.shuffle(tmp)
    data, labels = zip(*tmp)
    data   = np.array(data)
    labels = np.array(labels)

    learning_ratio = 0.8

    train_data = data[:int(len(data) * learning_ratio)]
    train_labels = labels[:int(len(labels) * learning_ratio)]

    test_data  = data[int(len(data) * learning_ratio):]
    test_labels  = labels[int(len(labels) * learning_ratio):]

#Bayesienne avec gaussienne
    res = bayes(train_data,train_labels,test_data,test_labels)
    print(res[0],end = '\t') #time
    print(res[2],end = '\t') #precision
#K-NN
    res = k_NN(train_data,train_labels,test_data,test_labels)
    print(res[0],end = '\t') #time
    print(res[2],end = '\t') #precision

def maintest():
    for _ in range(50):
        test("./data/data2.csv")
        test("./data/data3.csv")
        test("./data/data12.csv")
        print("",end='\n')

if __name__ == "__main__":
    # main()
    maintest()
