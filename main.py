import numpy as np
import random

from bayes import GaussianBayes
from utils import load_dataset, plot_scatter_hist


def bayes():
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

    # affichage
                                                            # TODO fix histogramm
    # plot_scatter_hist(train_data,train_labels)
    # input("[enter]")
    # plot_scatter_hist(test_data,test_labels)


    # Instanciation de la classe GaussianB
    g = GaussianBayes((1,)*10)

    # Apprentissage
    g.fit(train_data, train_labels)

    # Score
    score = g.score(test_data, test_labels)
    print("precision : {:.2f}".format(score))

    input("Press any key to exit...")



if __name__ == "__main__":
    bayes()
