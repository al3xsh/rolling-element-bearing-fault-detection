"""
visualisation_utils.py

make pretty graphs to show classifier performance

(most of these are based on the really useful examples from the 
scikit learn user guides!)

author:     alex shenfield
date:       27/04/2018
"""

# numpy is needed for everything :)
import numpy as np
import matplotlib.pyplot as plt

# utilities for managing the data
import itertools

# data analysis functions from scikit learn
from sklearn.metrics import confusion_matrix

#
def plot_confusion_matrix(y_true, y_pred):
    
    cm = confusion_matrix(y_true, y_pred)
    classes = np.unique(y_true)
    plot_cm(cm, classes=classes, title=None)

# define a function for plotting a confusion matrix
def plot_cm(cm, 
            classes,
            normalize=False,
            title='Confusion matrix',
            cmap=plt.cm.Blues):

    # should we normalise the confusion matrix?
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print('Confusion matrix, with normalization')
    else:
        print('Confusion matrix, without normalization')

    # display in command windows
    print(cm)

    # create a plot for the confusion matrix
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)    
    
    # if we want a title displayed
    if title:        
        plt.title(title)

    fmt = '.3f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    #plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')