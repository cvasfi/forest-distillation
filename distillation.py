from sklearn.datasets import fetch_mldata
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn import tree
import itertools

from sklearn.cluster import MeanShift
import warnings
import mxnet as mx

class Knowledge_Distiller:

    def __init__(self):
        mnist = mx.test_utils.get_mnist()
        self.X_train, self.X_test, self.y_train, self.y_test = (
        mnist['train_data'], mnist['test_data'], mnist['train_label'], mnist['test_label'])

        self.X_train_flat= self.X_train.reshape(self.X_train.shape[0], 784) #TODO: Generalize data shape
        self.X_test_flat = self.X_test.reshape(self.X_test.shape[0], 784)
        self.forest_size = 15
        self.num_classes = 10

        self.reservoirs = list(itertools.repeat({}, self.forest_size))

    def cnn_predict(self):
        batch_size = 100
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            data_iter = mx.io.NDArrayIter(self.X_train, self.y_train, batch_size)

            sym, args, auxs = mx.model.load_checkpoint('mnist', 16)
            mod = mx.mod.Module(symbol=sym, context=mx.cpu())
            mod.bind(for_training=False, data_shapes=data_iter.provide_data)
            mod.set_params(args, auxs)
        return mod.predict(eval_data=data_iter)

    def train_tree(self):
        print "--> training the tree"
        self.rfc = RandomForestClassifier(n_jobs=-1, n_estimators=self.forest_size, verbose=1)
        self.rfc.fit(self.X_train_flat, self.y_train)

    def get_leaves_and_labels(self):
        print "--> getting labels"
        self.cnn_predictions = self.cnn_predict().asnumpy()
        self.training_leaf_indices = self.rfc.apply(self.X_train_flat) #shape: [num_samples,forest_size]

    def fill_reservoirs(self):
        print "--> filling reservoirs"
        for cnn_prediction, tree_leaves in itertools.izip(self.cnn_predictions, self.training_leaf_indices):
            for idx, leaf in enumerate(tree_leaves):
                if (leaf in self.reservoirs[idx]):
                    self.reservoirs[idx][leaf].append(cnn_prediction)
                else:
                    self.reservoirs[idx][leaf] = [cnn_prediction]


    def update_leaves(self):
        print "--> Updating leaves"
        self.updated_leaves = list(itertools.repeat({}, self.forest_size))
        for idx, reservoir in enumerate(self.reservoirs):
            for leaf, p_list in reservoir.iteritems():
                guess = np.zeros((self.num_classes))
                for prob in p_list:
                    guess += prob
                guess = guess / len(p_list)
                self.updated_leaves[idx][leaf] = guess


    def distill(self):
        print "Starting distillation:"
        self.train_tree()
        self.get_leaves_and_labels()
        self.fill_reservoirs()
        self.update_leaves()
        print "distillation done"

    def print_predictions(self):

        forest_preds  = self.rfc.predict(self.X_test_flat)
        forest_leaves = self.rfc.apply(self.X_test_flat)

        distil_predictions = []

        for fidx, fp in enumerate(forest_preds):
            guesses = np.zeros(10)
            for lidx, leaf in enumerate(forest_leaves[fidx]):
                guesses += self.updated_leaves[lidx][leaf]
            guesses = guesses / self.forest_size
            distil_predictions.append(np.argmax(guesses))

        distil_predictions = np.array(distil_predictions)
        diff = distil_predictions - forest_preds

        distilled_acc = 1.0 - float(np.count_nonzero(diff)) / forest_preds.shape[0]

        print( "Before distillation: {}".format( self.rfc.score(self.X_test_flat, self.y_test)))
        print( "After distillation: {}".format(distilled_acc))





kd=Knowledge_Distiller()

kd.distill()
print "-----------------------"
kd.print_predictions()

