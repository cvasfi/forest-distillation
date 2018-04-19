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
from mxnet.gluon.data.vision import CIFAR10
from sklearn.cluster import MeanShift
import warnings
import mxnet as mx
import matplotlib.pyplot as plt
import time

class Knowledge_Distiller:

    def __init__(self,dataset_type='MNIST',num_classes=10, forest_size=5):

        self.dataset_type=dataset_type
        self.num_classes = num_classes
        self.forest_size = forest_size
        self.get_data()
        self.reservoirs = list(itertools.repeat({}, self.forest_size))
        self.forest_arr=[]
        self.distill_arr1=[]
        self.distill_arr2=[]

    def get_data(self):
        if(self.dataset_type=="MNIST"):
            mnist = mx.test_utils.get_mnist()
            self.X_train, self.X_test, self.y_train, self.y_test = (mnist['train_data'], mnist['test_data'], mnist['train_label'], mnist['test_label'])
            self.prefix='cnn_models/mnist'
            self.epoch=16
        elif(self.dataset_type=="cifar10"):
            def prepare(data):
                data._get_data()
                x = data._data.asnumpy()
                x = np.swapaxes(x, 2, 3)
                x = np.swapaxes(x, 1, 2)
                y = data._label
                return (x,y)

            train_data=CIFAR10(train=True)
            self.X_train, self.y_train = prepare(train_data)

            val_data=CIFAR10(train=False)
            self.X_test, self.y_test = prepare(val_data)

            self.prefix='cnn_models/resnet20'
            self.epoch=125



        train_shape=self.X_train.shape
        test_shape=self.X_test.shape
        self.X_train_flat= self.X_train.reshape(train_shape[0], train_shape[1]*train_shape[2]*train_shape[3])
        self.X_test_flat = self.X_test.reshape(test_shape[0], test_shape[1]*test_shape[2]*test_shape[3])

    def cnn_predict(self):
        batch_size = 100
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            data_iter = mx.io.NDArrayIter(self.X_train, self.y_train, batch_size)
            #self.val_iter=mx.io.NDArrayIter(self.X_test, self.y_test, batch_size)
            sym, args, auxs = mx.model.load_checkpoint(self.prefix, self.epoch)
            mod = mx.mod.Module(symbol=sym, context=mx.cpu())
            mod.bind(for_training=False, data_shapes=data_iter.provide_data)
            mod.set_params(args, auxs)
            self.mod=mod
        return mod.predict(eval_data=data_iter)

    def train_tree(self):
        #print "--> training the tree"
        self.rfc = RandomForestClassifier(n_jobs=-1, n_estimators=self.forest_size)
        self.rfc.fit(self.X_train_flat, self.y_train)

    def get_leaves_and_labels(self):
        #print "--> getting labels"
        self.cnn_predictions = self.cnn_predict().asnumpy()
        self.training_leaf_indices = self.rfc.apply(self.X_train_flat) #shape: [num_samples,forest_size]

    def fill_reservoirs(self):
        #print "--> filling reservoirs"
        for cnn_prediction, tree_leaves in itertools.izip(self.cnn_predictions, self.training_leaf_indices):
            for idx, leaf in enumerate(tree_leaves):
                if (leaf in self.reservoirs[idx]):
                    self.reservoirs[idx][leaf].append(cnn_prediction)
                else:
                    self.reservoirs[idx][leaf] = [cnn_prediction]


    def update_leaves(self):
        #print "--> Updating leaves"
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

    def predict_distilled(self, forest_preds, forest_leaves, method=1):
        distil_predictions=[]
        if method==1:
            for fidx, fp in enumerate(forest_preds):
                guesses = np.zeros(10)
                for lidx, leaf in enumerate(forest_leaves[fidx]):
                    guesses += self.updated_leaves[lidx][leaf]
                guesses = guesses / self.forest_size
                distil_predictions.append(np.argmax(guesses))

        else:
            for fidx, fp in enumerate(forest_preds):
                guesses = []
                for lidx, leaf in enumerate(forest_leaves[fidx]):
                    guesses.append(np.argmax(self.updated_leaves[lidx][leaf]))
                guess=np.bincount(guesses).argmax()
                distil_predictions.append(guess)

        distil_predictions = np.array(distil_predictions)

        return distil_predictions

    def print_predictions(self):

        #cnn_time=time.time()
        #self.mod.score(self.val_iter, mx.metric.Accuracy())
        #print("time to predict using CNN: {}".format(cnn_time))


        forest_preds  = self.rfc.predict(self.X_test_flat)

        distilled_time = time.time()

        forest_leaves = self.rfc.apply(self.X_test_flat)
        distil_predictions1 = self.predict_distilled(forest_preds,forest_leaves,method=1)
        print("time to predict using distilled forest: {}".format(distilled_time))

        diff1 = distil_predictions1 - self.y_test


        distil_predictions2 = self.predict_distilled(forest_preds,forest_leaves,method=2)
        diff2 = distil_predictions2 - self.y_test

        distilled_acc1 = 1.0 - float(np.count_nonzero(diff1)) / forest_preds.shape[0]
        distilled_acc2 = 1.0 - float(np.count_nonzero(diff2)) / forest_preds.shape[0]

        before_pred= self.rfc.score(self.X_test_flat, self.y_test)
        print( "Before distillation: {}".format(before_pred))
        print( "After distillation: {} (method 1), {} (method2)".format(distilled_acc1, distilled_acc2))
        self.forest_arr.append(before_pred)
        self.distill_arr1.append(distilled_acc1)
        self.distill_arr2.append(distilled_acc2)

    def scan_forest_size(self):
        results=[]
        self.cnn_predictions = self.cnn_predict().asnumpy()
        #xxrange = [1,10,50,100, 250, 400]
        xxrange = [1,5,10,20,30,40,50,100]
        for i in xxrange:
            self.forest_size=i
            self.reservoirs = list(itertools.repeat({}, self.forest_size))
            self.train_tree()
            self.training_leaf_indices = self.rfc.apply(self.X_train_flat)
            self.fill_reservoirs()
            self.update_leaves()
            print("=======================")
            print("forest size: {}".format(self.forest_size))
            self.print_predictions()
            print("========================")

        plt.plot(xxrange,self.forest_arr,label="Original accuracy")
        plt.plot(xxrange,self.distill_arr1,label="Distilled accuracy (average)")
        plt.plot(xxrange,self.distill_arr2,label="Distilled accuracy (voting)")
        plt.legend()
        plt.xlabel("forest size")
        plt.ylabel("accuracy")
        plt.grid(True)
        plt.show()








kd=Knowledge_Distiller()
#kd.distill()
###print "-----------------------"
#kd.print_predictions()
##

kd.scan_forest_size()