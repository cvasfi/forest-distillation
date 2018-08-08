import numpy as np
from sklearn.ensemble import RandomForestClassifier
import itertools
from mxnet.gluon.data.vision import CIFAR10
import warnings
import mxnet as mx
import matplotlib.pyplot as plt
import sklearn.preprocessing

class Knowledge_Distiller:

    def __init__(self,dataset_type='MNIST',num_classes=10, forest_size=5):

        self.dataset_type=dataset_type
        self.num_classes = num_classes
        self.forest_size = forest_size
        self.get_data()
        self.reservoirs = [{} for i in range(self.forest_size)]
        self.forest_arr=[]
        self.distill_arr_avg=[]
        self.distill_arr_voting=[]


    def get_data(self):
        if(self.dataset_type=="MNIST"):
            mnist = mx.test_utils.get_mnist()
            self.X_train, self.X_test, self.y_train, self.y_test = (mnist['train_data'], mnist['test_data'], mnist['train_label'], mnist['test_label'])
            self.X_CNN, self.y_CNN = self.X_train, self.y_train
            self.prefix='cnn_models/mnist'
            self.epoch=16
        elif(self.dataset_type=="cifar10"):
            def prepare(data,split=False):
                if(split):
                    data._get_data()
                    x = data._data.asnumpy()
                    x = np.swapaxes(x, 2, 3)
                    x = np.swapaxes(x, 1, 2)
                    x1 = x[x.shape[0]/2:]
                    x2 = x[0:x.shape[0] / 2]
                    y = data._label
                    y1 = y[y.shape[0]/2:]
                    y2 = y[0:y.shape[0]/2]
                    return x1, y1, x2, y2
                else:
                    data._get_data()
                    x = data._data.asnumpy()
                    x = np.swapaxes(x, 2, 3)
                    x = np.swapaxes(x, 1, 2)
                    y = data._label
                    return (x,y)

            train_data=CIFAR10(train=True)
            split=False         #TODO: remove this feature maybe
            if split:
                self.X_train, self.y_train, self.X_CNN, self.y_CNN = prepare(train_data,split=True)
            else:
                self.X_train, self.y_train = prepare(train_data)
                self.X_CNN, self.y_CNN = self.X_train, self.y_train

            val_data=CIFAR10(train=False)
            self.X_test, self.y_test = prepare(val_data)

            self.prefix='cnn_models/resnet20'
            self.epoch=125



        train_shape=self.X_train.shape
        CNN_shape=self.X_CNN.shape
        test_shape=self.X_test.shape

        self.X_train_flat= self.X_train.reshape(train_shape[0], train_shape[1]*train_shape[2]*train_shape[3])
        self.X_CNN_flat= self.X_CNN.reshape(CNN_shape[0], CNN_shape[1]*CNN_shape[2]*CNN_shape[3])
        self.X_test_flat = self.X_test.reshape(test_shape[0], test_shape[1]*test_shape[2]*test_shape[3])

    def cnn_predict(self):
        batch_size = 100
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            data_iter = mx.io.NDArrayIter(self.X_CNN, self.y_CNN, batch_size)
            sym, args, auxs = mx.model.load_checkpoint(self.prefix, self.epoch)
            mod = mx.mod.Module(symbol=sym, context=mx.gpu())
            mod.bind(for_training=False, data_shapes=data_iter.provide_data)
            mod.set_params(args, auxs)
            self.mod=mod
        return mod.predict(eval_data=data_iter)

    def train_tree(self):
        self.rfc = RandomForestClassifier(n_jobs=-1, n_estimators=self.forest_size)
        self.rfc.fit(self.X_train_flat, self.y_train)

    def get_leaves_and_labels(self):
        self.cnn_predictions = self.cnn_predict().asnumpy()
        self.training_leaf_indices = self.rfc.apply(self.X_CNN_flat) #shape: [num_samples,forest_size]

    def fill_reservoirs(self):
        self.reservoirs = [{} for i in range(self.forest_size)]


        for cnn_prediction, tree_leaves in itertools.izip(self.cnn_predictions, self.training_leaf_indices):
            for idx, leaf in enumerate(tree_leaves):
                if (leaf in self.reservoirs[idx]):
                    self.reservoirs[idx][leaf].append(cnn_prediction)
                else:
                    self.reservoirs[idx][leaf] = [cnn_prediction]


    def update_leaves(self):
        self.updated_leaves = [{} for i in range(self.forest_size)]
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

        def get_onehot(a):
            a = [a]
            label_binarizer = sklearn.preprocessing.LabelBinarizer()
            label_binarizer.fit(range(10))
            b = label_binarizer.transform(a)
            return b[0]



        distil_predictions=[]
        if method==1:
            for fidx, fp in enumerate(forest_preds):
                guesses = np.zeros(10)
                for lidx, leaf in enumerate(forest_leaves[fidx]):
                    if leaf in self.updated_leaves[lidx]:
                        guesses += self.updated_leaves[lidx][leaf]
                    else:
                        guesses += get_onehot(forest_preds[fidx])
                guesses = guesses / self.forest_size
                distil_predictions.append(np.argmax(guesses))

        else:
            for fidx, fp in enumerate(forest_preds):
                guesses = []
                for lidx, leaf in enumerate(forest_leaves[fidx]):
                    if leaf in self.updated_leaves[lidx]:
                        guesses.append(np.argmax(self.updated_leaves[lidx][leaf]))
                    else:
                        guesses.append(forest_preds[fidx])
                guess=np.bincount(guesses).argmax()
                distil_predictions.append(guess)

        distil_predictions = np.array(distil_predictions)

        return distil_predictions

    def print_predictions(self):
        res_sizes = [len(res) for leaf, res in self.reservoirs[0].iteritems()]
        print("mean: {}, std: {}, total: {}".format(np.mean(res_sizes), np.std(res_sizes), np.sum(res_sizes)))

        forest_preds  = self.rfc.predict(self.X_test_flat)
        forest_leaves = self.rfc.apply(self.X_test_flat)

        diff_avg = self.predict_distilled(forest_preds,forest_leaves,method=1) - self.y_test
        diff_voting = self.predict_distilled(forest_preds,forest_leaves,method=2) - self.y_test

        distilled_avg = 1.0 - float(np.count_nonzero(diff_avg)) / forest_preds.shape[0]
        distilled_voting = 1.0 - float(np.count_nonzero(diff_voting)) / forest_preds.shape[0]

        before_pred= self.rfc.score(self.X_test_flat, self.y_test)
        print( "Before distillation: {}".format(before_pred))
        print( "After distillation: {} (method 1), {} (method2)".format(distilled_avg, distilled_voting))
        self.forest_arr.append(before_pred)
        self.distill_arr_avg.append(distilled_avg)
        self.distill_arr_voting.append(distilled_voting)

    def scan_forest_size(self):
        self.cnn_predictions = self.cnn_predict().asnumpy()
        xxrange = [1,10,50,100, 200, 300, 400]
        for i in xxrange:
            self.forest_size=i
            self.reservoirs = [{} for i in range(self.forest_size)]
            self.train_tree()
            self.training_leaf_indices = self.rfc.apply(self.X_CNN_flat)
            self.fill_reservoirs()
            self.update_leaves()
            print("=======================")
            print("num leaves: {}".format(len(self.reservoirs[0])))
            res_sizes=[len(res) for leaf,res in self.reservoirs[0].iteritems()]
            print("mean: {}, std: {}, total: {}".format(np.mean(res_sizes),np.std(res_sizes),np.sum(res_sizes)))
            print("forest size: {}".format(self.forest_size))
            self.print_predictions()
            print("========================")

        plt.plot(xxrange,self.forest_arr,label="Original accuracy")
        plt.plot(xxrange,self.distill_arr_avg,label="Distilled accuracy (average)")
        plt.plot(xxrange,self.distill_arr_voting,label="Distilled accuracy (voting)")
        plt.legend()
        plt.xlabel("forest size")
        plt.ylabel("accuracy")
        plt.grid(True)
        plt.show()






if __name__ == "__main__":
    kd=Knowledge_Distiller()
    kd.scan_forest_size()