import mxnet as mx
import logging


logging.getLogger().setLevel(logging.DEBUG)
mnist = mx.test_utils.get_mnist()
batch_size = 100

train_iter = mx.io.NDArrayIter(mnist['train_data'], mnist['train_label'], batch_size, shuffle=True)
val_iter = mx.io.NDArrayIter(mnist['test_data'], mnist['test_label'], batch_size)


def get_cnn():
    data = mx.sym.var('data')
    # first conv layer
    conv1 = mx.sym.Convolution(data=data, kernel=(5, 5), num_filter=20)
    tanh1 = mx.sym.Activation(data=conv1, act_type="tanh")
    pool1 = mx.sym.Pooling(data=tanh1, pool_type="max", kernel=(2, 2), stride=(2, 2))
    # second conv layer
    conv2 = mx.sym.Convolution(data=pool1, kernel=(5, 5), num_filter=50)
    tanh2 = mx.sym.Activation(data=conv2, act_type="tanh")
    pool2 = mx.sym.Pooling(data=tanh2, pool_type="max", kernel=(2, 2), stride=(2, 2))
    # first fullc layer
    flatten = mx.sym.flatten(data=pool2)
    fc1 = mx.symbol.FullyConnected(data=flatten, num_hidden=500)
    tanh3 = mx.sym.Activation(data=fc1, act_type="tanh")
    # second fullc
    fc2 = mx.sym.FullyConnected(data=tanh3, num_hidden=10)

    # softmax loss
    return mx.sym.SoftmaxOutput(data=fc2, name='softmax')


def train():

    resume=True

    sym=get_cnn()
    args = None
    auxs = None
    epoch=0

    if resume:
        sym, args, auxs = mx.model.load_checkpoint('mnist', 10)
        epoch=10

    lenet_model = mx.mod.Module(symbol=sym, context=mx.cpu())




    lenet_model.fit(train_iter,
                    eval_data=val_iter,
                    optimizer='sgd',
                    optimizer_params={'learning_rate':0.1},
                    eval_metric='acc',
                    batch_end_callback = mx.callback.Speedometer(batch_size, 100),
                    epoch_end_callback=mx.callback.do_checkpoint("mnist"),
                    arg_params=args,
                    aux_params=auxs,
                    begin_epoch=epoch,
                    num_epoch=20)

def predict():
    print mnist['train_data'].shape
    print mnist['train_label'].shape

    sym, args, auxs = mx.model.load_checkpoint('mnist', 16)
    mod=mx.mod.Module(symbol=sym, context=mx.cpu())
    mod.bind(for_training=False, data_shapes=train_iter.provide_data)
    mod.set_params(args, auxs)
    result=mod.predict(eval_data=train_iter)

    print result.shape


predict()