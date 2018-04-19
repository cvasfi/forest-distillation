import mxnet as mx
import logging
from mxnet.gluon.data.vision import CIFAR10
import numpy as np

logging.getLogger().setLevel(logging.DEBUG)


def get_lenet():
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


def get_lr_scheduler(learning_rate, lr_refactor_step, lr_refactor_ratio,
                     num_example, batch_size, begin_epoch):
    iter_refactor = [int(r) for r in lr_refactor_step.split(',') if r.strip()]

    lr = learning_rate
    epoch_size = num_example // batch_size
    for s in iter_refactor:
        if begin_epoch >= s:
            lr *= lr_refactor_ratio
    if lr != learning_rate:
        logging.getLogger().info("Adjusted learning rate to {} for epoch {}".format(lr, begin_epoch))
    steps = [epoch_size * (x - begin_epoch) for x in iter_refactor if x > begin_epoch]
    if not steps:
        return (lr, None)
    lr_scheduler = mx.lr_scheduler.MultiFactorScheduler(step=steps, factor=lr_refactor_ratio)
    return (lr, lr_scheduler)


'''
Reproducing paper:
Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun. "Identity Mappings in Deep Residual Networks"
'''


def residual_unit(data, num_filter, stride, dim_match, name, bottle_neck=True, bn_mom=0.9, workspace=512, memonger=False):
    """Return ResNet Unit symbol for building ResNet
    Parameters
    ----------
    data : str
        Input data
    num_filter : int
        Number of output channels
    bnf : int
        Bottle neck channels factor with regard to num_filter
    stride : tupe
        Stride used in convolution
    dim_match : Boolen
        True means channel number between input and output is the same, otherwise means differ
    name : str
        Base name of the operators
    workspace : int
        Workspace used in convolution operator
    """
    if bottle_neck:
        # the same as https://github.com/facebook/fb.resnet.torch#notes, a bit difference with origin paper
        bn1 = mx.sym.BatchNorm(data=data, fix_gamma=False, eps=2e-5, momentum=bn_mom, name=name + '_bn1')
        act1 = mx.sym.Activation(data=bn1, act_type='relu', name=name + '_relu1')
        conv1 = mx.sym.Convolution(data=act1, num_filter=int(num_filter*0.25), kernel=(1,1), stride=(1,1), pad=(0,0),
                                      no_bias=True, workspace=workspace, name=name + '_conv1')
        bn2 = mx.sym.BatchNorm(data=conv1, fix_gamma=False, eps=2e-5, momentum=bn_mom, name=name + '_bn2')
        act2 = mx.sym.Activation(data=bn2, act_type='relu', name=name + '_relu2')
        conv2 = mx.sym.Convolution(data=act2, num_filter=int(num_filter*0.25), kernel=(3,3), stride=stride, pad=(1,1),
                                      no_bias=True, workspace=workspace, name=name + '_conv2')
        bn3 = mx.sym.BatchNorm(data=conv2, fix_gamma=False, eps=2e-5, momentum=bn_mom, name=name + '_bn3')
        act3 = mx.sym.Activation(data=bn3, act_type='relu', name=name + '_relu3')
        conv3 = mx.sym.Convolution(data=act3, num_filter=num_filter, kernel=(1,1), stride=(1,1), pad=(0,0), no_bias=True,
                                   workspace=workspace, name=name + '_conv3')
        if dim_match:
            shortcut = data
        else:
            shortcut = mx.sym.Convolution(data=act1, num_filter=num_filter, kernel=(1,1), stride=stride, no_bias=True,
                                            workspace=workspace, name=name+'_sc')
        if memonger:
            shortcut._set_attr(mirror_stage='True')
        return conv3 + shortcut
    else:
        bn1 = mx.sym.BatchNorm(data=data, fix_gamma=False, momentum=bn_mom, eps=2e-5, name=name + '_bn1')
        act1 = mx.sym.Activation(data=bn1, act_type='relu', name=name + '_relu1')
        conv1 = mx.sym.Convolution(data=act1, num_filter=num_filter, kernel=(3,3), stride=stride, pad=(1,1),
                                      no_bias=True, workspace=workspace, name=name + '_conv1')
        bn2 = mx.sym.BatchNorm(data=conv1, fix_gamma=False, momentum=bn_mom, eps=2e-5, name=name + '_bn2')
        act2 = mx.sym.Activation(data=bn2, act_type='relu', name=name + '_relu2')
        conv2 = mx.sym.Convolution(data=act2, num_filter=num_filter, kernel=(3,3), stride=(1,1), pad=(1,1),
                                      no_bias=True, workspace=workspace, name=name + '_conv2')
        if dim_match:
            shortcut = data
        else:
            shortcut = mx.sym.Convolution(data=act1, num_filter=num_filter, kernel=(1,1), stride=stride, no_bias=True,
                                            workspace=workspace, name=name+'_sc')
        if memonger:
            shortcut._set_attr(mirror_stage='True')
        return conv2 + shortcut

def resnet(units, num_stage, filter_list, num_class, data_type, bottle_neck=True, bn_mom=0.9, workspace=512, memonger=False):
    """Return ResNet symbol of cifar10 and imagenet
    Parameters
    ----------
    units : list
        Number of units in each stage
    num_stage : int
        Number of stage
    filter_list : list
        Channel size of each stage
    num_class : int
        Ouput size of symbol
    dataset : str
        Dataset type, only cifar10 and imagenet supports
    workspace : int
        Workspace used in convolution operator
    """
    num_unit = len(units)
    assert(num_unit == num_stage)
    data = mx.sym.Variable(name='data')
    data = mx.sym.BatchNorm(data=data, fix_gamma=True, eps=2e-5, momentum=bn_mom, name='bn_data')
    if data_type == 'cifar10':
        body = mx.sym.Convolution(data=data, num_filter=filter_list[0], kernel=(3, 3), stride=(1,1), pad=(1, 1),
                                  no_bias=True, name="conv0", workspace=workspace)
    elif data_type == 'imagenet':
        body = mx.sym.Convolution(data=data, num_filter=filter_list[0], kernel=(7, 7), stride=(2,2), pad=(3, 3),
                                  no_bias=True, name="conv0", workspace=workspace)
        body = mx.sym.BatchNorm(data=body, fix_gamma=False, eps=2e-5, momentum=bn_mom, name='bn0')
        body = mx.sym.Activation(data=body, act_type='relu', name='relu0')
        body = mx.symbol.Pooling(data=body, kernel=(3, 3), stride=(2,2), pad=(1,1), pool_type='max')
    else:
         raise ValueError("do not support {} yet".format(data_type))
    for i in range(num_stage):
        body = residual_unit(body, filter_list[i+1], (1 if i==0 else 2, 1 if i==0 else 2), False,
                             name='stage%d_unit%d' % (i + 1, 1), bottle_neck=bottle_neck, workspace=workspace,
                             memonger=memonger)
        for j in range(units[i]-1):
            body = residual_unit(body, filter_list[i+1], (1,1), True, name='stage%d_unit%d' % (i + 1, j + 2),
                                 bottle_neck=bottle_neck, workspace=workspace, memonger=memonger)
    bn1 = mx.sym.BatchNorm(data=body, fix_gamma=False, eps=2e-5, momentum=bn_mom, name='bn1')
    relu1 = mx.sym.Activation(data=bn1, act_type='relu', name='relu1')
    # Although kernel is not used here when global_pool=True, we should put one
    pool1 = mx.symbol.Pooling(data=relu1, global_pool=True, kernel=(7, 7), pool_type='avg', name='pool1')
    flat = mx.symbol.Flatten(data=pool1)
    fc1 = mx.symbol.FullyConnected(data=flat, num_hidden=num_class, name='fc1')
    return mx.symbol.SoftmaxOutput(data=fc1, name='softmax')





def train(prefix, epoch=10, dataset="MNIST", resume=True, depth=20, ctx=mx.cpu()):
    batch_size = 100


    args = None
    auxs = None


    if(dataset=="MNIST"):
        sym = get_lenet()

        mnist = mx.test_utils.get_mnist()

        train_iter = mx.io.NDArrayIter(mnist['train_data'], mnist['train_label'], batch_size, shuffle=True)
        val_iter = mx.io.NDArrayIter(mnist['test_data'], mnist['test_label'], batch_size)
    else:
        #def prepare(data):
        #    data._get_data()
        #    x = data._data.asnumpy()
        #    x = np.swapaxes(x, 2, 3)
        #    x = np.swapaxes(x, 1, 2)
        #    y = data._label
        #    return (x, y)
#
        #train_data = CIFAR10(train=True)
        #X_train, y_train = prepare(train_data)
#
        #val_data = CIFAR10(train=False)
        #X_test, y_test = prepare(val_data)

        #train_iter = mx.io.NDArrayIter(X_train, y_train, batch_size, shuffle=True)
        #val_iter = mx.io.NDArrayIter(X_test, y_test, batch_size)

        auglist=mx.image.CreateAugmenter((3,32,32), resize=0, rand_mirror=True, hue=0.1, brightness=0.2, saturation=0.1, rand_crop=True)
        train_iter=mx.image.ImageIter(batch_size=batch_size,data_shape=(3,32,32),path_imgrec="dataset/cifar10_train.rec",aug_list=auglist)
        val_iter=mx.image.ImageIter(batch_size=batch_size,data_shape=(3,32,32),path_imgrec="dataset/cifar10_val.rec")
        depth=depth
        per_unit = [(depth - 2) / 6]
        filter_list = [16, 16, 32, 64]
        bottle_neck = False
        units = per_unit * 3
        sym = resnet(units=units, num_stage=3, filter_list=filter_list, num_class=10,
                        data_type="cifar10", bottle_neck=bottle_neck)

    if resume:
        sym, args, auxs = mx.model.load_checkpoint(prefix, epoch)

    mod = mx.mod.Module(symbol=sym, context=ctx)

    learning_rate, lr_scheduler = get_lr_scheduler(0.001, '80, 160',
                                                   0.1, 48638, 128, epoch)

    optimizer_params = {'learning_rate': learning_rate,
                       'momentum': 0.9,
                       'wd': 0.0005,
                        'lr_scheduler':lr_scheduler,
                       'clip_gradient': None,
                       'rescale_grad': 1.0}

    mod.fit(train_iter,
                    eval_data=val_iter,
                    optimizer='sgd',
                    optimizer_params=optimizer_params,
                    eval_metric='acc',
                    batch_end_callback = mx.callback.Speedometer(batch_size, 100),
                    epoch_end_callback=mx.callback.do_checkpoint(prefix),
                    arg_params=args,
                    aux_params=auxs,
                    begin_epoch=epoch,
                    num_epoch=300)



def predict(prefix,epoch):
    #val_iter = mx.io.ImageRecordIter (
    #    path_imgrec="dataset/cifar10_val.rec", data_name="data", label_name="softmax_label",
    #batch_size=100, data_shape=(3, 32, 32))

    def prepare(data):
        data._get_data()
        x = data._data.asnumpy()
        x = np.swapaxes(x, 2, 3)
        x = np.swapaxes(x, 1, 2)
        y = data._label
        return (x, y)

    val_data = CIFAR10(train=False)
    X_test, y_test = prepare(val_data)
    val_iter = mx.io.NDArrayIter(X_test, y_test, 100)


    sym, args, auxs = mx.model.load_checkpoint(prefix, epoch)
    mod=mx.mod.Module(symbol=sym, context=mx.gpu())
    mod.bind(for_training=False, data_shapes=val_iter.provide_data,label_shapes=val_iter.provide_label)
    mod.set_params(args, auxs)
    result=mod.score(val_iter,mx.metric.Accuracy())
    print result


train("cnn_models/resnet56/resnet56",dataset="cifar10",ctx=mx.gpu(), resume=False, depth=56)
#predict("cnn_models/resnet20/resnet20", 125)