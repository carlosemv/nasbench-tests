from nasbench import api
from nasbench.lib import base_ops, config
from nasbench.lib.model_builder import build_module
import tensorflow as tf
import numpy as np
from tensorflow.keras.datasets import cifar10

def build_model(features, spec, config, mode=tf.estimator.ModeKeys.TRAIN):
    """Builds the model from the input features."""
    is_training = (mode == tf.estimator.ModeKeys.TRAIN)

    # Store auxiliary activations increasing in depth of network. First
    # activation occurs immediately after the stem and the others immediately
    # follow each stack.
    aux_activations = []

    # Initial stem convolution
    with tf.variable_scope('stem'):
        net = base_ops.conv_bn_relu(
            features, 3, config['stem_filter_size'],
            is_training, config['data_format'])
        aux_activations.append(net)

    for stack_num in range(config['num_stacks']):
        channels = net.get_shape()[channel_axis].value

        # Downsample at start (except first)
        if stack_num > 0:
            net = tf.layers.max_pooling2d(
                inputs=net,
                pool_size=(2, 2),
                strides=(2, 2),
                padding='same',
                data_format=config['data_format'])

            # Double output channels each time we downsample
            channels *= 2

        with tf.variable_scope('stack{}'.format(stack_num)):
            for module_num in range(config['num_modules_per_stack']):
                with tf.variable_scope('module{}'.format(module_num)):
                    net = build_module(
                        spec,
                        inputs=net,
                        channels=channels,
                        is_training=is_training)
            aux_activations.append(net)

    # Global average pool
    if config['data_format'] == 'channels_last':
        net = tf.reduce_mean(net, [1, 2])
    elif config['data_format'] == 'channels_first':
        net = tf.reduce_mean(net, [2, 3])
    else:
        raise ValueError('invalid data_format')

    # Fully-connected layer to labels
    logits = tf.layers.dense(
        inputs=net,
        units=config['num_labels'])
    return logits

if __name__=='__main__':
    best = np.load('best_file.npy', allow_pickle=True).item()
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    best_36 = best[36][0]
    model_spec = api.ModelSpec(matrix=best_36['module_adjacency'],
        ops=best_36['module_operations'])

    # inputs = tf.layers.Input(shape=x_train.shape[1:])
    inputs = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    model = build_model(x_train, model_spec, config.build_config())
    print(type(model))
    model.summary()