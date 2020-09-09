import tensorflow as tf
import numpy as np
from tensorflow.keras.datasets import cifar10
from nasbench.lib import config
from nasbench_keras import ModelSpec, build_keras_model

if __name__=='__main__':
    best = np.load('best_file.npy', allow_pickle=True).item()
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    best_36 = best[36][0]
    model_config = config.build_config()
    model_spec = ModelSpec(matrix=best_36['module_adjacency'],
        ops=best_36['module_operations'],
        data_format=model_config['data_format'])

    print(config)

    # inputs = tf.keras.layers.Input(x_train.shape[1:], 1)
    # net_outputs = build_keras_model(model_spec, inputs,
    #     model_spec.ops, model_config)
    # net = tf.keras.Model(inputs=inputs, outputs=net_outputs)
    # net.summary()