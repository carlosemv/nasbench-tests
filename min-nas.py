import os
import time
import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from nasbench_keras import ModelSpec, build_keras_model
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.experimental import CosineDecay
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical

if __name__=='__main__':
    epochs = 36
    verbose = 1
    val_split = 0.2
    num_classes = 10

    batch_size = 256

    matrix = [[0, 1, 0, 1, 1, 1],
        [0, 0, 1, 1, 0, 0],
        [0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 1],
        [0, 0, 0, 0, 0, 0]]
    ops = ['input', 'conv3x3-bn-relu', 'maxpool3x3',
        'conv3x3-bn-relu', 'conv1x1-bn-relu', 'output']

    config = {'data_format': 'channels_last',
        'num_labels': num_classes,
        'module_vertices': 7, 'max_edges': 9,
        'available_ops': ['conv3x3-bn-relu',
            'conv1x1-bn-relu', 'maxpool3x3'],
        'stem_filter_size': 128, 'num_stacks': 3,
        'num_modules_per_stack': 3, 'batch_size': batch_size,
        'train_epochs': epochs, 'learning_rate': 0.1,
        'lr_decay_method': 'COSINE_BY_STEP',
        'momentum': 0.9, 'weight_decay': 0.0001, 'max_attempts': 5,
        'intermediate_evaluations': ['0.5'], 'num_repeats': 3}

    model_spec = ModelSpec(matrix=best, ops=ops,
        data_format=config['data_format'])

    # load data
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    y_train = to_categorical(y_train, num_classes)
    y_test = to_categorical(y_test, num_classes)

    # build model
    inputs = tf.keras.layers.Input(x_train.shape[1:], batch_size)
    net_outputs = build_keras_model(model_spec, inputs,
        ops, model_config)
    model = tf.keras.Model(inputs=inputs, outputs=net_outputs)

    num_train_imgs = int(x_train.shape[0] * (1-val_split))
    decay_steps = int(epochs * num_train_imgs / batch_size)
    cos_decay = CosineDecay(config['learning_rate'], decay_steps)

    model.compile(
        loss='categorical_crossentropy',
        optimizer=RMSprop(cos_decay),
        metrics=['accuracy'])

    logger = CSVLogger(f'cifar10_nasbench{epochs}.csv')

    # This will do preprocessing and realtime data augmentation:
    datagen = ImageDataGenerator(
        featurewise_center=False,
        samplewise_center=False,
        featurewise_std_normalization=False,
        samplewise_std_normalization=False,
        zca_whitening=False,
        rotation_range=0,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
        vertical_flip=False,
        validation_split=val_split)

    # Compute quantities required for featurewise normalization
    # (std, mean, and principal components if ZCA whitening is applied).
    datagen.fit(x_train)

    t0 = time.time()

    # Fit the model on the batches generated by datagen.flow().
    model.fit(
        datagen.flow(x_train, y_train,
            batch_size=batch_size, subset='training'),
        validation_data=datagen.flow(x_train, y_train,
            batch_size=batch_size, subset='validation'),
        epochs=epochs, verbose=verbose,
        workers=4, callbacks=[logger])

    print("done training in {}s".format(time.time()-t0))

    # Score trained model.
    scores = model.evaluate(x_test, y_test,
        verbose=verbose, batch_size=batch_size)
    print('Test loss:', scores[0])
    print('Test accuracy:', scores[1])
