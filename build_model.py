import os
import time
import argparse
import tensorflow as tf
import numpy as np
from tensorflow.keras.datasets import cifar10
from nasbench.lib import config
from nasbench_keras import ModelSpec, build_keras_model
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.experimental import CosineDecay
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger
from tensorflow.keras.preprocessing.image import ImageDataGenerator

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('epochs', type=int, choices=[4,12,36,108],
        help='train and test best model at this budget point')
    parser.add_argument('--verbose', type=int, default=1)
    args = parser.parse_args()

    epochs = args.epochs
    verbose = args.verbose

    best = np.load('best_file.npy', allow_pickle=True).item()
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    best_enc = best[epochs][0]
    model_config = config.build_config()
    model_spec = ModelSpec(matrix=best_enc['module_adjacency'],
        ops=best_enc['module_operations'],
        data_format=model_config['data_format'])

    print(model_config)

    batch_size = model_config['batch_size']
    inputs = tf.keras.layers.Input(x_train.shape[1:], batch_size)
    net_outputs = build_keras_model(model_spec, inputs,
        model_spec.ops, model_config)
    model = tf.keras.Model(inputs=inputs, outputs=net_outputs)

    num_train_imgs = int(x_train.shape[0]*0.8)
    decay_steps = int(epochs * num_train_imgs / batch_size)
    cos_decay = CosineDecay(model_config['learning_rate'], decay_steps)

    model.compile(
        loss='categorical_crossentropy',
        optimizer=RMSprop(cos_decay),
        metrics=['accuracy'])
    model.summary()
    print(len(model.layers))

    # Prepare model model saving directory.
    save_dir = os.path.join(os.getcwd(), 'saved_models')
    os.makedirs(save_dir, exist_ok=True)
    model_type = "{}epochs".format(epochs)
    model_name = 'cifar10_{0}_model.{{epoch:03d}}.h5'.format(model_type)

    filepath = os.path.join(save_dir, model_name)

    # Prepare callbacks for model saving and for learning rate adjustment.
    checkpoint = ModelCheckpoint(
        filepath=filepath,
        monitor='val_acc',
        verbose=1,
        save_best_only=True)

    logger = CSVLogger(f'cifar10_{model_type}.csv')

    # callbacks = [checkpoint, lr_reducer, lr_scheduler, logger]
    callbacks = [checkpoint, logger]

    print('Using real-time data augmentation.')
    # This will do preprocessing and realtime data augmentation:
    datagen = ImageDataGenerator(
        # set input mean to 0 over the dataset
        featurewise_center=False,
        # set each sample mean to 0
        samplewise_center=False,
        # divide inputs by std of dataset
        featurewise_std_normalization=False,
        # divide each input by its std
        samplewise_std_normalization=False,
        # apply ZCA whitening
        zca_whitening=False,
        # randomly rotate images in the range (deg 0 to 180)
        rotation_range=0,
        # randomly shift images horizontally
        width_shift_range=0.1,
        # randomly shift images vertically
        height_shift_range=0.1,
        # randomly flip images
        horizontal_flip=True,
        # randomly flip images
        vertical_flip=False,
        # validation split
        validation_split=0.2)

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
        workers=4, callbacks=callbacks)

    print("done training in {}s".format(time.time()-t0))

    # Score trained model.
    scores = model.evaluate(x_test, y_test, verbose=verbose)
    print('Test loss:', scores[0])
    print('Test accuracy:', scores[1])
