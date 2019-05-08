# TODO: this code is an absolute mess

import os
import glob
import pickle
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import math
import pandas as pd
import numpy as np
# import sklearn
from sklearn import metrics
import seaborn as sns
import itertools
import keras
from itertools import product
from vis.utils import utils
from vis.visualization import visualize_activation, visualize_cam, overlay
from matplotlib import cm
import warnings
from skimage.transform import rescale

from keras import activations
from vis.visualization import get_num_filters
from tqdm import tqdm_notebook, tqdm

from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

import matplotlib

from keras import backend as K
tf_conf = K.tf.ConfigProto( 
    intra_op_parallelism_threads=7, 
    inter_op_parallelism_threads=7
)
K.set_session(K.tf.Session(config=tf_conf))

PLANTVILLAGE_DATASET_PATH = '/Users/matthijm/Downloads/plant-village-slim/'
PLANTVILLAGE_CLASS_NAMES = [
    'Pepper__bell___Bacterial_spot',
    'Pepper__bell___healthy',
    'Potato___Early_blight',
    'Potato___Late_blight',
    'Potato___healthy',
    'Tomato_Bacterial_spot',
    'Tomato_Early_blight',
    'Tomato_Late_blight',
    'Tomato_Leaf_Mold',
    'Tomato_Septoria_leaf_spot',
    'Tomato_Spider_mites_Two_spotted_spider_mite',
    'Tomato__Target_Spot',
    'Tomato__Tomato_YellowLeaf__Curl_Virus',
    'Tomato__Tomato_mosaic_virus',
    'Tomato_healthy'
]


def plot_history(history):
    sns.set(style='white')
    plt.rc('legend', fontsize=16)
    plt.rc('axes', titlesize=18, labelsize=16)
    plt.rc('xtick', labelsize=16)
    plt.rc('ytick', labelsize=16)
    
    fig, axes = plt.subplots(figsize=(10, 12), ncols=1, nrows=2)

    # Accuracy

    df = pd.DataFrame({
        'acc': history.history['acc'],
        'epoch': history.epoch,
    })
    if 'val_acc' in history.history:
        df['val_acc'] = history.history['val_acc']

    df = df.melt(id_vars=['epoch'])

    sns.lineplot(data=df, x='epoch', y='value', hue='variable', ax=axes[0])
    axes[0].set_title('Accuracy')
    axes[0].set_xlabel('')

    # Loss

    df = pd.DataFrame({
        'loss': history.history['loss'],
        'epoch': history.epoch,
    })
    if 'val_loss' in history.history:
        df['val_loss'] = history.history['val_loss']

    df = df.melt(id_vars=['epoch'], var_name='metric')

    sns.lineplot(data=df, x='epoch', y='value', hue='metric', ax=axes[1])
    
    return fig, axes



def plot_activations(model, layer_name, num):
    layer_idx = utils.find_layer_idx(model, layer_name)
    
    filters = np.arange(get_num_filters(model.layers[layer_idx]))[:num]

    # Generate input image for each filter.
    vis_images = []
    for idx in tqdm_notebook(filters, 'Generating images'):
        img = visualize_activation(model, layer_idx, filter_indices=idx, input_range=(0., 1.))
        vis_images.append(img)

    # Generate stitched image palette with 8 cols.
    stitched = utils.stitch_images(vis_images, cols=8)
    plt.figure(figsize=(16, 16))
    plt.axis('off')
    plt.imshow(stitched)
    plt.title(layer_name)

    

def _generate_plantvillage_tomato_twoclass_dataset(path, x_output_path, y_output_path):
    INPUT_SHAPE = (256, 256)

    gen = keras.preprocessing.image.ImageDataGenerator(rescale=1 / 255)

    dataset = gen.flow_from_directory(
        path,
        target_size=INPUT_SHAPE,
        class_mode="categorical",
        color_mode="rgb",
        batch_size=1,
        shuffle=True
    )

    X = np.zeros([len(dataset)] + list(dataset.image_shape), dtype=np.float32)
    Y = np.zeros([len(dataset), dataset.num_classes], dtype=np.float32)

    for i in range(len(dataset)):
        image, label = next(dataset)
        X[i] = image
        Y[i] = label

    np.save(x_output_path, X)
    np.save(y_output_path, Y)
        
    return X, Y


def generate_plantvillage_small_dataset():
    INPUT_SHAPE = (64, 64)

    gen = keras.preprocessing.image.ImageDataGenerator(
        validation_split = 0.2,    # we set aside 20% of our data for validation during training
        rescale=1 / 255            # this will rescale the images from the range 0-255 to the range 0-1
    )

    data_train = gen.flow_from_directory(
        'data/plant-village/',     # the data is located here, a subdirectory per class
        target_size=INPUT_SHAPE,   # Keras will rescale all input data to this size
        class_mode="categorical",  # the labels will be categorical, i.e. one-hot-encoded
        color_mode="rgb",          # these are colour images
        batch_size=1,     # the batch size
        subset='training',         # we need the training data
        shuffle=True               # we want to shuffle the input data to improve performance
    )

    data_val = gen.flow_from_directory(
        'data/plant-village/',     # the data is located here, a subdirectory per class
        target_size=INPUT_SHAPE,   # Keras will rescale all input data to this size
        class_mode="categorical",  # the labels will be categorical, i.e. one-hot-encoded
        color_mode="rgb",          # these are colour images
        batch_size=1,     # the batch size
        subset='validation'        # we need the training data
    )

    # DEBUG

    X_train = np.zeros([len(data_train)] + list(data_train.image_shape), dtype=np.float32)
    Y_train = np.zeros([len(data_train), data_train.num_classes], dtype=np.float32)

    for i in tqdm_notebook(list(range(len(data_train)))):
        image, label = next(data_train)
        X_train[i] = image
        Y_train[i] = label
        
    np.save('data/plant-village-small/X_train.npy', X_train)
    np.save('data/plant-village-small/Y_train.npy', Y_train)

    X_val = np.zeros([len(data_val)] + list(data_val.image_shape), dtype=np.float32)
    Y_val = np.zeros([len(data_val), data_val.num_classes], dtype=np.float32)

    for i in tqdm_notebook(list(range(len(data_val)))):
        image, label = next(data_val)
        X_val[i] = image
        Y_val[i] = label

    np.save('data/plant-village-small/X_val.npy', X_val)
    np.save('data/plant-village-small/Y_val.npy', Y_val)
    
    labels = list(data_train.class_indices.keys())
    with open('data/plant-village-small/labels.pkl', 'wb') as stream:
        pickle.dump(labels, stream)

        
def generate_plantvillage_large_dataset():
    INPUT_SHAPE = (128, 128)

    gen = keras.preprocessing.image.ImageDataGenerator(
        validation_split = 0.2,    # we set aside 20% of our data for validation during training
        rescale=1 / 255            # this will rescale the images from the range 0-255 to the range 0-1
    )

    data_train = gen.flow_from_directory(
        'data/plant-village/',     # the data is located here, a subdirectory per class
        target_size=INPUT_SHAPE,   # Keras will rescale all input data to this size
        class_mode="categorical",  # the labels will be categorical, i.e. one-hot-encoded
        color_mode="rgb",          # these are colour images
        batch_size=1,     # the batch size
        subset='training',         # we need the training data
        shuffle=True               # we want to shuffle the input data to improve performance
    )

    data_val = gen.flow_from_directory(
        'data/plant-village/',     # the data is located here, a subdirectory per class
        target_size=INPUT_SHAPE,   # Keras will rescale all input data to this size
        class_mode="categorical",  # the labels will be categorical, i.e. one-hot-encoded
        color_mode="rgb",          # these are colour images
        batch_size=1,     # the batch size
        subset='validation'        # we need the training data
    )

    # DEBUG

    X_train = np.zeros([len(data_train)] + list(data_train.image_shape), dtype=np.float32)
    Y_train = np.zeros([len(data_train), data_train.num_classes], dtype=np.float32)

    for i in tqdm_notebook(list(range(len(data_train)))):
        image, label = next(data_train)
        X_train[i] = image
        Y_train[i] = label
        
    np.save('data/plant-village-large/X_train.npy', X_train)
    np.save('data/plant-village-large/Y_train.npy', Y_train)

    X_val = np.zeros([len(data_val)] + list(data_val.image_shape), dtype=np.float32)
    Y_val = np.zeros([len(data_val), data_val.num_classes], dtype=np.float32)

    for i in tqdm_notebook(list(range(len(data_val)))):
        image, label = next(data_val)
        X_val[i] = image
        Y_val[i] = label

    np.save('data/plant-village-large/X_val.npy', X_val)
    np.save('data/plant-village-large/Y_val.npy', Y_val)
    
    labels = list(data_train.class_indices.keys())
    with open('data/plant-village-large/labels.pkl', 'wb') as stream:
        pickle.dump(labels, stream)


def dataset_plant_village_small():
    ROOT = 'data/plant-village-small'
    return (
        (
            np.load(os.path.join(ROOT, 'X_train.npy')),
            np.load(os.path.join(ROOT, 'Y_train.npy')),
        ),
        (
            np.load(os.path.join(ROOT, 'X_val.npy')),
            np.load(os.path.join(ROOT, 'Y_val.npy')),
        ),
        pickle.load(open(os.path.join(ROOT, 'labels.pkl'), 'rb'))
    )


def dataset_plant_village_large():
    ROOT = 'data/plant-village-large'
    return (
        (
            np.load(os.path.join(ROOT, 'X_train.npy')),
            np.load(os.path.join(ROOT, 'Y_train.npy')),
        ),
        (
            np.load(os.path.join(ROOT, 'X_val.npy')),
            np.load(os.path.join(ROOT, 'Y_val.npy')),
        ),
        pickle.load(open(os.path.join(ROOT, 'labels.pkl'), 'rb'))
    )
        
        
def _generate_plantvillage_tomato_dataset(path, x_output_path, y_output_path):
    INPUT_SHAPE = (128, 128)

    gen = keras.preprocessing.image.ImageDataGenerator(rescale=1 / 255)

    dataset = gen.flow_from_directory(
        path,
        target_size=INPUT_SHAPE,
        class_mode="categorical",
        color_mode="rgb",
        batch_size=1,
        shuffle=True
    )
    print(dataset.class_indices.keys())

    X = np.zeros([len(dataset)] + list(dataset.image_shape), dtype=np.float32)
    Y = np.zeros([len(dataset), dataset.num_classes], dtype=np.float32)

    for i in range(len(dataset)):
        image, label = next(dataset)
        X[i] = image
        Y[i] = label

    np.save(x_output_path, X)
    np.save(y_output_path, Y)
        
    return X, Y


def dataset_plant_village_tomato_blight():
    ROOT = 'data/plant-village-tomato-blight'
    X = np.load(os.path.join(ROOT, 'X.npy'))
    
    X_rescaled = np.zeros((X.shape[0], 128, 128, 3))
    
    # Last-minute fix for lower-capacity instances
    
    with warnings.catch_warnings():
        for i in tqdm_notebook(list(range(X.shape[0]))):
            warnings.simplefilter("ignore")
            X_rescaled[i] = rescale(X[i], 128 / 256, multichannel=True)

    return (
        X_rescaled,
        np.load(os.path.join(ROOT, 'Y.npy')),
        ['diseased', 'healthy']
    )



def dataset_plant_village_tomato():
    ROOT = 'data/plant-village-tomato'
    return (
        np.load(os.path.join(ROOT, 'X.npy')),
        np.load(os.path.join(ROOT, 'Y.npy')),
        [
            'bacterial-sport',
            'early-blight',
            'late-blight',
            'leaf-mold',
            'septoria-spot',
            'spider-mites',
            'target-spot',
            'yellowleaf-curl-virus',
            'mosaic-virus',
            'healthy'
        ]
    )


def dataset_pcam():
    ROOT = 'data/pcam'
    return (
        np.load(os.path.join(ROOT, 'X.npy')),
        np.load(os.path.join(ROOT, 'Y.npy')),
        [
            'healthy',
            'metastasis'
        ]
    )


def dataset_pneumonia_small():
    """
    Return batches with downsampled images of the data set.
    """
    INPUT_SHAPE = (32, 32)  # scale down images
    gen = keras.preprocessing.image.ImageDataGenerator(
        # rescale=1 / 255  # scale images into 0-1 range
        samplewise_center=True,
        samplewise_std_normalization=True
    )
    return gen.flow_from_directory(
        DATASET_PATH,        # our data is in this folder
        target_size=INPUT_SHAPE,  # all images will be scaled to this size 
        class_mode="categorical", # tells Keras to treat the folders as classes
        color_mode="grayscale",   # only 1 dimension of pixel values (32, 32, 1)
        batch_size=32,            # compute the gradient every 32 images
        shuffle=True,
        seed=0
    )


def dataset_plantvillage_small():
    INPUT_SHAPE = (64, 64)
    BATCH_SIZE = 32

    gen = keras.preprocessing.image.ImageDataGenerator(rescale=1 / 255)

    return gen.flow_from_directory(
        PLANTVILLAGE_DATASET_PATH,
        target_size=INPUT_SHAPE,
        class_mode="categorical",
        color_mode="rgb",
        batch_size=BATCH_SIZE
    )


def plot_activations_twoclass(model, labels):
    layer_idx = -1

    model.layers[layer_idx].activation = activations.linear
    model = utils.apply_modifications(model)

    fig, axes = plt.subplots(figsize=(10, 6), ncols=2, nrows=1)
    for i, ax, label in zip(range(2), axes, labels):
        img = visualize_activation(model, -1, filter_indices=i, input_range=(0., 1.))
        ax.imshow(img)
        ax.axis('off')
        ax.set_title(label)
    
    return fig, axes


def plot_cam(model, image, class_index, labels):
    # TODO: this is a right old mess
    layer_indices = [i for i, l in enumerate(model.layers) if 'conv' in l.name][1:]

    from_index = max(0, class_index - 2)
    to_index = min(class_index + 3, len(labels))
    num_classes = to_index - from_index
    
    fig, axes = plt.subplots(
        figsize=(15, 4 * num_classes),
        ncols=len(layer_indices),
        nrows=num_classes
    )
    
    class_indices = list(range(from_index, to_index))
    class_names = [labels[i] for i in class_indices]
    
    entries = zip(axes.reshape((1, -1)).squeeze().tolist(), list(product(class_indices, layer_indices)))

    for axis, (_class_index, layer_index) in tqdm_notebook(list(entries), 'Generating class activation maps'):
        grads = visualize_cam(
            model,
            layer_idx=layer_index,
            filter_indices=_class_index,
            seed_input=image,
            backprop_modifier='guided'
        )
        jet_heatmap = np.uint8(cm.jet(grads)[..., :3] * 255)
        axis.imshow(overlay(jet_heatmap, image * 255, alpha=.4))
        axis.get_xaxis().set_ticks([])
        axis.get_yaxis().set_ticks([])

    for ax, col in zip(axes[0], [model.layers[i].name for i in layer_indices]):
        ax.set_title(col, fontsize=20, y=1.02)

    for ax, row in zip(axes[:,0], class_names):
        if row == labels[class_index]:
            weight = 'bold'
        else:
            weight = 'normal'
            
        ax.set_ylabel(row, rotation=90, fontsize=20, labelpad=10, weight=weight)

    fig.suptitle('Label: {}'.format(labels[class_index]), fontsize=22, y=1.02)
    plt.tight_layout()
    
    return fig, axes



def plot_confusion_matrix(model, X, Y, labels, verbose=True):
    predictions = model.predict(X, verbose=verbose)
    cm = confusion_matrix(np.argmax(Y, axis=1), np.argmax(predictions, axis=1))
    
    tick_marks = np.arange(Y.shape[1])
    plt.xticks(tick_marks, labels, rotation=45, fontsize=16)
    plt.yticks(tick_marks, labels, fontsize=16)
    plt.title("Confusion matrix", fontsize=20)
    fmt = 'd'
    thresh = cm.max() / 2.
    for i, j in product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black", fontsize=20)
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.ylabel('True label', fontsize=16)
    plt.xlabel('Predicted label', fontsize=16)


def plot_examples(X, Y, labels, ncols=4):
    # Sort by label for a nicer overview

    class_indices = np.argmax(Y, axis=1)
    indices = np.argsort(class_indices)
    class_indices = class_indices[indices]
    
    Y = Y[indices]
    X = X[indices]

    nrows = math.ceil(len(class_indices) / ncols)
    height = nrows * 3.75
    f, ax = plt.subplots(nrows, ncols, figsize=(15, height))
    for row in range(nrows):
        for column in range(ncols):
            ax[row, column].imshow(X[column + ncols * row])
            ax[row, column].get_xaxis().set_visible(False)
            ax[row, column].get_yaxis().set_visible(False)
            
            label = labels[class_indices[column + ncols * row]]
                
            ax[row, column].set_title(label, fontsize=18, y=1.02)
            
    plt.tight_layout()
    
    return f, ax


def plot_batch(batch, n_cols=8):
    """
    Show images of a batch of a given data set.
    """

    # Sort by label for a nicer picture

    labels = np.argmax(batch[1], axis=1)
    indices = np.argsort(labels)

    labels = labels[indices]
    images = batch[0][indices]

    n_rows = math.ceil(len(labels) / n_cols)
    f, ax = plt.subplots(n_rows, n_cols, figsize=(16, 20))
    for row in range(n_rows):
        for column in range(n_cols):
            ax[row, column].imshow(images[column + n_cols * row, :, :])
            ax[row, column].get_xaxis().set_visible(False)
            ax[row, column].get_yaxis().set_visible(False)
            
            label = PLANTVILLAGE_CLASS_NAMES[labels[column + n_cols * row]]
                
            ax[row, column].set_title(label, fontsize=16)
            
    # plt.tight_layout()
    plt.show()


def plot_training(history):
    """
    Plot training and loss curves for training set only.
    """
    fig, axes = plt.subplots(figsize=(10, 8), ncols=1, nrows=2)
    data = history.history
    data['epoch'] = history.epoch
    df = pd.DataFrame(data)
    sns.lineplot(data=df, x='epoch', y='acc', ax=axes[0]);
    axes[0].set_xlabel('')
    axes[0].set_ylim([df['acc'].min(), 1])
    sns.lineplot(data=df, x='epoch', y='loss', ax=axes[1]);
    axes[1].set_ylim([0, df['loss'].max()])
    return fig, axes



def download_data():
    os.system("curl https://surfdrive.surf.nl/files/index.php/s/rLviHBv10MeJwVT/download | tar -xz")


def print_dataset_statistics():
    normal_images =  glob.glob("selection/train/NORMAL/*.jpeg")
    pneumonia_images =  glob.glob("selection/train/PNEUMONIA/*.jpeg")

    print("The dataset is split 80%/20% between training/validation and test set.")
    normal_images_test =  glob.glob("selection/test/NORMAL/*.jpeg")
    pneumonia_images_test =  glob.glob("selection/test/PNEUMONIA/*.jpeg")
    print("Training set, normal examples: ", len(normal_images))
    print("Training set, pneumonia examples: ", len(pneumonia_images))
    print("Test set, normal examples: ", len(normal_images_test))
    print("Test set, pneumonia examples: ", len(pneumonia_images_test))
    print('Size of images: 32 x 32 pixels')
    # print("split = ", (len(normal_images_test) + len(pneumonia_images_test)) / (len(normal_images) + len(pneumonia_images) + len(normal_images_test) + len(pneumonia_images_test)))
    return normal_images, pneumonia_images



def get_datasets():
    INPUT_SHAPE = (32, 32)
    gen = keras.preprocessing.image.ImageDataGenerator(validation_split=0.2,
                                                       rescale=1./255)# This is multiplied to all elementes
    # target_size, all images will be resized to this size
    train_batches = gen.flow_from_directory("selection/train",        # our data is in this folder
                                            target_size=INPUT_SHAPE,  # all images will be scaled to this size 
                                            class_mode="categorical", # tells Keras to treat the folders as classes
                                            color_mode="grayscale",   # only 1 dimension of pixel values (32, 32, 1)
                                            batch_size=32,            # compute the gradient every 32 images
                                            subset="training")        # this is the training set
    validation_batches = gen.flow_from_directory("selection/train",   # same folder
                                            target_size=INPUT_SHAPE,
                                            class_mode="categorical",
                                            color_mode="grayscale",
                                            batch_size=32,
                                            subset="validation")      # but we use a subset for validation
    test_batches = gen.flow_from_directory("selection/test",
                                           target_size=INPUT_SHAPE,
                                           class_mode="categorical",
                                           shuffle=False,             # Do not shuffle the data
                                           color_mode="grayscale", 
                                           batch_size=1)              # evaluate 1 image at a time
    return train_batches, validation_batches, test_batches


def plot_train_val(history):
    sns.set(style='white')  

    df = pd.DataFrame({'acc': history.history['acc'], 'val_acc': history.history['val_acc'], 'epoch': history.epoch})
    df = df.melt(id_vars=['epoch'])

    plt.figure(figsize=(10, 8))
    sns.lineplot(data=df, x='epoch', y='value', hue='variable')
    plt.title('Training vs. validation accuracy', fontsize=16)
    plt.show()

def report_model(model, history, test_batches):
    predictions = evaluate_model(model, test_batches)
    show_confusion_matrix(predictions)
    plot_train_val(history)

def evaluate_model(model, test_batches):
    predictions = model.predict_generator(test_batches, verbose=True, steps=632)
    df = pd.DataFrame(predictions)
    df["filename"] = test_batches.filenames
    df["actual"] = (df["filename"].str.contains("PNEUMONIA")).apply(int)
    df["predicted"] = (df[1]>0.5).apply(int)
    df["file_path"] = 'selection/test/' + df["filename"].astype(str)
    df["diff"] = abs(df[0] - df[1])
    print("Accuracy on test set:", metrics.accuracy_score(df["actual"], df["predicted"]))
    return df

def show_confusion_matrix(predictions):
    cm = sklearn.metrics.confusion_matrix(predictions["actual"], predictions["predicted"])
    tick_marks = np.arange(2)
    classes = ["Normal", "Pneumonia"]
    plt.xticks(tick_marks, classes, rotation=45, fontsize=16)
    plt.yticks(tick_marks, classes, fontsize=16)
    plt.title("Confusion matrix", fontsize=20)
    fmt = 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.tight_layout()
    plt.ylabel('True label', fontsize=16)
    plt.xlabel('Predicted label', fontsize=16)

def show_flat_weights(model):
    plt.figure(figsize=(10, 6))
    plt.subplot(1, 2, 1)
    plt.title("Normal weights", fontsize=16)
    plt.imshow(model.get_weights()[0][:, 0].reshape((32, 32)), cmap='hot', interpolation='nearest')
    plt.axis('off')
    plt.colorbar()
    plt.subplot(1, 2, 2)
    plt.title("Pneumonia weights", fontsize=16)
    plt.imshow(model.get_weights()[0][:, 1].reshape((32, 32)), cmap='hot', interpolation='nearest')
    plt.axis('off')
    plt.colorbar()
    plt.show()

def get_worst_samples(predictions):
    misclassified = predictions[predictions["actual"] != predictions["predicted"]]
    return misclassified.sort_values("diff", ascending=False)


def print_worst_examples(predictions):
    worst = get_worst_samples(predictions)
    show_images(worst["file_path"].head(10).tolist())
    print(worst.head(10))

# def plot_images(images, ncols=4, colormap='magma'):
#     nrows = int(math.ceil(len(images) / ncols))
#     _, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(16, 4 * nrows))
#     axes = axes.flatten()
#     for index, (ax, image) in enumerate(zip(axes, images)):
#         ax.imshow(image, cmap=colormap)
#         ax.get_xaxis().set_visible(False)
#         ax.get_yaxis().set_visible(False)
#         ax.set_title(index)


