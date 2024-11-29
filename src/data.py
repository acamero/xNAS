import os
import tensorflow as tf
import numpy as np


class DataLoader():
    
    def __init__(self, data_path="localdata/"):
        self.datasets = dict()
        self.datasets['CIFAR10'] = self.load_data_CIFAR10
        self.data_path = data_path


    def load_cifar10_1(self, version_string=''):
        """
        CIFAR10.1 data loader
        """
        filename = 'cifar10.1'
        if version_string == '':
            version_string = 'v7'
        if version_string in ['v4', 'v6', 'v7']:
            filename += '_' + version_string
        else:
            raise ValueError('Unknown dataset version "{}".'.format(version_string))
        label_filename = filename + '_labels.npy'
        imagedata_filename = filename + '_data.npy'
        label_filepath = os.path.abspath(os.path.join(self.data_path, label_filename))
        imagedata_filepath = os.path.abspath(os.path.join(self.data_path, imagedata_filename))
        labels = np.load(label_filepath)
        imagedata = np.load(imagedata_filepath)
        assert len(labels.shape) == 1
        assert len(imagedata.shape) == 4
        assert labels.shape[0] == imagedata.shape[0]
        assert imagedata.shape[1] == 32
        assert imagedata.shape[2] == 32
        assert imagedata.shape[3] == 3
        if version_string == 'v6' or version_string == 'v7':
            assert labels.shape[0] == 2000
        elif version_string == 'v4':
            assert labels.shape[0] == 2021

        labels = np.reshape(labels, (labels.shape[0], 1))
        return imagedata, labels


    def load_data_CIFAR10(self):
        # Load the data
        (train_images, train_labels_ix), (test_images, test_labels_ix) = tf.keras.datasets.cifar10.load_data()
        # Converting the pixels data to float type
        train_images = train_images.astype('float32')
        test_images = test_images.astype('float32') 
        # Standardizing (255 is the total number of pixels an image can have)
        train_images = train_images / 255
        test_images = test_images / 255 
        # One hot encoding the target class (labels)
        num_classes = 10
        train_labels = tf.keras.utils.to_categorical(train_labels_ix, num_classes)
        test_labels = tf.keras.utils.to_categorical(test_labels_ix, num_classes)
        # Load the shifted images (CIFAR10.1)
        shifted_images, shifted_labels_ix = self.load_cifar10_1('v4')
        shifted_images = shifted_images.astype('float32')
        shifted_images = shifted_images / 255
        shifted_labels = tf.keras.utils.to_categorical(shifted_labels_ix, num_classes)
    
        return (train_images, train_labels_ix, train_labels, 
               test_images, test_labels_ix, test_labels, 
               shifted_images, shifted_labels_ix, shifted_labels,
               num_classes)


    def load_data(self, dataset):
       assert dataset in self.datasets
       return self.datasets[dataset]()
