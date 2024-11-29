import tensorflow as tf
from sklearn import metrics

import array
import random
import json
import math

from scipy import stats
import numpy as np

import argparse
import json

import time

import models as m
import data as d


def compute_stats(metric):
    """
    Compute stat desc of a metric
    """
    stats = dict()
    stats['mean'] = np.mean(metric)
    stats['std'] = np.std(metric)
    stats['min'] = np.min(metric)
    stats['max'] = np.max(metric)
    return stats



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
          '--seed',
          type=int,
          default=None,
          help='Random seed.'
    )
    parser.add_argument(
          '--epochs',
          type=int,
          default=4,
          help='Number of epochs (default 4).'
    )
    parser.add_argument(
          '--valsplit',
          type=float,
          default=0.2,
          help='Validation split (default 0.2).'
    )
    parser.add_argument(
          '--model',
          type=str,
          default='cnn',
          help='Deep learning model (one from: ' + str(m.ModelLoader.available_models()) + ") or path to keras model"
    )
    
    

    FLAGS, unparsed = parser.parse_known_args()
    print(FLAGS, '\n')
    
    np.random.seed(FLAGS.seed)

    # Load the data
    dataLoader = d.DataLoader()
    (train_images, train_labels_ix, train_labels, 
           test_images, test_labels_ix, test_labels, 
           shifted_images, shifted_labels_ix, shifted_labels,
           num_classes ) = dataLoader.load_data('CIFAR10')
    
    # Load/create initial seed model
    t1 = time.time()    
    m_loader = m.ModelLoader(train_images, train_labels, num_classes, epochs=FLAGS.epochs, validation_split=FLAGS.valsplit)
    init_model = m_loader.get_model(FLAGS.model)
    t2 = time.time()
    init_model.summary()
    
    # Compute reference performance metrics
    init_pred = init_model.predict(test_images)
    init_pred_classes = np.argmax(init_pred, axis=1)   
    init_acc = metrics.accuracy_score(test_labels_ix, init_pred_classes)
    init_iou = metrics.jaccard_score(test_labels_ix, init_pred_classes, average="weighted")
    init_f1 = metrics.f1_score(test_labels_ix, init_pred_classes, average="weighted")    
    # references for the shifted data set
    init_pred_shifted = init_model.predict(shifted_images)
    init_pred_classes_shifted = np.argmax(init_pred_shifted, axis=1)
    init_acc_shifted = metrics.accuracy_score(shifted_labels_ix, init_pred_classes_shifted)
    init_iou_shifted = metrics.jaccard_score(shifted_labels_ix, init_pred_classes_shifted, average="weighted")
    init_f1_shifted = metrics.f1_score(shifted_labels_ix, init_pred_classes_shifted, average="weighted")
    
    print("Performance metrics normal vs shifted")
    print("Accuracy: %5.4f vs %5.4f" % (init_acc, init_acc_shifted))
    print("IoU: %5.4f vs %5.4f" % (init_iou, init_iou_shifted))
    print("F1 Score: %5.4f vs %5.4f" % (init_f1, init_f1_shifted))
   
    data = dict()
    data['flags'] = vars(FLAGS)
    data['time_model'] = t2 - t1
    data['version'] = 1.0
    data['id'] = hash(frozenset(vars(FLAGS)))
    
    data['base'] = dict()
    data['base']['acc'] = init_acc
    data['base']['iou'] = init_iou
    data['base']['f1'] = init_f1
    data['base']['acc_shifted'] = init_acc_shifted
    data['base']['iou_shifted'] = init_iou_shifted
    data['base']['f1_shifted'] = init_f1_shifted
    
    with open('metrics_single.json', 'a', encoding='utf-8') as f:
        json.dump(data, f)
        f.write('\n')
           
    filename = "h" + str(data['id']) + "_probs.npy"
    with open(filename, 'wb') as f:
        np.save(f, init_pred)
        np.save(f, init_pred_shifted)
        
    filename = "h" + str(data['id']) + "_model.keras"
    init_model.save(filename)
