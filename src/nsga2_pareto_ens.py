import tensorflow as tf
from sklearn import metrics

from deap import algorithms
from deap import base
from deap import benchmarks
from deap.benchmarks.tools import diversity, convergence, hypervolume
from deap import creator
from deap import tools

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


class Model():
    """
    Base class for NSGA-II
    """
    def __init__(self, seed_model):
        self.tf_model = tf.keras.models.clone_model(seed_model)
        self.tf_model.set_weights(seed_model.get_weights())
        
    def get_weights(self):
        return self.tf_model.get_weights()
        
    def set_weights(self, weights):
        self.tf_model.set_weights(weights)


def rand_init(seed_model):
    """
    Individual initialization using a seed model and an initial perturbation
    """
    # seed model is a tf keras model
    model = tf.keras.models.clone_model(seed_model)
    model.compile(optimizer='adam', 
                  loss=tf.keras.losses.categorical_crossentropy, 
                  metrics=['accuracy'])
    model.set_weights(seed_model.get_weights())    
    gaussian_mutation(model)
    return model


def f1_per_class(test_images, test_labels, individual):
    """
    Fitness function using per class f1
    """
    pred = individual.tf_model.predict(test_images)
    pred_classes = np.argmax(pred, axis=1)
    f1_per_class = metrics.f1_score(test_labels, pred_classes, average=None)
    return f1_per_class
    

def f1_per_class_oa(test_images, test_labels, individual):
    """
    Fitness function using per class f1 and oa f1
    """
    pred = individual.tf_model.predict(test_images)
    pred_classes = np.argmax(pred, axis=1)
    f1_per_class = metrics.f1_score(test_labels, pred_classes, average=None)
    f1 = metrics.f1_score(test_labels, pred_classes, average='weighted')
    return np.append(f1_per_class, f1)


def gaussian_mutation(individual, delta=1e-4):
    """
    Simple additive gaussian weight mutation
    """
    weights = individual.get_weights()
    for w in weights:
        w += np.random.normal(loc=0.0, scale=delta, size=w.shape)        
    individual.set_weights(weights)
    return individual


def drop_gaussian_mutation(individual, delta=1e-4, drop=0.2):
    weights = individual.get_weights()
    for w in weights:
        w += np.random.normal(loc=0.0, scale=delta, size=w.shape)
        w *= (np.random.uniform(size=w.shape) > drop)      
    individual.set_weights(weights)
    return individual


def opt_ensemble(seed=None, pop_size=4, generations=2):
    """
    NSGA-2 optimization of the ensemble
    """
    random.seed(seed)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean, axis=0)
    stats.register("std", np.std, axis=0)
    stats.register("min", np.min, axis=0)
    stats.register("max", np.max, axis=0)
    logbook = tools.Logbook()
    logbook.header = "gen", "evals", "std", "min", "avg", "max"
    pop = toolbox.population(n=pop_size)    
    invalid_ind = [ind for ind in pop if not ind.fitness.valid]
    fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit
    # This is just to assign the crowding distance to the individuals
    # no actual selection is done
    pop = toolbox.select(pop, len(pop))
    record = stats.compile(pop)
    logbook.record(gen=0, evals=len(invalid_ind), **record)
    print(logbook.stream)
    # Begin the generational process
    for gen in range(1, generations):
        # dominance selection. If there is no interdominance, then crowding distance
        offspring = tools.selTournamentDCD(pop, len(pop))
        offspring = [toolbox.clone(ind) for ind in offspring]
        for ind in offspring:
            toolbox.mutate(ind)            
            del ind.fitness.values        
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit
        # Select the next generation population
        pop = toolbox.select(pop + offspring, pop_size)
        record = stats.compile(pop)
        logbook.record(gen=gen, evals=len(invalid_ind), **record)
        print(logbook.stream)
    return pop, logbook


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
          '--fitsplit',
          type=int,
          default=1000,
          help='Number of samples used to compute the fitness of individuals (default 1000).'
    )
    parser.add_argument(
          '--popsize',
          type=int,
          default=12,
          help='Population size (default 12).'
    )
    parser.add_argument(
          '--generations',
          type=int,
          default=10,
          help='Number of generations (default 10).'
    )
    parser.add_argument(
          '--oa',
          action=argparse.BooleanOptionalAction,
          help='Include overall metric')
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
    
    # Prepare optimization problem
    toolbox = base.Toolbox()
    
    if FLAGS.oa:
        # Version including overall performance
        creator.create("FitnessMax", base.Fitness, weights=((1.0, ) * (num_classes+1) ))
    else:
        # Version only class performance
        creator.create("FitnessMax", base.Fitness, weights=((1.0, ) * num_classes ))

    creator.create("Individual", Model, fitness=creator.FitnessMax)
    toolbox.register("rand_init", rand_init, init_model) # init_model is the model loaded/created above
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.rand_init)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    # select subset of images for the fitness evaluation
    # TODO maybe do this with a function that returns different samples every time?
    pos = np.random.choice(len(train_images), size=FLAGS.fitsplit)
    val_images = train_images[pos,:,:,:]
    val_labels = train_labels_ix[pos,:]

    if FLAGS.oa:
        # fitness including overall
        toolbox.register("evaluate", f1_per_class_oa, val_images, val_labels)
    else:
        # fitness per class 	
        toolbox.register("evaluate", f1_per_class, val_images, val_labels)
    
    # TODO select with command
    toolbox.register("mutate", gaussian_mutation)
    # toolbox.register("mutate", drop_gaussian_mutation)
    toolbox.register("select", tools.selNSGA2)
    
    # optimze the ensemble
    t3 = time.time()    
    pop, pop_stats = opt_ensemble(seed=FLAGS.seed, pop_size=FLAGS.popsize, generations=FLAGS.generations)
    t4 = time.time()
    
    acc = []
    iou = []
    f1 = []
    ind_pred_classes = None
    ind_pred_probs = list()
    acc_shifted = []
    iou_shifted = []
    f1_shifted = []
    ind_pred_classes_shifted = None
    ind_pred_shifted_probs = list()

    for ind in pop:
        pred = ind.tf_model.predict(test_images)
        ind_pred_probs.append(pred)
        pred_shifted = ind.tf_model.predict(shifted_images)
        ind_pred_shifted_probs.append(pred_shifted)
        pred_classes = np.argmax(pred, axis=1)
        pred_classes_shifted = np.argmax(pred_shifted, axis=1)
        
        if ind_pred_classes is None:
            ind_pred_classes = pred_classes
        else:
            ind_pred_classes = np.vstack([ind_pred_classes, pred_classes])
            
        if ind_pred_classes_shifted is None:
            ind_pred_classes_shifted = pred_classes_shifted
        else:
            ind_pred_classes_shifted = np.vstack([ind_pred_classes_shifted, pred_classes_shifted])
            
        acc.append(metrics.accuracy_score(test_labels_ix, pred_classes))
        iou.append(metrics.jaccard_score(test_labels_ix, pred_classes, average="weighted"))
        f1.append(metrics.f1_score(test_labels_ix, pred_classes, average="weighted"))
        
        acc_shifted.append(metrics.accuracy_score(shifted_labels_ix, pred_classes_shifted))
        iou_shifted.append(metrics.jaccard_score(shifted_labels_ix, pred_classes_shifted, average="weighted"))
        f1_shifted.append(metrics.f1_score(shifted_labels_ix, pred_classes_shifted, average="weighted"))
        
    stats_acc = compute_stats(acc)
    stats_iou = compute_stats(iou)
    stats_f1 = compute_stats(f1)
    
    # make ensemble predictions
    pred_mode = stats.mode(ind_pred_classes).mode    
    stats_acc_shifted = compute_stats(acc_shifted)
    stats_iou_shifted = compute_stats(iou_shifted)
    stats_f1_shifted = compute_stats(f1_shifted)
    
    pred_mode_shifted = stats.mode(ind_pred_classes_shifted).mode
    mode_acc = metrics.accuracy_score(test_labels_ix, pred_mode)
    mode_iou = metrics.jaccard_score(test_labels_ix, pred_mode, average="weighted")
    mode_f1 = metrics.f1_score(test_labels_ix, pred_mode, average="weighted")
    
    mode_acc_shifted = metrics.accuracy_score(shifted_labels_ix, pred_mode_shifted)
    mode_iou_shifted = metrics.jaccard_score(shifted_labels_ix, pred_mode_shifted, average="weighted")
    mode_f1_shifted = metrics.f1_score(shifted_labels_ix, pred_mode_shifted, average="weighted")
    
    # ensemble performance metrics
    # variance
    tmp = np.stack(ind_pred_probs, axis=1)
    var_samp_class = np.var(tmp, axis=1)
    var_samp = np.sum(var_samp_class, axis=1)
    var_ens = np.mean(var_samp)
    
    tmp = np.stack(ind_pred_shifted_probs, axis=1)
    var_samp_class = np.var(tmp, axis=1)
    var_samp = np.sum(var_samp_class, axis=1)
    var_ens_shifted = np.mean(var_samp)

    # entropy
    l_z = None
    for i in range(ind_pred_classes.shape[0]):
        if l_z is None:
            l_z = (ind_pred_classes[i,:] == test_labels_ix[:,0]) * 1
        else:
            l_z = l_z + (ind_pred_classes[i,:] == test_labels_ix[:,0]) * 1

    entropy_ens = np.mean(np.minimum(l_z, ind_pred_classes.shape[0] - l_z) / (ind_pred_classes.shape[0] - np.ceil(ind_pred_classes.shape[0]/2)))
    
    l_z = None
    for i in range(ind_pred_classes_shifted.shape[0]):
        if l_z is None:
            l_z = (ind_pred_classes_shifted[i,:] == shifted_labels_ix[:,0]) * 1
        else:
            l_z = l_z + (ind_pred_classes_shifted[i,:] == shifted_labels_ix[:,0]) * 1

    entropy_ens_shifted = np.mean(np.minimum(l_z, ind_pred_classes_shifted.shape[0] - l_z) / (ind_pred_classes_shifted.shape[0] - np.ceil(ind_pred_classes_shifted.shape[0]/2)))


    print("Initial vs Ensemble")
    print("Accuracy: %5.4f vs %5.4f" % (init_acc, mode_acc))
    print("IoU: %5.4f vs %5.4f" % (init_iou, mode_iou))
    print("F1 Score: %5.4f vs %5.4f" % (init_f1, mode_f1))
    
    print("\n\nInitial vs Ensemble shifted data set")
    print("Accuracy: %5.4f vs %5.4f" % (init_acc_shifted, mode_acc_shifted))
    print("IoU: %5.4f vs %5.4f" % (init_iou_shifted, mode_iou_shifted))
    print("F1 Score: %5.4f vs %5.4f" % (init_f1_shifted, mode_f1_shifted))
    
    print("\n\nVar ensemble %5.4f" % var_ens)
    print("Entropy ensemble %5.4f" % entropy_ens)
    print("\n\nVar ensemble shifted %5.4f" % var_ens_shifted)
    print("Entropy ensemble shifted %5.4f" % entropy_ens_shifted)
   
    data = dict()
    data['flags'] = vars(FLAGS)
    data['time_model'] = t2 - t1
    data['time_ensemble'] = t4 - t3
    data['version'] = 1.0
    data['id'] = hash(frozenset(vars(FLAGS)))
    
    data['base'] = dict()
    data['base']['acc'] = init_acc
    data['base']['iou'] = init_iou
    data['base']['f1'] = init_f1
    data['base']['acc_shifted'] = init_acc_shifted
    data['base']['iou_shifted'] = init_iou_shifted
    data['base']['f1_shifted'] = init_f1_shifted
    
    data['mode'] = dict()
    data['mode']['acc'] = mode_acc
    data['mode']['iou'] = mode_iou
    data['mode']['f1'] = mode_f1
    data['mode']['acc_shifted'] = mode_acc_shifted
    data['mode']['iou_shifted'] = mode_iou_shifted
    data['mode']['f1_shifted'] = mode_f1_shifted
    data['mode']['var_ens'] = var_ens.astype(float)
    data['mode']['var_ens_shifted'] = var_ens_shifted.astype(float)
    data['mode']['entropy_ens'] = entropy_ens.astype(float)
    data['mode']['entropy_ens_shifted'] = entropy_ens_shifted.astype(float)
    
    data['pop'] = dict()
    data['pop']['acc'] = stats_acc
    data['pop']['iou'] = stats_iou
    data['pop']['f1'] = stats_f1
    data['pop']['acc_shifted'] = stats_acc_shifted
    data['pop']['iou_shifted'] = stats_iou_shifted
    data['pop']['f1_shifted'] = stats_f1_shifted
      
    with open('metrics_nsga2.json', 'a', encoding='utf-8') as f:
        json.dump(data, f)
        f.write('\n')
        
    predictions = dict()
    predictions['flags'] = vars(FLAGS)
    predictions['version'] = 1.0
    predictions['test'] = np.stack(ind_pred_probs, axis=1).astype(float).tolist()
    predictions['shifted'] = np.stack(ind_pred_shifted_probs, axis=1).astype(float).tolist()
    
        
    filename = "h" + str(data['id']) + "_ens_probs.npy"
    with open(filename, 'wb') as f:
        np.save(f, np.stack(ind_pred_probs, axis=1))
        np.save(f, np.stack(ind_pred_shifted_probs, axis=1))
        
    filename = "h" + str(data['id']) + "_init_probs.npy"
    with open(filename, 'wb') as f:
        np.save(f, init_pred)
        np.save(f, init_pred_shifted)
