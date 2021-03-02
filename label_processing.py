import itertools
import os
import numpy as np
from multiprocessing import Process
from multiprocessing import Manager
from sklearn import metrics
from operator import itemgetter

def relabel(pred, old_label, new_label):
    return [i if i != old_label else new_label for i in pred]

# index in list represent original label and value is what they're replabled to
def relabel_list(labels, clusters, relabel_list):
    count = 0
    for i in clusters:
        labels = relabel(labels, i, relabel_list[count] * -1) # i represents the cluster
        count += 1
    return [abs(val) for val in labels]

def get_combinations(unique_labels, length, label_list, sequence_list):
    if (len(label_list) >= length):
        sequence_list.append(label_list)
    else:
        labels_copy = unique_labels.copy()
        for i in range(0, len(unique_labels)):
            list_copy = label_list.copy()
            list_copy.append(unique_labels[i])
            get_combinations(labels_copy, length, list_copy, sequence_list)
            labels_copy.pop(0)

def reassign_labels(pred, true, unique_labels):
    if (len(set(true)) == len(unique_labels)):
        relabel_lists = list(set(itertools.permutations(unique_labels)))
    elif(len(set(true)) < len(unique_labels)):
        length = len(unique_labels)
        relabel_lists = []
        sequence_list = []
        get_combinations(list(set(true)), length, list(set(true)), sequence_list)
        for i in range(0, len(sequence_list)):
            relabel_lists.extend(list(set(itertools.permutations(sequence_list[i]))))
    else:
        raise Exception("more classes than clusters!")
    
    number_of_processes = min(os.cpu_count() - 1, len(relabel_lists))
    chunk_relabel_lists = np.array_split(relabel_lists, number_of_processes)
        
    results = Manager().list()
    processes = []
    clusters = list(set(pred))
    clusters.sort()
        
    for i in range(0, number_of_processes):
        p = Process(target=best_label_list, args=(pred.copy(), true.copy(), clusters.copy(), chunk_relabel_lists[i], results))
        processes.append(p)
        
    for process in processes:
        process.start()
        
    for process in processes:
        process.join()

    return max(results,key=itemgetter(0))[1]
    
def best_label_list(pred, true, clusters, relabel_lists, results):
    f1 = 0
    label_list = None
    for i in range(0, len(relabel_lists)):
        score = metrics.f1_score(relabel_list(pred.copy(), clusters, list(relabel_lists[i])), true, average='weighted')
        if (score > f1):
            f1 = score
            label_list = list(relabel_lists[i])
    results.append((f1, label_list))

def print_results(pred, true, labels):
    print("Assigned Labels:")
    print(labels)
    print()
    print("F1:")
    print(metrics.f1_score(pred, true, average='weighted'))
    print()
    print("Accuracy:")
    print(metrics.accuracy_score(pred, true))
    print()
    print("ARI:")
    print(metrics.adjusted_rand_score(pred, true))