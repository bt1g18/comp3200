import itertools
import numpy as np
from multiprocessing import Process
from multiprocessing import Manager
from sklearn import metrics
from operator import itemgetter

def relabel(pred, old_label, new_label):
    return [i if i != old_label else new_label for i in pred]

def swap_labels(pred, label_1, label_2):
    u_pred = relabel(pred, label_1, label_2)
    return u_pred

# index in list represent original label and value is what they're replabled to
def relabel_list(labels, relabel_list):
    for i in range(0, len(set(labels))):
        labels = swap_labels(labels, i, relabel_list[i] * -1) # i represents the cluster
    return [abs(val) for val in labels]

def get_combinations(unique_labels, length, label_list, combin_list):
    if (len(label_list) >= length):
        combin_list.append(label_list)
    else:
        labels_copy = unique_labels.copy()
        for i in range(0, len(unique_labels)):
            list_copy = label_list.copy()
            list_copy.append(unique_labels[i])
            get_combinations(labels_copy, length, list_copy, combin_list)
            labels_copy.pop(0)

def reassign_labels(pred, true, unique_labels):
    if (len(set(true)) == len(unique_labels)):
        relabel_lists = list(itertools.permutations(unique_labels))
    elif(len(set(true)) < len(unique_labels)):
        length = len(unique_labels)
        relabel_lists = []
        combin_list = []
        get_combinations(list(set(true)), length, list(set(true)), combin_list)
        for i in range(0, len(combin_list)):
            relabel_lists.extend(list(itertools.permutations(combin_list[i])))
    else:
        raise Exception("more classes than clusters!")
    
    number_of_processes = min(15, len(relabel_lists))
    chunk_relabel_lists = np.array_split(relabel_lists, number_of_processes)

        
    results = Manager().list()
    processes = []
    
    for i in range(0, number_of_processes):
        p = Process(target=best_label_list, args=(pred, true, chunk_relabel_lists[i], results))
        processes.append(p)
        
    for process in processes:
        process.start()
        
    for process in processes:
        process.join()

    return max(results,key=itemgetter(0))[1]
    
def best_label_list(pred, true, relabel_lists, results):
    f1 = 0
    label_list = None
    for i in range(0, len(relabel_lists)):
        score = metrics.f1_score(relabel_list(pred.copy(), list(relabel_lists[i])), true, average='weighted')
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