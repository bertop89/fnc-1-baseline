import random
import os
from collections import defaultdict
import numpy as np
from random import randint


def generate_hold_out_split (dataset, training = 0.8, base_dir="splits"):
    r = random.Random()
    r.seed(1489215)

    article_ids = list(dataset.articles.keys())  # get a list of article ids
    r.shuffle(article_ids)  # and shuffle that list

    amount = len(article_ids)
    training_ids = article_ids[:int(training * amount)]
    hold_out_ids = article_ids[int(training * amount):]

    # write the split body ids out to files for future use
    with open(base_dir+ "/"+ "training_ids.txt", "w+") as f:
        f.write("\n".join([str(id) for id in training_ids]))

    with open(base_dir+ "/"+ "hold_out_ids.txt", "w+") as f:
        f.write("\n".join([str(id) for id in hold_out_ids]))


def read_ids(file,base):
    ids = []
    with open(base+"/"+file,"r") as f:
        for line in f:
           ids.append(int(line))
        return ids


def kfold_split(dataset, training = 0.8, n_folds = 10, base_dir="splits"):
    if not (os.path.exists(base_dir+ "/"+ "training_ids.txt")
            and os.path.exists(base_dir+ "/"+ "hold_out_ids.txt")):
        generate_hold_out_split(dataset,training,base_dir)

    training_ids = read_ids("training_ids.txt", base_dir)
    hold_out_ids = read_ids("hold_out_ids.txt", base_dir)

    folds = []
    for k in range(n_folds):
        folds.append(training_ids[int(k*len(training_ids)/n_folds):int((k+1)*len(training_ids)/n_folds)])

    return folds,hold_out_ids


def get_stances_for_folds(dataset,folds,hold_out):
    stances_folds = defaultdict(list)
    stances_hold_out = []
    for stance in dataset.stances:
        if stance['Body ID'] in hold_out:
            stances_hold_out.append(stance)
        else:
            fold_id = 0
            for fold in folds:
                if stance['Body ID'] in fold:
                    stances_folds[fold_id].append(stance)
                fold_id += 1

    return stances_folds,stances_hold_out

def load_train_nn(Xnn,ynn,train_i):
    if not os.path.isfile('features/nn/headlines.npy'):
        
        train_headlines = np.vstack([np.expand_dims(x, 0) for x in Xnn[:train_i,0]])
        train_bodies = np.vstack([np.expand_dims(x, 0) for x in Xnn[:train_i,1]])
        train_labels = np.vstack([np.expand_dims(x, 0) for x in ynn[:train_i]])

        agree_idx = []
        while len(agree_idx) < 2500:
            idx = randint(0,train_headlines.shape[0]-1)
            if (train_labels[idx] == [1,0,0]).all():
                agree_idx.append(idx)

        disagree_idx = []
        while len(disagree_idx) < 4000:
            idx = randint(0,train_headlines.shape[0]-1)
            if (train_labels[idx] == [0,1,0]).all():
                disagree_idx.append(idx)

        for i in agree_idx:
            train_headlines = np.append(train_headlines, train_headlines[i].reshape(1,20), axis=0)
            train_bodies = np.append(train_bodies, train_bodies[i].reshape(1,200), axis=0)
            train_labels = np.append(train_labels, train_labels[i].reshape(1,3), axis=0)

        for i in disagree_idx:
            train_headlines = np.append(train_headlines, train_headlines[i].reshape(1,20), axis=0)
            train_bodies = np.append(train_bodies, train_bodies[i].reshape(1,200), axis=0)
            train_labels = np.append(train_labels, train_labels[i].reshape(1,3), axis=0)

        np.save('features/nn/headlines.npy', train_headlines)
        np.save('features/nn/bodies.npy', train_bodies)
        np.save('features/nn/labels.npy', train_labels)

    return np.load('features/nn/headlines.npy'), np.load('features/nn/bodies.npy'), np.load('features/nn/labels.npy')
