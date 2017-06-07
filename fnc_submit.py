import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.externals import joblib
from utils.dataset import DataSet
from feature_engineering import refuting_features, word_overlap_features, polarity_features, hand_features, gen_or_load_feats, glove_features
from collections import defaultdict
from rnn.conditional import predict_submission
from utils.score import LABELS
import pandas as pd

def generate_features(dataset):
    h, b = [],[]

    for stance in dataset.stances:
        h.append(stance['Headline'])
        b.append(dataset.articles[stance['Body ID']])

    X_overlap = gen_or_load_feats(word_overlap_features, h, b, "features/overlap.npy")
    X_refuting = gen_or_load_feats(refuting_features, h, b, "features/refuting.npy")
    X_polarity = gen_or_load_feats(polarity_features, h, b, "features/polarity.npy")
    X_hand = gen_or_load_feats(hand_features, h, b, "features/hand.npy")
    

    X = np.c_[X_refuting, X_polarity, X_hand, X_overlap]
    return X

def generate_features_nn(dataset):
    h, b = [],[]

    for stance in dataset.stances:
        h.append(stance['Headline'])
        b.append(dataset.articles[stance['Body ID']])

    X_glove = gen_or_load_feats(glove_features, h, b, "features/glove_features.npy")

    return X_glove

if __name__ == "__main__":

    d = DataSet(set_type='test')
    
    limit = 50
    
    Xs = generate_features(d)[:limit]
    Xnn = generate_features_nn(d)[:limit]
    
    predictions = {}
    
    clf = joblib.load('model/gb.pkl') 
    
    Ys = clf.predict(Xs[:limit])
    
    X_remain = []
    ids_remain = []
    for idx,val in enumerate(Ys):
        if val == 0:
            predictions[idx] = 3
        else:
           X_remain.append(Xnn[idx])
           ids_remain.append(idx)
           
        
    X_remain = np.asarray(X_remain)

    pred_nn = predict_submission(X_remain[:,0], X_remain[:,1], ids_remain, 'softmax', 50, 15)
    
    predictions = { **predictions, **pred_nn}
    
    print([LABELS[predictions.get(i,3)] for i in range(50)])
    
#    df = pd.read_csv('fnc-1/test_stances_unlabeled.csv')
#    
#    df['Stance'] = [LABELS[predictions.get(i,3)] for i in range(25413)]
#    
#    df.to_csv('submission.csv',index=False)