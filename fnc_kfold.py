import sys
import numpy as np
import time
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
from feature_engineering import refuting_features, polarity_features, hand_features, gen_or_load_feats, sentiment_features, cosine_tfidf_features, bleu_features, glove_features
from feature_engineering import word_overlap_features, get_glove_matrix
from rnn.conditional import get_model_conditional, test_trainer, predict_holdout
from utils.dataset import DataSet
from utils.generate_test_splits import kfold_split, get_stances_for_folds, load_train_nn
from utils.score import report_score, LABELS, LABELS_ONE_HOT, score_submission
from utils.system import parse_params, check_version


def generate_features(stances,dataset,name,filters=False):
    h, b, y = [],[],[]

    for stance in stances:
        if filters:
            if LABELS.index(stance['Stance']) == 3:
                y.append(0)
            else:
                y.append(1)
        else :
            y.append(LABELS.index(stance['Stance']))
        h.append(stance['Headline'])
        b.append(dataset.articles[stance['Body ID']])

    X_overlap = gen_or_load_feats(word_overlap_features, h, b, "features/overlap."+name+".npy")
    X_refuting = gen_or_load_feats(refuting_features, h, b, "features/refuting."+name+".npy")
    X_polarity = gen_or_load_feats(polarity_features, h, b, "features/polarity."+name+".npy")
    X_hand = gen_or_load_feats(hand_features, h, b, "features/hand."+name+".npy")
    #X_sentiment = gen_or_load_feats(sentiment_features, h, b, "features/sentiment."+name+".npy")
    #X_cosinetfidf = gen_or_load_feats(cosine_tfidf_features, h, b, "features/cosinetfidf."+name+".npy")
    #X_bleu = gen_or_load_feats(bleu_features, h, b, "features/bleu."+name+".npy")
    

    X = np.c_[X_refuting, X_polarity, X_hand, X_overlap]
    return X,y

def generate_features_nn(stances,dataset,name,filters=False):
    h, b, y = [],[],[]

    for stance in stances:
        if stance['Stance'] != 'unrelated' or not filters:
            y.append(LABELS_ONE_HOT[stance['Stance']])
            h.append(stance['Headline'])
            b.append(dataset.articles[stance['Body ID']])

    y = np.asarray(y)

    X_glove = gen_or_load_feats(glove_features, h, b, "features/glove_features."+name+".npy")

    return X_glove,y


if __name__ == "__main__":

    #params = { 'n_folds' : 10, 'n_estimators' : 200}
    params = { 'n_folds' : 5, 'n_estimators' : 25}

    check_version()
    parse_params()

    time_1 = time.time()
    d = DataSet()
    time_2 = time.time()
    print('Dataset load: '+str(time_2-time_1))

    folds,hold_out = kfold_split(d,n_folds=params['n_folds'])

    time_3 = time.time()
    print('Kfold_split: '+str(time_3-time_2))

    fold_stances, hold_out_stances = get_stances_for_folds(d,folds,hold_out)
    time_4 = time.time()
    print('Get stances: '+str(time_4-time_3))

    Xs = dict()
    ys = dict()
    Xnn = dict()
    ynn = dict()

    # Load/Precompute all features now
    X_holdout,y_holdout = generate_features(hold_out_stances,d,"holdout")
    Xnn_holdout,ynn_holdout = generate_features_nn(hold_out_stances,d,"holdout")

    for fold in fold_stances:
        Xs[fold],ys[fold] = generate_features(fold_stances[fold],d,str(fold), True)
        Xnn[fold],ynn[fold] = generate_features_nn(fold_stances[fold],d,str(fold), True)
    time_5 = time.time()
    print('Generate features: '+str(time_5-time_4))


    best_score = 0
    best_fold = None


    #Classifier for each fold
    for fold in fold_stances:
        ids = list(range(len(folds)))
        del ids[fold]

        X_train = np.vstack(tuple([Xs[i] for i in ids]))
        y_train = np.hstack(tuple([ys[i] for i in ids]))

        X_test = Xs[fold]
        y_test = ys[fold]

        clf = GradientBoostingClassifier(n_estimators=params['n_estimators'], random_state=14128)
        clf.fit(X_train, y_train)
        

        predicted =  clf.predict(X_test)
        actual = y_test

        fold_score = f1_score(actual, predicted)
        max_fold_score = f1_score(actual, actual)

        score = fold_score/max_fold_score

        print("Score for fold "+ str(fold) + " was - f1 " + str(score) + " - " + str(accuracy_score(actual,predicted)))
        if score > best_score:
            best_score = score
            best_fold = clf
        break


    pred_res = []
    truth_res = []
    X_remain = []
    Y_remain = []

    # X_holdout = X_holdout[:10]
    # y_holdout = y_holdout[:10]
    # Xnn_holdout = Xnn_holdout[:10]
    # ynn_holdout = ynn_holdout[:10]

    predicted = best_fold.predict(X_holdout)

    for idx, val in enumerate(predicted):
        if val == 0:
            pred_res.append(LABELS[3])
            truth_res.append(LABELS[int(y_holdout[idx])])
        else:
            X_remain.append(Xnn_holdout[idx])
            Y_remain.append(ynn_holdout[idx])

    hidden_size = 50
    max_epochs = 10
    tanhOrSoftmax = "softmax"
    dropout = True


    X_train = np.vstack(tuple(value for key,value in Xnn.items()))
    y_train = np.vstack(tuple(value for key,value in ynn.items()))

    train_i = int(X_train.shape[0]*0.8)

    train_headlines, train_bodies, train_labels = load_train_nn(X_train,y_train,train_i)

    y_copy = ["".join(map(str, i)) for i in train_labels]

    unique, counts = np.unique(y_copy, return_counts=True)
    print(np.asarray((unique,counts)).T)

    test_headlines = np.vstack([np.expand_dims(x, 0) for x in X_train[train_i:,0]])
    test_bodies = np.vstack([np.expand_dims(x, 0) for x in X_train[train_i:,1]])
    test_labels = np.vstack([np.expand_dims(x, 0) for x in y_train[train_i:]])

    test_trainer(train_headlines, train_bodies, train_labels, test_headlines, test_bodies, test_labels, hidden_size, max_epochs, tanhOrSoftmax, dropout,0)

    X_remain = np.asarray(X_remain)
    Y_remain = np.asarray(Y_remain)

    pred_nn, actual_nn = predict_holdout(X_remain[:,0], X_remain[:,1], Y_remain, tanhOrSoftmax, hidden_size, max_epochs)

    pred_res = pred_res + pred_nn
    truth_res = truth_res + actual_nn

    report_score(truth_res,pred_res)

    time_6 = time.time()
    print('Train classifier: '+str(time_6-time_5))
