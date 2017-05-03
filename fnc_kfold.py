import sys
import numpy as np
import time
from sklearn.ensemble import GradientBoostingClassifier
from feature_engineering import refuting_features, polarity_features, hand_features, gen_or_load_feats, sentiment_features, cosine_tfidf_features, bleu_features, glove_features
from feature_engineering import word_overlap_features
from conditional import get_model_conditional
from utils.dataset import DataSet
from utils.generate_test_splits import kfold_split, get_stances_for_folds
from utils.score import report_score, LABELS, score_submission
from utils.system import parse_params, check_version


def generate_features(stances,dataset,name):
    h, b, y = [],[],[]

    for stance in stances:
        y.append(LABELS.index(stance['Stance']))
        h.append(stance['Headline'])
        b.append(dataset.articles[stance['Body ID']])

    X_overlap = gen_or_load_feats(word_overlap_features, h, b, "features/overlap."+name+".npy")
    X_refuting = gen_or_load_feats(refuting_features, h, b, "features/refuting."+name+".npy")
    X_polarity = gen_or_load_feats(polarity_features, h, b, "features/polarity."+name+".npy")
    X_hand = gen_or_load_feats(hand_features, h, b, "features/hand."+name+".npy")
    X_sentiment = gen_or_load_feats(sentiment_features, h, b, "features/sentiment."+name+".npy")
    X_cosinetfidf = gen_or_load_feats(cosine_tfidf_features, h, b, "features/cosinetfidf."+name+".npy")
    #X_bleu = gen_or_load_feats(bleu_features, h, b, "features/bleu."+name+".npy")
    

    X = np.c_[X_refuting, X_polarity, X_hand, X_overlap]
    return X,y

def generate_features_nn(stances,dataset,name):
    h, b, y = [],[],[]

    for stance in stances:
        y.append(LABELS.index(stance['Stance']))
        h.append(stance['Headline'])
        b.append(dataset.articles[stance['Body ID']])

    X_glove = gen_or_load_feats(glove_features, h, b, "features/glove_features."+name+".npy")

    return X_glove,y


if __name__ == "__main__":

    #params = { 'n_folds' : 10, 'size' : 1.0, 'n_estimators' : 200}
    params = { 'n_folds' : 5, 'size' : 0.2, 'n_estimators' : 25}
    check_version()
    parse_params()

    time_1 = time.time()
    d = DataSet()
    time_2 = time.time()
    print('Dataset load: '+str(time_2-time_1))

    folds,hold_out = kfold_split(d,n_folds=params['n_folds'],size=params['size'])
    time_3 = time.time()
    print('Kfold_split: '+str(time_3-time_2))

    fold_stances, hold_out_stances = get_stances_for_folds(d,folds,hold_out)
    time_4 = time.time()
    print('Get stances: '+str(time_4-time_3))

    Xs, Xnn = dict()
    ys, ynn = dict()

    # Load/Precompute all features now
    X_holdout,y_holdout = generate_features(hold_out_stances,d,"holdout")
    Xnn_holdout,ynn_holdout = generate_features_nn(hold_out_stances,d,"holdout")
    for fold in fold_stances:
        Xs[fold],ys[fold] = generate_features(fold_stances[fold],d,str(fold))
        Xnn[fold],ynn[fold] = generate_features_nn(fold_stances[fold],d,str(fold))
    time_5 = time.time()
    print('Generate features: '+str(time_5-time_4))

    best_score = 0
    best_fold = None


    # Classifier for each fold
    for fold in fold_stances:
        ids = list(range(len(folds)))
        del ids[fold]

        X_train = np.vstack(tuple([Xs[i] for i in ids]))
        y_train = np.hstack(tuple([ys[i] for i in ids]))

        X_test = Xs[fold]
        y_test = ys[fold]

        clf = GradientBoostingClassifier(n_estimators=params['n_estimators'], random_state=14128)
        clf.fit(X_train, y_train)
        

        predicted = [LABELS[int(a)] for a in clf.predict(X_test)]
        actual = [LABELS[int(a)] for a in y_test]

        fold_score, _ = score_submission(actual, predicted)
        max_fold_score, _ = score_submission(actual, actual)

        score = fold_score/max_fold_score

        print("Score for fold "+ str(fold) + " was - " + str(score))
        if score > best_score:
            best_score = score
            best_fold = clf

    time_6 = time.time()
    print('Train classifier: '+str(time_6-time_5))

    learning_rate = 0.0001
    batch_size = 70
    input_size = 100
    target_size = 4
    max_seq_length = len(Xs[0])
    vocab_size = 50
    model, placeholders = get_model_conditional(batch_size, max_seq_length, input_size, 
        hidden_size, target_size, vocab_size, pretrain, tanhOrSoftmax, dropout)


    #Run on Holdout set and report the final score on the holdout set
    predicted = [LABELS[int(a)] for a in best_fold.predict(X_holdout)]
    actual = [LABELS[int(a)] for a in y_holdout]

    report_score(actual,predicted)
