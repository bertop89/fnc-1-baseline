import tensorflow as tf
import numpy as np
import time
from feature_engineering import get_glove_matrix
from tensorflow.contrib.rnn import core_rnn_cell, BasicLSTMCell
from tensorflow.python.ops import variable_scope as vs
from utils.score import report_score, LABELS
from rnn.utils import Encoder, Projector, Hook, AccuracyHook, LossHook, SpeedHook, SemEvalHook, AccuracyHookIgnoreNeutral, BatchBucketSampler, TraceHook, SaveModelHookDev, Trainer, load_model_dev, load_model_holdout

def get_model_conditional(batch_size, max_seq_length, max_seq_length_h, input_size, hidden_size, target_size,
                          vocab_size, pretrain, tanhOrSoftmax, dropout,i):
    """
    Unidirectional conditional encoding model
    :param pretrain:  "pre": use pretrained word embeddings, "pre-cont": use pre-trained embeddings and continue training them, otherwise: random initialisation
    """

    # batch_size x max_seq_length
    inputs_cond = tf.placeholder(tf.int32, [batch_size, max_seq_length], "inputs_cond")
    inputs = tf.placeholder(tf.int32, [batch_size, max_seq_length_h], "inputs")

    tf.add_to_collection("inputs", inputs)
    tf.add_to_collection("inputs_cond", inputs_cond)

    cont_train = True
    if pretrain == "pre": # continue training embeddings or not. Currently works better to continue training them.
        cont_train = False
    embedding_matrix = tf.Variable(tf.random_uniform([vocab_size, input_size], -0.1, 0.1),  #input_size is embeddings size
                                   name="embedding_matrix", trainable=cont_train)

    # batch_size x max_seq_length x input_size
    embedded_inputs = tf.nn.embedding_lookup(embedding_matrix, inputs)
    embedded_inputs_cond = tf.nn.embedding_lookup(embedding_matrix, inputs_cond)

    # [batch_size x inputs_size] with max_seq_length elements
    # fixme: possibly inefficient
    # inputs_list[0]: batch_size x input[0] <-- word vector of the first word
    inputs_list = [tf.squeeze(x) for x in
                   tf.split(embedded_inputs, max_seq_length_h, 1)]
    inputs_cond_list = [tf.squeeze(x) for x in
                        tf.split(embedded_inputs_cond, max_seq_length, 1)]

    drop_prob = None
    if dropout:
        drop_prob = 0.1
    lstm_encoder = Encoder(BasicLSTMCell, input_size, hidden_size, drop_prob, drop_prob)
        ##drop_prob, drop_prob)

    state = lstm_encoder.zero_state(batch_size)

    # [h_i], [h_i, c_i] <-- LSTM
    # [h_i], [h_i] <-- RNN
    outputs, states = lstm_encoder(inputs_cond_list, state, "Encoder1"+str(i))

    # running a second LSTM conditioned on the last state of the first
    lstm_encoder2 = Encoder(BasicLSTMCell, input_size, hidden_size, drop_prob, drop_prob)
    
    outputs_cond, states_cond = lstm_encoder2(inputs_list, states, "Encoder2"+str(i))

    outputs_fin = outputs_cond[-1]
    if tanhOrSoftmax == "tanh":
        model = Projector(target_size, non_linearity=tf.nn.tanh, bias=True)(outputs_fin, 'proj'+str(i)) #tf.nn.softmax
    else:
        model = Projector(target_size, non_linearity=tf.nn.softmax, bias=True)(outputs_fin, 'proj'+str(i))  # tf.nn.softmax

    tf.add_to_collection("model", model)

    return model, [inputs, inputs_cond]

def test_trainer(headlines, bodies, labels, headlines_test, bodies_test, labels_test, hidden_size, max_epochs, tanhOrSoftmax, dropout,i):

    time_1 = time.time()

    # parameters
    learning_rate = 0.01
    batch_size = 70
    input_size = 100
    target_size = 3

    max_seq_length = len(bodies[0])
    max_seq_length_h = len(headlines[0])


    ids = np.vstack([np.expand_dims(x, 0) for x in list(range(1,len(bodies)+1))])
    data = [np.asarray(headlines), np.asarray(bodies), np.asarray(labels), np.asarray(ids)]

    X = get_glove_matrix()
    vocab_size = 400000

    model, placeholders = get_model_conditional(batch_size, max_seq_length, max_seq_length_h, input_size, hidden_size, target_size,
                                                                   vocab_size, "pre_cont", tanhOrSoftmax, dropout,i)

    ids = tf.placeholder(tf.float32, [batch_size, 1], "ids")
    targets = tf.placeholder(tf.float32, [batch_size, target_size], "targets")
    tf.add_to_collection("targets", targets)

    loss = tf.nn.softmax_cross_entropy_with_logits(logits=model, labels=targets)   # targets: labels (e.g. pos/neg/neutral)

    optimizer = tf.train.AdamOptimizer(learning_rate)

    batcher = BatchBucketSampler(data, batch_size)
    acc_batcher = BatchBucketSampler(data, batch_size)

    placeholders += [targets]
    placeholders += [ids]

    pad_nr = batch_size - (
    len(labels_test) % batch_size) + 1  # since train/test batches need to be the same size, add padding for test

    ids_test = np.vstack([np.expand_dims(x, 0) for x in list(range(1,len(labels_test)+1))])

    data_test = [np.lib.pad(headlines_test, ((0, pad_nr), (0, 0)), 'constant', constant_values=(0)),
                 np.lib.pad(bodies_test, ((0, pad_nr), (0, 0)), 'constant', constant_values=(0)),
                 np.lib.pad(labels_test, ((0, pad_nr), (0, 0)), 'constant', constant_values=(0)),
                 np.lib.pad(ids_test, ((0, pad_nr), (0, 0)), 'constant', constant_values=(0))]

    corpus_test_batch = BatchBucketSampler(data_test, batch_size)

    outfolder = "_".join(["hidden-" + str(hidden_size), tanhOrSoftmax])


    with tf.Session() as sess:

        summary_writer = tf.summary.FileWriter("rnn/save", graph=sess.graph)

        hooks = [
            SpeedHook(summary_writer, iteration_interval=50, batch_size=batch_size),
            SaveModelHookDev(path="rnn/save/" + outfolder, at_every_epoch=1),
            LossHook(summary_writer, iteration_interval=50),
            #AccuracyHook(summary_writer, acc_batcher, placeholders, 2),
            #AccuracyHookIgnoreNeutral(summary_writer, acc_batcher, placeholders, 2)
        ]

        trainer = Trainer(optimizer, max_epochs, hooks)
        epoch = trainer(batcher=batcher, acc_thresh=0.9, pretrain='pre_cont', embedd=X, placeholders=placeholders,
                        loss=loss, model=model, sep=False)

        print("Applying to test data, getting predictions for NONE/AGAINST/FAVOR")

        predictions_all = []
        truth_all = []
        ids_all = []

        load_model_dev(sess, "rnn/save/" + outfolder + "_ep" + str(epoch), "model.tf")

        total = 0
        correct = 0
        for values in corpus_test_batch:
            total += len(values[-1])
            feed_dict = {}
            for i in range(0, len(placeholders)):
                feed_dict[placeholders[i]] = values[i]
            truth = [3 if (x == [1,1,1]).all() else np.argmax(x) for x in values[-2]]  # values[2] is a 3-length one-hot vector containing the labels
            ids_all.extend(values[-1])
            predicted = sess.run(tf.arg_max(tf.nn.softmax(model), 1),
                                     feed_dict=feed_dict)
            predictions_all.extend(predicted)
            truth_all.extend(truth)
            correct += sum(truth == predicted)

        print("TEST Num testing samples " + str(total) +
                      "\tAcc " + str(float(correct)/total) +
                      "\tCorrect " + str(correct) + "\tTotal " + str(total) + "\tTime " + str((time.time()-time_1)/60))        

            
    predicted = [LABELS[int(a)] for a in predictions_all]
    actual = [LABELS[int(a)] for a in truth_all]

    return predicted,actual


def predict_holdout(headlines_holdout, bodies_holdout, labels_holdout, tanhOrSoftmax, hidden_size, epoch):

    batch_size = 70

    outfolder = "_".join(["hidden-" + str(hidden_size), tanhOrSoftmax])

    ids_holdout = list(range(1,len(labels_holdout)+1))

    pad_nr = batch_size - (
    len(labels_holdout) % batch_size) + 1  # since train/test batches need to be the same size, add padding for test

    data_test = [np.lib.pad(np.vstack([np.expand_dims(x, 0) for x in headlines_holdout]), ((0, pad_nr), (0, 0)), 'constant', constant_values=(0)),
                 np.lib.pad(np.vstack([np.expand_dims(x, 0) for x in bodies_holdout]), ((0, pad_nr), (0, 0)), 'constant', constant_values=(0)),
                 np.lib.pad(np.asarray(labels_holdout), ((0, pad_nr), (0, 0)), 'constant', constant_values=(0)),
                 np.lib.pad(np.vstack([np.expand_dims(x, 0) for x in ids_holdout]), ((0, pad_nr), (0, 0)), 'constant', constant_values=(0))]

    corpus_test_batch = BatchBucketSampler(data_test, batch_size)

    with tf.Session() as sess:

        load_model_holdout(sess, "rnn/save/" + outfolder + "_ep" + str(epoch-1), "model.tf")

        model = tf.get_collection("model")[0]

        placeholders = [
            tf.get_collection("inputs")[0],
            tf.get_collection("inputs_cond")[0],
            tf.get_collection("targets")[0],
            tf.placeholder(tf.float32, [batch_size, 1], "ids")
        ]

        total = 0
        correct = 0

        predictions_all = []
        truth_all = []
        ids_all = []

        for values in corpus_test_batch:
            total += len(values[-1])
            feed_dict = {}
            for i in range(0, len(placeholders)):
                feed_dict[placeholders[i]] = values[i]

            truth = [3 if (x == [1,1,1]).all() else np.argmax(x) for x in values[-2]]  # values[2] is a 3-length one-hot vector containing the labels
            ids_all.extend(values[-1])
            predicted = sess.run(tf.arg_max(tf.nn.softmax(model), 1),
                                     feed_dict=feed_dict)
            predictions_all.extend(predicted)
            truth_all.extend(truth)
            correct += sum(truth == predicted)

            print("HOLDOUT Num testing samples " + str(total) +
                          "\tAcc " + str(float(correct)/total) +
                          "\tCorrect " + str(correct) + "\tTotal " + str(total))

    final_predicted = []
    final_actual = []
    for idx, val in enumerate(predictions_all):
        if ids_all[idx] != 0:
            final_predicted.append(val)
            final_actual.append(truth_all[idx])

    predicted = [LABELS[int(a)] for a in final_predicted]
    actual = [LABELS[int(a)] for a in final_actual]

    return predicted,actual

def predict_submission(headlines, bodies, ids, tanhOrSoftmax, hidden_size, epoch):

    batch_size = 70

    outfolder = "_".join(["hidden-" + str(hidden_size), tanhOrSoftmax])

    labels_holdout = np.tile([0,0,0],(len(headlines),1))
    

    pad_nr = batch_size - (
    len(headlines) % batch_size) + 1  # since train/test batches need to be the same size, add padding for test

    data_test = [np.lib.pad(np.vstack([np.expand_dims(x, 0) for x in headlines]), ((0, pad_nr), (0, 0)), 'constant', constant_values=(0)),
                 np.lib.pad(np.vstack([np.expand_dims(x, 0) for x in bodies]), ((0, pad_nr), (0, 0)), 'constant', constant_values=(0)),
                 np.lib.pad(np.asarray(labels_holdout), ((0, pad_nr), (0, 0)), 'constant', constant_values=(0)),
                 np.lib.pad(np.vstack([np.expand_dims(x, 0) for x in ids]), ((0, pad_nr), (0, 0)), 'constant', constant_values=(0))]

    corpus_test_batch = BatchBucketSampler(data_test, batch_size)

    with tf.Session() as sess:

        load_model_holdout(sess, "rnn/save/" + outfolder + "_ep" + str(epoch-1), "model.tf")

        model = tf.get_collection("model")[0]

        placeholders = [
            tf.get_collection("inputs")[0],
            tf.get_collection("inputs_cond")[0],
            tf.get_collection("targets")[0],
            tf.placeholder(tf.float32, [batch_size, 1], "ids")
        ]

        total = 0

        predictions_all = []
        ids_all = []

        for values in corpus_test_batch:
            total += len(values[-1])
            feed_dict = {}
            for i in range(0, len(placeholders)):
                feed_dict[placeholders[i]] = values[i]
            ids_all.extend(values[-1])
            predicted = sess.run(tf.arg_max(tf.nn.softmax(model), 1),
                                     feed_dict=feed_dict)
            predictions_all.extend(predicted)
            print(ids_all)
            print(predictions_all)
            print('PREDICTIONS: '+str(total))

    final_predicted = {}
    for idx, val in enumerate(predictions_all):
        if ids_all[idx][0] != 0:
            final_predicted[ids_all[idx][0]] = val

    return final_predicted