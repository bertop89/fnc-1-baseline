import tensorflow as tf
import numpy as np
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
    inputs_cond = tf.placeholder(tf.int32, [batch_size, 500], "inputs_cond")
    inputs = tf.placeholder(tf.int32, [batch_size, 50], "inputs")

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
    outputs, states = lstm_encoder(inputs_list, state, "Encoder1"+str(i))

    # running a second LSTM conditioned on the last state of the first
    lstm_encoder2 = Encoder(BasicLSTMCell, input_size, hidden_size, drop_prob, drop_prob)
    
    outputs_cond, states_cond = lstm_encoder2(inputs_cond_list, states, "Encoder2"+str(i))

    outputs_fin = outputs_cond[-1]
    if tanhOrSoftmax == "tanh":
        model = Projector(target_size, non_linearity=tf.nn.tanh, bias=True)(outputs_fin, 'proj'+str(i)) #tf.nn.softmax
    else:
        model = Projector(target_size, non_linearity=tf.nn.softmax, bias=True)(outputs_fin, 'proj'+str(i))  # tf.nn.softmax

    tf.add_to_collection("model", model)

    return model, [inputs, inputs_cond]

def test_trainer(headlines, bodies, labels, headlines_test, bodies_test, labels_test, hidden_size, max_epochs, tanhOrSoftmax, dropout,i):

    # parameters
    learning_rate = 0.0001
    batch_size = 70
    input_size = 50
    target_size = 3

    max_seq_length = len(bodies[0])
    max_seq_length_h = len(headlines[0])

    data = [np.asarray(headlines), np.asarray(bodies), np.asarray(labels)]

    X = get_glove_matrix()
    vocab_size = 400000

    model, placeholders = get_model_conditional(batch_size, max_seq_length, max_seq_length_h, input_size, hidden_size, target_size,
                                                                   vocab_size, "pre_cont", tanhOrSoftmax, dropout,i)

    targets = tf.placeholder(tf.float32, [batch_size, target_size], "targets")
    tf.add_to_collection("targets", targets)

    loss = tf.nn.softmax_cross_entropy_with_logits(logits=model, labels=targets)   # targets: labels (e.g. pos/neg/neutral)

    optimizer = tf.train.AdamOptimizer(learning_rate)

    batcher = BatchBucketSampler(data, batch_size)
    acc_batcher = BatchBucketSampler(data, batch_size)

    placeholders += [targets]

    pad_nr = batch_size - (
    len(labels_test) % batch_size) + 1  # since train/test batches need to be the same size, add padding for test

    data_test = [np.asarray(headlines_test), np.asarray(bodies_test), np.asarray(labels_test)]

    corpus_test_batch = BatchBucketSampler(data_test, batch_size)

    outfolder = "_".join(["hidden-" + str(hidden_size), tanhOrSoftmax])


    with tf.Session() as sess:

        summary_writer = tf.summary.FileWriter("rnn/save", graph_def=sess.graph_def)

        hooks = [
            SpeedHook(summary_writer, iteration_interval=50, batch_size=batch_size),
            SaveModelHookDev(path="rnn/save/" + outfolder, at_every_epoch=1),
            #SemEvalHook(corpus_test_batch, placeholders, 1),
            LossHook(summary_writer, iteration_interval=50),
            AccuracyHook(summary_writer, acc_batcher, placeholders, 2),
            #AccuracyHookIgnoreNeutral(summary_writer, acc_batcher, placeholders, 2)
        ]

        trainer = Trainer(optimizer, max_epochs, hooks)
        epoch = trainer(batcher=batcher, acc_thresh=0.9, pretrain='pre_cont', embedd=X, placeholders=placeholders,
                        loss=loss, model=model, sep=False)

        print("Applying to test data, getting predictions for NONE/AGAINST/FAVOR")

        predictions_all = []
        truth_all = []

        load_model_dev(sess, "rnn/save/" + outfolder + "_ep" + str(epoch), "model.tf")

        total = 0
        correct = 0
        for values in corpus_test_batch:
            total += len(values[-1])
            feed_dict = {}
            for i in range(0, len(placeholders)):
                batch_xs = np.vstack([np.expand_dims(x, 0) for x in values[i]])
                feed_dict[placeholders[i]] = batch_xs
            truth = np.argmax(values[-1], 1)  # values[2] is a 3-length one-hot vector containing the labels
            predicted = sess.run(tf.arg_max(tf.nn.softmax(model), 1),
                                     feed_dict=feed_dict)
            predictions_all.extend(predicted)
            truth_all.extend(truth)
            correct += sum(truth == predicted)

        print("Num testing samples " + str(total) +
                      "\tAcc " + str(float(correct)/total) +
                      "\tCorrect " + str(correct) + "\tTotal " + str(total))        

            
    predicted = [LABELS[int(a)] for a in predictions_all]
    actual = [LABELS[int(a)] for a in truth_all]

    return predicted,actual


def predict_holdout(headlines_holdout, bodies_holdout, labels_holdout, tanhOrSoftmax, hidden_size):

    batch_size = 70

    outfolder = "_".join(["hidden-" + str(hidden_size), tanhOrSoftmax])

    data_test = [np.asarray(headlines_holdout), np.asarray(bodies_holdout), np.asarray(labels_holdout)]

    corpus_test_batch = BatchBucketSampler(data_test, batch_size)

    with tf.Session() as sess:

        load_model_holdout(sess, "rnn/save/" + outfolder + "_ep" + str(1), "model.tf")

        model = tf.get_collection("model")[0]

        placeholders = [
            tf.get_collection("inputs")[0],
            tf.get_collection("inputs_cond")[0],
            tf.get_collection("targets")[0]
        ]

        total = 0
        correct = 0

        predictions_all = []
        truth_all = []

        for values in corpus_test_batch:

            total += len(values[-1])
            feed_dict = {}
            for i in range(0, len(placeholders)):
                batch_xs = np.vstack([np.expand_dims(x, 0) for x in values[i]])
                feed_dict[placeholders[i]] = batch_xs

            truth = np.argmax(values[-1], 1)  # values[2] is a 3-length one-hot vector containing the labels
            predicted = sess.run(tf.arg_max(tf.nn.softmax(model), 1),
                                     feed_dict=feed_dict)

            print(predicted[0])
            predictions_all.extend(predicted)
            truth_all.extend(truth)
            correct += sum(truth == predicted)

            print("Num testing samples " + str(total) +
                          "\tAcc " + str(float(correct)/total) +
                          "\tCorrect " + str(correct) + "\tTotal " + str(total))

            if (total > 1000):
                break

    predicted = [LABELS[int(a)] for a in predictions_all]
    actual = [LABELS[int(a)] for a in truth_all]

    return predicted,actual