import tensorflow as tf
import numpy as np
from tensorflow.models.rnn import rnn_cell

def get_model_conditional(batch_size, max_seq_length, input_size, hidden_size, target_size,
                          vocab_size, pretrain, tanhOrSoftmax, dropout):
    """
    Unidirectional conditional encoding model
    :param pretrain:  "pre": use pretrained word embeddings, "pre-cont": use pre-trained embeddings and continue training them, otherwise: random initialisation
    """

    # batch_size x max_seq_length
    inputs = tf.placeholder(tf.int32, [batch_size, max_seq_length])
    inputs_cond = tf.placeholder(tf.int32, [batch_size, max_seq_length])

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
                   tf.split(1, max_seq_length, embedded_inputs)]
    inputs_cond_list = [tf.squeeze(x) for x in
                        tf.split(1, max_seq_length, embedded_inputs_cond)]

    drop_prob = None
    if dropout:
        drop_prob = 0.1
    lstm_encoder = Encoder(rnn_cell.BasicLSTMCell, input_size, hidden_size, drop_prob, drop_prob)

    start_state = tf.zeros([batch_size, lstm_encoder.state_size])

    # [h_i], [h_i, c_i] <-- LSTM
    # [h_i], [h_i] <-- RNN
    outputs, states = lstm_encoder(inputs_list, start_state, "LSTM")

    # running a second LSTM conditioned on the last state of the first
    outputs_cond, states_cond = lstm_encoder(inputs_cond_list, states[-1],
                                             "LSTMcond")

    outputs_fin = outputs_cond[-1]
    if tanhOrSoftmax == "tanh":
        model = Projector(target_size, non_linearity=tf.nn.tanh, bias=True)(outputs_fin) #tf.nn.softmax
    else:
        model = Projector(target_size, non_linearity=tf.nn.softmax, bias=True)(outputs_fin)  # tf.nn.softmax

	return model, [inputs, inputs_cond]

