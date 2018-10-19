import tensorflow as tf


def blstm(index, num_hidden, input_x, seq_len, return_all=False):
    """
    Bidirectional LSTM layer
    Input shape [batch_size, seq_length, embedding_dimension]
    Output shape [batch_size, seq_length, embedding_dimension]
    :return: BLSTM concatenated outputs
    """

    with tf.variable_scope('blstm_{}'.format(index)):
        cell_fw = tf.nn.rnn_cell.LSTMCell(num_hidden, state_is_tuple=True)
        cell_bw = tf.nn.rnn_cell.LSTMCell(num_hidden, state_is_tuple=True)
        outputs, states = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw,
                                                          inputs=input_x,
                                                          dtype=tf.float32,
                                                          sequence_length=seq_len)
        concat = tf.concat(outputs, 2)
    if return_all:
        concat = tf.transpose(concat, [1, 0, 2])[-1]
    print('BLSTM-{}'.format(index), concat.shape)
    return concat


    return tf.concat((even, odd), axis=2)