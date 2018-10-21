import tensorflow as tf


class ClassifyEmotion():
    """
    Classify utterances into emotions
    3 Bi-directional LSTM layers followed by a dense softmax
    Model architecture based off of emotion recognition paper by Vladimir Chernykh
    https://github.com/vladimir-chernykh/emotion_recognition/tree/master/code/notebooks
    """

    def __init__(self, num_features=39, num_classes=7, num_hidden=128,
                 batch_size=16, max_length=298, dense_hidden=64, lr=1e-4):
        self.num_features = num_features
        self.num_classes = num_classes
        self.num_hidden = num_hidden
        self.dense_hidden = dense_hidden
        self.learning_rate = lr

    def _build_model(self):
        with tf.name_scope('inputs'):
            x = tf.placeholder(shape=(None, None, self.num_features),
                               dtype=tf.float32, name='x')
            y = tf.placeholder(shape=(None, self.num_classes), dtype=tf.float32, name='y')
            seq_len = tf.placeholder(shape=(None), dtype=tf.int32, name='seq_len')

        concat_lstm1 = blstm(index=0,
                             num_hidden=self.num_hidden,
                             input_x=x,
                             seq_len=seq_len,
                             return_all=True)

        concat_lstm2 = blstm(index=1,
                             num_hidden=self.num_hidden,
                             seq_len=seq_len,
                             input_x=concat_lstm1,
                             return_all=False)

        with tf.name_scope('dense'):
            dense_0 = tf.layers.dense(concat_lstm2, self.dense_hidden,
                                      activation=tf.nn.tanh)
            print(dense_0.shape)
            logits = tf.layers.dense(dense_0, self.num_classes,
                                     activation=tf.nn.softmax,
                                     name='logits')
            print(logits.shape)

        with tf.name_scope('loss'):
            loss = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits_v2(labels=y,
                                                           logits=logits))
        return x, y, seq_len, logits, loss

    def _step(self, loss):
        return tf.train.AdamOptimizer(self.learning_rate).minimize(loss)

    def _accuracy(self, logits, labels):
        with tf.name_scope('accuracy'):
            correct = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
            return tf.reduce_mean(tf.cast(correct, tf.float32))


def blstm(index, num_hidden, input_x, seq_len, keep_prob=0.9, return_all=False):
    """
    Bidirectional LSTM layer
    Input shape [batch_size, seq_length, embedding_dimension]
    Output shape [batch_size, seq_length, embedding_dimension]
    :return: BLSTM concatenated outputs
    """

    with tf.variable_scope('blstm_{}'.format(index)):
        cell_fw = tf.nn.rnn_cell.LSTMCell(num_hidden, state_is_tuple=True)
        fw_dropout = tf.contrib.rnn.DropoutWrapper(cell_fw,
                                                   input_keep_prob=keep_prob)
        cell_bw = tf.nn.rnn_cell.LSTMCell(num_hidden, state_is_tuple=True)
        bw_dropout = tf.contrib.rnn.DropoutWrapper(cell_bw,
                                                   input_keep_prob=keep_prob)
        outputs, states = tf.nn.bidirectional_dynamic_rnn(fw_dropout, bw_dropout,
                                                          inputs=input_x,
                                                          dtype=tf.float32,
                                                          sequence_length=seq_len)
    concat = tf.concat(outputs, 2)  # [batch_size, output_dim, timesteps]
    if return_all:
        print('BLSTM-{}'.format(index), concat.shape)
        return(concat)
    else:
        batch_range = tf.range(tf.shape(outputs[0])[0])
        indices = tf.stack([batch_range, seq_len-1], axis=1)
        fw_last = tf.gather_nd(outputs[0], indices)
        bw_last = tf.gather_nd(outputs[1], indices)
        concat_last = tf.concat((fw_last, bw_last), 1)
        print('BLSTM-{}'.format(index), concat_last.shape)
        return(concat_last)


# if __name__=='__main__':
#     model = ClassifyEmotion()
#     model._build_model()