#########################################
# Calculate validation stats
#
# Note that I made an error by not saving the original test set
# Validation statistics will be over-inflated due to the mix of training
# and test data
#########################################

import pickle
import tensorflow as tf
import numpy as np
from sklearn.metrics import f1_score, accuracy_score, \
    precision_score, recall_score, confusion_matrix

EXPORT_DIR = '/Users/kailinlu/Documents/emotional-speech/final_model'
TEST_SIZE = 0.2

def predict(val_x, val_len):
    with tf.Session(graph=tf.Graph()) as sess:
        tf.saved_model.loader.load(sess, ['serve'], EXPORT_DIR)
        graph = tf.get_default_graph()

        x = graph.get_tensor_by_name('inputs/Placeholder:0')
        seq_len = graph.get_tensor_by_name('inputs/Placeholder_2:0')

        logits = graph.get_tensor_by_name('dense/dense_1/Softmax:0')
        feed_dict = {x: val_x, seq_len: val_len}
        logits = sess.run(logits, feed_dict=feed_dict)
        return np.argmax(logits, axis=1)


if __name__=='__main__':


    with open('/Users/kailinlu/Documents/emotional-speech/pickle/data.p','rb') as f:
        data = pickle.load(f)

    with open('/Users/kailinlu/Documents/emotional-speech/pickle/labels.p','rb') as f:
        labels = pickle.load(f)

    x = data[0]
    lengths = data[1]
    y = labels[0]
    int_y = labels[1]

    pred = predict(x, lengths)
    print('Accuracy: ', accuracy_score(int_y, pred))
    print('F1: ', f1_score(int_y, pred, average='macro'))
    print('Precision: ', precision_score(int_y, pred, average='macro'))
    print('Recall: ', recall_score(int_y, pred, average='macro'))
    print(confusion_matrix(int_y, pred))