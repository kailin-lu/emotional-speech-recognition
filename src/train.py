# Train emotion recognition model

import pickle
import datetime
import random
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from model import ClassifyEmotion

BATCH_SIZE = 16
EPOCHS = 30
LR = 5e-5
VERBOSE = True
TEST_SIZE = 0.2
EXPORT_DIR = '/Users/kailinlu/Documents/emotional-speech/final_model'

def train(train_x, train_y, train_lengths,
          val_x, val_y, val_lengths,
          epochs=EPOCHS,
          verbose=VERBOSE,
          lr=LR,
          save_path=EXPORT_DIR):
    tf.reset_default_graph()
    tf.set_random_seed(0)

    model = ClassifyEmotion(lr=lr)

    x, y, seq_len, logits, loss = model._build_model()
    step = model._step(loss)
    accuracy = model._accuracy(logits, y)

    # Summary ops for loss and accuracy
    tf.summary.scalar('loss', loss)
    tf.summary.scalar('accuracy', accuracy)
    merged = tf.summary.merge_all()

    run = datetime.datetime.now().strftime("%Y%m%d%H%M%S")

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    init = tf.global_variables_initializer()
    with tf.Session(config=config) as sess:
        sess.run(init)
        # Create summary writers for training and testing variables
        train_writer = tf.summary.FileWriter('/tmp/emo/train/' + run,
                                             sess.graph)
        test_writer = tf.summary.FileWriter('/tmp/emo/test/' + run)


        for epoch in range(epochs):
            # Shuffle batches
            batches = list(zip(train_x, train_lengths, train_y))
            random.shuffle(batches)
            train_x, train_lengths, train_y = zip(*batches)

            batch_losses = []
            batch_accuracy = []
            for i, batch in enumerate(train_x):
                len_batch = train_lengths[i]
                y_batch = train_y[i]
                feed_dict = {
                    x: batch,
                    y: y_batch,
                    seq_len: len_batch
                }
                _ = sess.run(step, feed_dict=feed_dict)
                err, acc = sess.run([loss, accuracy], feed_dict=feed_dict)
                batch_losses.append(err)
                batch_accuracy.append(acc)

            summary = sess.run(merged, feed_dict=feed_dict)
            train_writer.add_summary(summary, epoch)

            val_errors = []
            val_accuracy = []
            for i, val_batch in enumerate(val_x):
                val_feed_dict = {
                    x: val_batch,
                    y: val_y[i],
                    seq_len: val_lengths[i]
                }
            summary, val_err, val_acc = sess.run([merged, loss, accuracy],
                                        feed_dict=val_feed_dict)
            val_errors.append(val_err)
            val_accuracy.append(val_acc)
            test_writer.add_summary(summary, epoch)
            print('Epoch {} '
                  'mean loss {:4f} '
                  'mean acc {:4f} '
                  'val mean loss {:4f} '
                  'val mean acc {:4f}'.format(epoch, np.mean(batch_losses),
                                              np.mean(batch_accuracy),
                                              np.mean(val_errors),
                                              np.mean(val_accuracy)))

        tf.saved_model.simple_save(sess, export_dir=save_path,
                                   inputs={'x':x, 'seq_len': seq_len},
                                   outputs={'y':y})



def batch(data, batch_size=BATCH_SIZE):
    return [data[i:i + batch_size] for i in range(0, len(data), batch_size)]


if __name__ == '__main__':
    with open('/Users/kailinlu/Documents/emotional-speech/pickle/data.p','rb') as f:
        data = pickle.load(f)
    with open('/Users/kailinlu/Documents/emotional-speech/pickle/labels.p','rb') as f:
        labels = pickle.load(f)

    print(labels[2])

    # Split into train test sets
    x = data[0]
    lengths = data[1]
    y = labels[0]

    train_x, val_x, train_lengths, val_lengths, train_y, val_y = train_test_split(
        x, lengths, y, test_size=TEST_SIZE)

    # Batch data
    batched_x = batch(train_x)
    batched_lengths = batch(train_lengths)
    batched_y = batch(train_y)

    # Batch validation for resource management
    val_batched_x = batch(val_x)
    val_batched_lengths = batch(val_lengths)
    val_batched_y = batch(val_y)

    assert len(batched_x) == len(
        batched_y), 'Data and labels must have same size.'
    assert len(batched_x) == len(
        batched_y), 'Data and lengths must have the same size.'

    print('Created {} batches of size {}'.format(len(batched_x), BATCH_SIZE))
    print('Created Validation {} batches of size {}'.format(
        len(val_batched_x), BATCH_SIZE))
    print('Data Batch Shape: {} Labels Shape: {} '.format(
        batched_x[0].shape,
        batched_y[0].shape))

    train(train_x=batched_x, train_lengths=batched_lengths, train_y=batched_y,
          val_x=val_batched_x, val_lengths=val_batched_lengths, val_y=val_batched_y)