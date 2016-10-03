import argparse
import csv
import matplotlib.pyplot as plt
import pickle
import random

import numpy as np
import tensorflow as tf

DATA_SPLIT = [0.8, 0.1, 0.1]


class Classifier:

    def __init__(self, data_dim, data_in, class_in, class_out, train_step, correct_out, accuracy_out):
        self.data_dim = data_dim
        self.data_in = data_in
        self.class_in = class_in
        self.class_out = class_out
        self.train_step = train_step
        self.correct_out = correct_out
        self.accuracy_out = accuracy_out


def weight_bias(shape):
    weight = tf.truncated_normal(shape, stddev=1e-2)
    bias = tf.constant(0.1, shape=[shape[1]])
    return tf.Variable(weight), tf.Variable(bias)


def partition_data(aligned_datasets, data_split):
    partitions = [[[] for _ in range(len(aligned_datasets))]
                  for _ in range(len(data_split))]
    assert len(aligned_datasets[0]) == len(
        max(aligned_datasets, key=len)) == len(min(aligned_datasets, key=len))
    for i in range(len(aligned_datasets[0])):
        placement = random.random()
        for q in range(len(data_split)):
            if placement < data_split[q]:
                for t in range(len(aligned_datasets)):
                    partitions[q][t].append(aligned_datasets[t][i])
                break
            else:
                placement = placement - data_split[q]
    return partitions


def create_linear_classifier(data_dim, learn_rate):
    data = tf.placeholder(tf.float32, shape=[None, data_dim])
    W, b = weight_bias([data_dim, 2])
    y = tf.nn.log_softmax(tf.matmul(data, W) + b)

    y_ = tf.placeholder(tf.float32, shape=[None, 2])
    cross_entropy = -tf.reduce_sum(y_ * y)
    correct = tf.equal(tf.argmax(y_, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

    train_step = tf.train.AdamOptimizer(learn_rate).minimize(
        cross_entropy
    )
    return Classifier(data_dim, data, y_, y, train_step, correct, accuracy)


def data_to_predictions(data, classifiers):
    classifications = [[] for _ in range(len(data))]
    for t in range(len(classifiers)):
        output = classifiers[t].class_out.eval(
            feed_dict={classifiers[t].data_in: data})
        for i in range(len(output)):
            classifications[i].append(np.argmax(output[i]))
    return classifications


def create_deep_ensemble_network(num_classifiers, num_layers, layer_width, learn_rate):
    num_layers = max(1, num_layers)
    input_predictions = tf.placeholder(
        tf.float32, shape=[None, num_classifiers])
    W, b = weight_bias([num_classifiers, layer_width])
    h = tf.nn.relu(tf.matmul(input_predictions, W) + b)
    for i in range(num_layers - 1):
        W, b = weight_bias([layer_width, layer_width])
        h = tf.nn.relu(tf.matmul(h, W) + b)
    W, b = weight_bias([layer_width, 2])
    y = tf.nn.log_softmax(tf.matmul(h, W) + b)

    y_ = tf.placeholder(tf.float32, shape=[None, 2])
    cross_entropy = -tf.reduce_sum(y_ * y)
    correct = tf.equal(tf.argmax(y_, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

    train_step = tf.train.AdamOptimizer(learn_rate).minimize(
        cross_entropy
    )
    return Classifier(num_classifiers, input_predictions, y_, y, train_step, correct, accuracy)


parser = argparse.ArgumentParser()
parser.add_argument("--epochs", type=int, default=100, help="number of epochs")
parser.add_argument("--clearn", type=float,
                    default=1e-2, help="classifier learning rate")
parser.add_argument("--dlearn", type=float,
                    default=1e-3, help="network learning rate")
parser.add_argument("--lwidth", type=int,
                    default=256, help="width of network layers")
parser.add_argument("--ldepth", type=int, default=4, help="depth of network")
parser.add_argument("--classifiers", type=int,
                    default=100, help="number of classifiers")
parser.add_argument("--batch", type=int, default=50, help="size of batches")
parser.add_argument("--cvision", type=float, default=2e-3,
                    help="portion of dataset seen by classifiers")
parser.add_argument("--cconverge", type=float,
                    default=1e-7, help="convergence margin for classifiers")
parser.add_argument(
    "--pklprefix", default="ensemble_", help="prefix for pickle files")
parser.add_argument("--datafile", help="file containing data")
parser.add_argument("--loadpredictions", action="store_true",
                    help="load predictions from pickle file")
parser.add_argument("--savebest", action="store_true", help="save best model")
parser.add_argument(
    "--saveeach", action="store_true", help="save model at each epoch")
args = parser.parse_args()

with tf.Session() as sess:
    x_ = []
    y_ = []
    print "Reading data from", args.datafile
    with open(args.datafile, 'rb') as data_file:
        reader = csv.reader(data_file, delimiter=',', quotechar='|')
        for row in reader:
            x_.append(map(float, row[:len(row) - 1]))
            y_.append([0, 1] if row[len(row) - 1] == "h" else [1, 0])
    data_dim = len(x_[0])
    print "Data read."

    print "Initializing networks."
    classifiers = [create_linear_classifier(data_dim, args.clearn)
                   for _ in range(args.classifiers)]
    ensemble_net = create_deep_ensemble_network(
        args.classifiers, args.ldepth, args.lwidth, args.dlearn)
    print "Networks initialized."

    sess.run(tf.initialize_all_variables())
    if args.loadpredictions is False:
        print "Splitting data into train, validation, and test sets."
        train_data, validation_data, test_data = partition_data(
            [x_, y_], DATA_SPLIT)
        print train_data[1][0:50]
        print "Data split."

        best_validation = 0
        best_test = 0
        print "Training classifiers."
        for classifier in classifiers:
            viewed_data_x, viewed_data_y = partition_data(
                train_data, [args.cvision, 1 - args.cvision])[0]
            prev_acc = float(0)
            acc = float('inf')
            while abs(acc - prev_acc) > args.cconverge:
                prev_acc = acc
                classifier.train_step.run(
                    feed_dict={classifier.data_in: viewed_data_x, classifier.class_in: viewed_data_y})
                acc = classifier.accuracy_out.eval(
                    feed_dict={classifier.data_in: viewed_data_x, classifier.class_in: viewed_data_y})
            valid_acc = classifier.accuracy_out.eval(
                feed_dict={classifier.data_in: validation_data[0], classifier.class_in: validation_data[1]})
            print "Classifier trained. Accuracy on validation set:", valid_acc
            if valid_acc > best_validation:
                best_validation = valid_acc
                best_test = classifier.accuracy_out.eval(
                    feed_dict={classifier.data_in: test_data[0], classifier.class_in: test_data[1]})
        print "All classifiers trained."
        print "Calculating baselines."
        majority_vote_correct = float(0)
        majority_vote_total = 0
        for i in range(len(test_data[0])):
            prediction = [1, 0] if sum(
                test_data[0][i]) > args.classifiers / 2.0 else [0, 1]
            if prediction == test_data[1][i]:
                majority_vote_correct += 1
            majority_vote_total += 1
        majority_vote_acc = majority_vote_correct / majority_vote_total
        print "Generating train predictions..."
        train_predictions = data_to_predictions(train_data[0], classifiers)
        print "Generating validation predictions..."
        validation_predictions = data_to_predictions(
            validation_data[0], classifiers)
        print "Generating test predictions..."
        test_predictions = data_to_predictions(test_data[0], classifiers)
        print "Pickling..."
        pickle.dump(majority_vote_acc, open(
            args.pklprefix + "majority_vote.p", "wb"))
        pickle.dump(best_test, open(args.pklprefix + "best_test.p", "wb"))
        pickle.dump(train_data, open(args.pklprefix + "train_data.p", "wb"))
        pickle.dump(validation_data, open(
            args.pklprefix + "validation_data.p", "wb"))
        pickle.dump(test_data, open(args.pklprefix + "test_data.p", "wb"))
        pickle.dump(train_predictions, open(
            args.pklprefix + "train_predictions.p", "wb"))
        pickle.dump(validation_predictions, open(
            args.pklprefix + "validation_predictions.p", "wb"))
        pickle.dump(test_predictions, open(
            args.pklprefix + "test_predictions.p", "wb"))
        print "Pickling complete."
    else:
        print "Loading data from pickle."
        majority_vote_acc = pickle.load(
            open(args.pklprefix + "majority_vote.p", "rb"))
        best_test = pickle.load(open(args.pklprefix + "best_test.p", "rb"))
        train_predictions = pickle.load(
            open(args.pklprefix + "train_predictions.p", "rb"))
        validation_predictions = pickle.load(
            open(args.pklprefix + "validation_predictions.p", "rb"))
        test_predictions = pickle.load(
            open(args.pklprefix + "test_predictions.p", "rb"))
        train_data = pickle.load(open(args.pklprefix + "train_data.p", "rb"))
        validation_data = pickle.load(
            open(args.pklprefix + "validation_data.p", "rb"))
        test_data = pickle.load(open(args.pklprefix + "test_data.p", "rb"))
        print "Loading complete."

    print "Majority vote accuracy on test data:", majority_vote_acc
    print "Best classifier on test data:", best_test
    train_accs = []
    valid_accs = []
    max_valid = 0
    for epoch in range(args.epochs):
        for t in range(0, len(train_predictions), args.batch):
            batch_x, batch_y = train_predictions[
                t:t + args.batch], train_data[1][t:t + args.batch]
            ensemble_net.train_step.run(
                feed_dict={ensemble_net.data_in: batch_x, ensemble_net.class_in: batch_y})

        train_acc = ensemble_net.accuracy_out.eval(
            feed_dict={ensemble_net.data_in: train_predictions, ensemble_net.class_in: train_data[1]})
        valid_acc = ensemble_net.accuracy_out.eval(
            feed_dict={ensemble_net.data_in: validation_predictions, ensemble_net.class_in: validation_data[1]})
        train_accs.append(train_acc)
        valid_accs.append(valid_acc)
        plt.plot(range(epoch + 1), train_accs, color='blue')
        plt.plot(range(epoch + 1), valid_accs, color='red')
        plt.legend(['train', 'valid'], loc='upper left')
        plt.draw()
        plt.show(block=False)
        print "epoch", epoch, "- train accuracy:", train_acc, "validation accuracy:", valid_acc
        if args.savebest and valid_acc > max_valid:
            max_valid = valid_acc
            saver = tf.train.Saver()
            saver.save(sess, "ensemble_best.chk")
        if args.saveeach:
            saver = tf.train.Saver()
            saver.save(sess, "ensemble_e" + str(epoch) + ".chk")
    test_acc = ensemble_net.accuracy_out.eval(
        feed_dict={ensemble_net.data_in: test_predictions, ensemble_net.class_in: test_data[1]})
    print "final test accuracy:", test_acc
    plt.show()
