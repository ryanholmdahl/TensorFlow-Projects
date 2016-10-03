import argparse
import os
import string
import sys

import numpy as np
import tensorflow as tf


class EncoderModel:

    def __init__(self, document_in, style_out):
        self.document_in = document_in
        self.style_out = style_out


class DecoderModel:

    def __init__(self, encoder, e_out, p_out, reg_loss, p_loss, loss_out, train_step, learning_rate):
        self.encoder = encoder
        self.e_out = e_out
        self.p_out = p_out
        self.reg_loss = reg_loss
        self.p_loss = p_loss
        self.loss_out = loss_out
        self.train_step = train_step
        self.learning_rate = learning_rate


class DocumentReader:

    def __init__(self):
        self.documents = {"train": [], "validation": [], "test": []}
        self.vocab = {}
        self.vocab_counts = {}
        self.inv_vocab = []
        self.document_iterator = {"train": 0, "validation": 0, "test": 0}

    def read_vocab(self, file_path):
        with open(file_path, "rb") as file:
            text = set(file.read().split())
            self.text_to_vocab(string.join(text, " "))

    def text_to_vocab(self, text):
        for token in text.split():
            self.add_token_to_vocab(token)

    def add_token_to_vocab(self, token):
        if token in self.vocab:
            self.vocab_counts[token] += 1
        else:
            self.vocab[token] = len(self.vocab)
            self.inv_vocab.append(token)
            self.vocab_counts[token] = 1

    def remove_low_freq_vocab(self, min_count):
        self.inv_vocab = []
        for token in self.vocab:
            if self.vocab_counts[token] >= min_count:
                self.inv_vocab.append(token)
        self.vocab = dict()
        for i in range(len(self.inv_vocab)):
            self.vocab[self.inv_vocab[i]] = i

    def next_batch(self, batch_size, document_type):
        if document_type not in self.documents:
            raise Exception("Document type not recognized!")
        counts = []
        for i in range(batch_size):
            count = self.documents[document_type][
                self.document_iterator[document_type]]
            self.document_iterator[document_type] += 1
            if self.document_iterator[document_type] >= len(self.documents[document_type]):
                self.document_iterator[document_type] = 0
            counts.append(count)
        return counts

    def read_document(self, file_path, document_type):
        with open(file_path, "rb") as file:
            text = file.read()
        self.text_to_doc(text, document_type)

    def text_to_doc(self, text, document_type):
        if document_type not in self.documents:
            raise Exception("Document type not recognized!")
        document_sequence = []
        for token in text.split():
            if token in self.vocab:
                document_sequence.append(self.vocab[token])
        self.documents[document_type].append(
            np.ndarray.tolist(np.bincount(document_sequence, minlength=len(self.inv_vocab))))


def weight_bias(shape):
    weight = tf.truncated_normal(shape, stddev=1e-2)
    bias = tf.constant(0.1, shape=[shape[1]])
    return tf.Variable(weight), tf.Variable(bias)


def build_encoder(vocab_size, style_size, layer_width, num_layers):
    document_in = tf.placeholder(tf.float32, shape=[None, vocab_size])
    W, b = weight_bias([vocab_size, layer_width])
    h = tf.nn.relu(tf.matmul(document_in, W) + b)
    for i in range(num_layers - 1):
        W, b = weight_bias([layer_width, layer_width])
        h = tf.nn.relu(tf.matmul(h, W) + b)
    W_style, b_style = weight_bias([layer_width, style_size])
    style_out = tf.matmul(h, W_style) + b_style
    return EncoderModel(document_in, style_out)


def build_decoder(vocab_size, style_size, encoder):
    semantic_embedding_R, semantic_embedding_b = weight_bias(
        [style_size, vocab_size])
    e_out = - \
        tf.matmul(encoder.style_out, semantic_embedding_R) + \
        semantic_embedding_b
    p_out = tf.squeeze(tf.nn.softmax(e_out))
    p_loss = - \
        tf.reduce_mean(
            tf.log(tf.mul(p_out, tf.squeeze(encoder.document_in)) + 1e-10))

    reg_loss = args.rlambda * tf.reduce_mean(tf.square(encoder.style_out))
    loss_out = p_loss + reg_loss
    learning_rate = tf.placeholder(tf.float32, shape=[])
    train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss_out)
    return DecoderModel(encoder, e_out, p_out, reg_loss, p_loss, loss_out, train_step, learning_rate)


def get_losses(decoder, num_documents, document_type):
    p_loss_sum = 0
    reg_loss_sum = 0
    onoff_delta_sum = 0
    for t in range(num_documents):
        count_valid = reader.next_batch(1, document_type)
        p_loss_sum += decoder.p_loss.eval(
            feed_dict={encoder.document_in: count_valid})
        reg_loss_sum += decoder.reg_loss.eval(
            feed_dict={encoder.document_in: count_valid})
        p_out = decoder.p_out.eval(
            feed_dict={encoder.document_in: count_valid})
        count_valid_onehot = [
            1 if entry > 0 else 0 for entry in count_valid[0]]
        count_valid_onehot_inv = [
            0 if entry > 0 else 1 for entry in count_valid[0]]
        avg_on = sum(np.multiply(p_out, count_valid_onehot)) / sum(
            count_valid_onehot)
        avg_off = sum(np.multiply(p_out, count_valid_onehot_inv)) / sum(
            count_valid_onehot_inv)
        onoff_delta_sum += avg_on / avg_off
    return p_loss_sum / num_documents, reg_loss_sum / num_documents, onoff_delta_sum / num_documents


parser = argparse.ArgumentParser()
parser.add_argument("--epochs", type=int, default=100, help="number of epochs")
parser.add_argument("--sdim", type=int,
                    default=50, help="dimension of style representation")
parser.add_argument("--rlambda", type=float,
                    default=1, help="regularization constant")
parser.add_argument("--lwidth", type=int,
                    default=256, help="width of network layers")
parser.add_argument("--ldepth", type=int,
                    default=4, help="depth of encoder network")
parser.add_argument("--traindir", help="root path of training files")
parser.add_argument("--batch", type=int, default=50, help="size of batches")
parser.add_argument("--testdir",
                    help="root path of testing files")
parser.add_argument("--validdir", help="root path of validation files")
parser.add_argument(
    "--learnrate", type=float, default=1e-4, help="network learning rate")
parser.add_argument("--minvocab", type=int, default=25,
                    help="minimum instances of word required for use in vocab")
parser.add_argument("--savebest", action="store_true", help="save best model")
parser.add_argument(
    "--saveeach", action="store_true", help="save model at each epoch")
args = parser.parse_args()

if None in (args.traindir, args.validdir, args.testdir):
    print "NDM.py requires --trainDir= --validDir=, --testDir="
    sys.exit(1)

reader = DocumentReader()
print "Reading vocab from", args.traindir
for root, subFolders, files in os.walk(args.traindir):
    for file in files:
        reader.read_vocab(os.path.join(root, file))
initial_vocab_length = len(reader.vocab)
reader.remove_low_freq_vocab(args.minvocab)
print "Vocab created. Original length:", initial_vocab_length, "tokens. Reduced length:", len(reader.vocab)

print "Reading training data from", args.traindir
for root, subFolders, files in os.walk(args.traindir):
    for file in files:
        reader.read_document(os.path.join(root, file), "train")
nTrain = len(reader.documents["train"])
print "Done reading training data. Total train files:", nTrain

print "Reading validation data from", args.validdir
for root, subFolders, files in os.walk(args.validdir):
    for file in files:
        reader.read_document(os.path.join(root, file), "validation")
nValid = len(reader.documents["validation"])
print "Done reading validation data. Total validation files:", nValid

print "Reading testing data from", args.testdir
for root, subFolders, files in os.walk(args.testdir):
    for file in files:
        reader.read_document(os.path.join(root, file), "test")
nTest = len(reader.documents["test"])
print "Done reading test data. Total test files:", nTest

print "Building encoder."
encoder = build_encoder(
    len(reader.inv_vocab), args.sdim, args.lwidth, args.ldepth)
print "Building decoder."
decoder = build_decoder(len(reader.inv_vocab), args.sdim, encoder)

with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())
    min_loss = float('inf')
    print "Beginning training."
    for epoch in range(args.epochs):
        for i in range(0, nTrain, args.batch):
            count = reader.next_batch(args.batch, "train")
            decoder.train_step.run(
                feed_dict={encoder.document_in: count, decoder.learning_rate: args.learnrate})
        p_loss, reg_loss, onoff_delta = get_losses(
            decoder, nValid, "validation")
        print "validation, epoch", epoch, "- prediction loss:", p_loss, "regularization loss:", reg_loss, "on-off delta:", onoff_delta
        if args.saveeach:
            saver = tf.train.Saver()
            saver.save(sess, "NDM_e" + str(epoch) + ".chk")
        if args.savebest and p_loss + reg_loss < min_loss:
            min_loss = p_loss + reg_loss
            saver = tf.train.Saver()
            saver.save(sess, "NDM_best_model.chk")
    print "Training complete."
    p_loss, reg_loss, onoff_delta = get_losses(decoder, nTest, "test")
    print "test - prediction loss:", p_loss, "regularization loss:", reg_loss, "on-off delta:", onoff_delta
    saver = tf.train.Saver()
    saver.save(sess, "NDM_final.chk")
