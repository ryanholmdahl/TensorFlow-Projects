import argparse
import matplotlib.pyplot as plt

import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)


class EncoderModel:

    def __init__(self, image_in, style_out):
        self.image_in = image_in
        self.style_out = style_out


class DecoderModel:

    def __init__(self, digit_in, image_out, error, learning_rate, train_step):
        self.digit_in = digit_in
        self.image_out = image_out
        self.error = error
        self.learning_rate = learning_rate
        self.train_step = train_step


class GeneratorModel:

    def __init__(self, style_in, digit_in, image_out):
        self.style_in = style_in
        self.digit_in = digit_in
        self.image_out = image_out


class DiscriminatorModel:

    def __init__(self, style_in, class_out, class_in, cross_entropy, accuracy, learning_rate, train_step):
        self.style_in = style_in
        self.class_out = class_out
        self.class_in = class_in
        self.cross_entropy = cross_entropy
        self.accuracy = accuracy
        self.learning_rate = learning_rate
        self.train_step = train_step


class ConfusionModel:

    def __init__(self, class_out, class_in, cross_entropy, learning_rate, train_step):
        self.class_out = class_out
        self.class_in = class_in
        self.cross_entropy = cross_entropy
        self.learning_rate = learning_rate
        self.train_step = train_step


def weight_bias(shape):
    weight = tf.truncated_normal(shape, stddev=1e-2)
    bias = tf.constant(0.1, shape=[shape[1]])
    return tf.Variable(weight), tf.Variable(bias)


def discriminator_y(count, sign):
    return np.array([[(sign + 1) / 2, (1 - sign) / 2]] * count)


def build_encoder(layer_width, num_layers, data_size, style_size):
    image_in = tf.placeholder(tf.float32, shape=[None, data_size])
    W, b = weight_bias([data_size, layer_width])
    h = tf.nn.relu(tf.matmul(image_in, W) + b)
    encoder_variables.append(W)
    encoder_variables.append(b)
    for i in range(num_layers - 1):
        W, b = weight_bias([layer_width, layer_width])
        h = tf.nn.relu(tf.matmul(h, W) + b)
        encoder_variables.append(W)
        encoder_variables.append(b)
    W, b = weight_bias([layer_width, style_size])
    style_out = tf.matmul(h, W) + b
    encoder_variables.append(W)
    encoder_variables.append(b)
    return EncoderModel(image_in, style_out)


def build_decoder(encoder, layer_width, num_layers, style_size, cat_size, data_size):
    digit_in = tf.placeholder(tf.float32, shape=[None, cat_size])
    W, b = weight_bias([style_size + cat_size, layer_width])
    h = tf.nn.relu(
        tf.matmul(tf.concat(1, [digit_in, encoder.style_out]), W) + b)
    decoder_variables.append(W)
    decoder_variables.append(b)
    for i in range(num_layers - 1):
        W, b = weight_bias([layer_width, layer_width])
        h = tf.nn.relu(tf.matmul(h, W) + b)
        decoder_variables.append(W)
        decoder_variables.append(b)
    W, b = weight_bias([layer_width, data_size])
    image_out = tf.nn.sigmoid(tf.matmul(h, W) + b)
    decoder_variables.append(W)
    decoder_variables.append(b)
    error = tf.reduce_mean(0.5 * tf.square(encoder.image_in - image_out))
    learning_rate = tf.placeholder(tf.float32, shape=[])
    train_step = tf.train.AdamOptimizer(learning_rate).minimize(error)
    return DecoderModel(digit_in, image_out, error, learning_rate, train_step)


def build_generator(style_size, cat_size):
    digit_in = tf.placeholder(tf.float32, shape=[None, cat_size])
    style_in = tf.placeholder(tf.float32, shape=[None, style_size])
    W, b = decoder_variables[0], decoder_variables[1]
    h = tf.nn.relu(tf.matmul(tf.concat(1, [digit_in, style_in]), W) + b)
    for i in range(2, len(decoder_variables) - 2, 2):
        W, b = decoder_variables[i], decoder_variables[i + 1]
        h = tf.nn.relu(tf.matmul(h, W) + b)
    W, b = decoder_variables[
        len(decoder_variables) - 2], decoder_variables[len(decoder_variables) - 1]
    image_out = tf.nn.sigmoid(tf.matmul(h, W) + b)
    return GeneratorModel(style_in, digit_in, image_out)


def build_discriminator(layer_width, num_layers, style_size):
    style_in = tf.placeholder(tf.float32, shape=[None, args.sdim])
    W, b = weight_bias([style_size, layer_width])
    h = tf.nn.relu(tf.matmul(style_in, W) + b)
    discriminator_variables.append(W)
    discriminator_variables.append(b)
    for i in range(num_layers - 1):
        W, b = weight_bias([layer_width, layer_width])
        h = tf.nn.relu(tf.matmul(h, W) + b)
        discriminator_variables.append(W)
        discriminator_variables.append(b)
    W, b = weight_bias([layer_width, 2])
    class_out = tf.nn.log_softmax(tf.matmul(h, W) + b)
    discriminator_variables.append(W)
    discriminator_variables.append(b)

    class_in = tf.placeholder(tf.float32, shape=[None, 2])
    cross_entropy = -tf.reduce_sum(class_in * class_out)
    num_correct = tf.equal(tf.argmax(class_in, 1), tf.argmax(class_out, 1))
    accuracy = tf.reduce_mean(tf.cast(num_correct, tf.float32))
    learning_rate = tf.placeholder(tf.float32, shape=[])
    train_step = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy)
    return DiscriminatorModel(style_in, class_out, class_in, cross_entropy, accuracy, learning_rate, train_step)


def build_confuser(encoder):
    W, b = discriminator_variables[0], discriminator_variables[1]
    h = tf.nn.relu(tf.matmul(encoder.style_out, W) + b)
    for i in range(2, len(discriminator_variables) - 2, 2):
        W, b = discriminator_variables[i], discriminator_variables[i + 1]
        h = tf.nn.relu(tf.matmul(h, W) + b)
    W, b = discriminator_variables[
        len(discriminator_variables) - 2], discriminator_variables[len(discriminator_variables) - 1]
    class_out = tf.nn.log_softmax(tf.matmul(h, W) + b)
    class_in = tf.placeholder(tf.float32, shape=[None, 2])
    cross_entropy = -tf.reduce_sum(class_in * class_out)
    learning_rate = tf.placeholder(tf.float32, shape=[])
    train_step = tf.train.AdamOptimizer(learning_rate).minimize(
        cross_entropy, var_list=encoder_variables)
    return ConfusionModel(class_out, class_in, cross_entropy, learning_rate, train_step)


parser = argparse.ArgumentParser()
parser.add_argument(
    "--epochs", type=int, default=1000, help="number of epochs")
parser.add_argument("--imdim", type=int, default=28, help="dimension of image")
parser.add_argument("--categories", type=int,
                    default=10, help="number of label categories")
parser.add_argument("--lwidth", type=int,
                    default=1024, help="width of network layers")
parser.add_argument("--ldepth", type=int, default=2, help="depth of networks")
parser.add_argument("--sdim", type=int, default=10,
                    help="dimension of style representation")
parser.add_argument("--batch", type=int, default=100, help="size of batches")
parser.add_argument("--generate", action="store_true", help="generate only")
parser.add_argument("--restorefile", help="model file to load from")
parser.add_argument("--learnrate", type=float,
                    default=1e-5, help="learning rate for networks")
parser.add_argument(
    "--saveeach", action="store_true", help="save model at each epoch")
parser.add_argument("--savebest", action="store_true", help="save best model")
parser.add_argument("--showclose", action="store_true",
                    help="show closest training image to generated images")
args = parser.parse_args()

encoder_variables = []
decoder_variables = []
discriminator_variables = []

with tf.Session() as sess:
    print "Initializing networks."
    encoder = build_encoder(
        args.lwidth, args.ldepth, args.imdim * args.imdim, args.sdim)
    decoder = build_decoder(
        encoder, args.lwidth, args.ldepth, args.sdim, args.categories, args.imdim * args.imdim)
    generator = build_generator(args.sdim, args.categories)
    discriminator = build_discriminator(args.lwidth, args.ldepth, args.sdim)
    confuser = build_confuser(encoder)
    print "Networks initialized."

    sess.run(tf.initialize_all_variables())
    if not args.generate:
        min_loss = float('inf')
        if args.restorefile is not None:
            print "Restoring model from checkpoint."
            saver = tf.train.Saver()
            saver.restore(
                sess, args.restorefile)
            print "Model restored."
        for epoch in range(args.epochs):
            for batch_index in range(mnist.train.num_examples / args.batch):
                batch = mnist.train.next_batch(args.batch)
                gaussians = np.random.normal(
                    0, 1, size=[args.batch, args.sdim])
                styles = encoder.style_out.eval(
                    feed_dict={
                        encoder.image_in: batch[0]
                    }
                )
                decoder.train_step.run(
                    feed_dict={
                        encoder.image_in: batch[0] + np.random.normal(0, 0.3, (args.batch, args.imdim * args.imdim)),
                        decoder.digit_in: batch[1],
                        decoder.learning_rate: args.learnrate
                    }
                )
                discriminator.train_step.run(
                    feed_dict={
                        discriminator.style_in: gaussians,
                        discriminator.class_in: discriminator_y(args.batch, 1),
                        discriminator.learning_rate: args.learnrate
                    }
                )
                discriminator.train_step.run(
                    feed_dict={
                        discriminator.style_in: styles,
                        discriminator.class_in: discriminator_y(args.batch, -1),
                        discriminator.learning_rate: args.learnrate
                    }
                )
                confuser.train_step.run(
                    feed_dict={
                        encoder.image_in: batch[0],
                        confuser.class_in: discriminator_y(args.batch, 1),
                        confuser.learning_rate: args.learnrate
                    }
                )
            test_error = decoder.error.eval(
                feed_dict={
                    encoder.image_in: mnist.test.images,
                    decoder.digit_in: mnist.test.labels
                }
            )
            gaussians = np.random.normal(
                0, 1, size=[mnist.test.num_examples, args.sdim])
            styles = encoder.style_out.eval(
                feed_dict={
                    encoder.image_in: mnist.test.images
                }
            )
            disc_acc = 0.5 * discriminator.accuracy.eval(
                feed_dict={
                    discriminator.style_in: gaussians,
                    discriminator.class_in: discriminator_y(mnist.test.num_examples, 1)
                }
            ) + 0.5 * discriminator.accuracy.eval(
                feed_dict={
                    discriminator.style_in: styles,
                    discriminator.class_in: discriminator_y(mnist.test.num_examples, -1)
                }
            )
            print(styles[0])
            print("epoch %d: reconstruction loss %g, discriminator accuracy %g" %
                  (epoch, test_error, disc_acc))
            if args.saveeach:
                saver = tf.train.Saver()
                saver.save(
                    sess, "aa_e" + str(epoch) + ".chk")
            if min_loss > test_error:
                saver = tf.train.Saver()
                saver.save(sess, "aa_best.chk")
                min_loss = test_error
            style_randomized = np.random.normal(0, 1, size=[1, args.sdim])
            for i in range(10):
                digit_one_hot = np.array(
                    [1 if _ == i else 0 for _ in range(args.categories)])
                digit_one_hot.shape = (1, args.categories)
                generated_image = generator.image_out.eval(
                    feed_dict={
                        generator.style_in: style_randomized,
                        generator.digit_in: digit_one_hot
                    }
                )
                plt.subplot(340 + i)
                plt.title("Digit " + str(i))
                plt.imshow(generated_image.reshape(28, 28), cmap='gray')
            plt.tight_layout()
            plt.draw()
            plt.show(block=False)

        print("Training complete.")
    else:
        print "Restoring model for generator."
        saver = tf.train.Saver()
        saver.restore(
            sess, args.restorefile)
        print "Model restored."
    test_error = decoder.error.eval(
        feed_dict={
            encoder.image_in: mnist.test.images,
            decoder.digit_in: mnist.test.labels
        }
    )
    print("Final test reconstruction loss: %g" % (test_error))
    print("Attempting new digit generation.")
    while True:
        style_randomized = np.random.normal(0, 1, size=[1, args.sdim])
        generated_images = []
        for i in range(args.categories):
            digit_one_hot = np.array(
                [1 if _ == i else 0 for _ in range(args.categories)])
            digit_one_hot.shape = (1, args.categories)
            generated_image = generator.image_out.eval(
                feed_dict={
                    generator.style_in: style_randomized,
                    generator.digit_in: digit_one_hot
                }
            )
            generated_images.append(generated_image[0])
            plt.subplot(5, 4, i * 2 + 1)
            plt.title("Generated " + str(i))
            plt.imshow(generated_image.reshape(
                args.imdim, args.imdim), cmap='gray')
        if args.showclose:
            closest_images = [None for _ in range(args.categories)]
            closest_distances = [float('inf') for _ in range(args.categories)]
            for i in range(mnist.train.num_examples):
                batch = mnist.train.next_batch(1)
                image = batch[0][0]
                for t in range(args.categories):
                    if batch[1][0][t] == 1:
                        label = t
                        break
                diff = np.square(generated_images[label] - image)
                distance = sum(diff) / len(diff)
                if distance < closest_distances[label]:
                    closest_images[label] = image
                    closest_distances[label] = distance

            for i in range(args.categories):
                plt.subplot(5, 4, i * 2 + 2)
                plt.title("Closest " + str(i))
                plt.imshow(closest_images[i].reshape(
                    args.imdim, args.imdim), cmap='gray')
        plt.tight_layout()
        plt.draw()
        plt.show()
