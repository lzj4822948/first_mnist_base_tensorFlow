# -*- coding: utf-8 -*-
import os
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import mnist_inference

BATCH_SIZE = 100
LEARNING_RATE_BASE = 0.8
LEARNING_RATE_DECAY = 0.99
REGULARAZTION_RATE = 0.0001
TRAINING_STEPS = 30000
MOVING_AVERAGE_DECAY = 0.99

MODEL_SAVE_PATH = ""
MODEL_NAME = "mnistModel.ckpt"

def createModelPath():
	MODEL_SAVE_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "model")
	if not os.path.exists(MODEL_SAVE_PATH):
		os.makedirs(MODEL_SAVE_PATH)

def train(mnist):
	x = tf.placeholder(tf.float32, [None, mnist_inference.INPUT_NODE], name = 'x-input')
	y_ = tf.placeholder(tf.float32, [None, mnist_inference.OUTPUT_NODE], name = 'y-input')

	regularizer = tf.contrib.layers.l2_regularizer(REGULARAZTION_RATE)
	y = mnist_inference.inference(x, regularizer)
	global_step = tf.Variable(0, trainable = False)

	variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
	variable_averages_op = variable_averages.apply(tf.trainable_variables())
	cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits = y, labels = tf.argmax(y_, 1))
	cross_entropy_mean = tf.reduce_mean(cross_entropy)
	loss = cross_entropy_mean + tf.add_n(tf.get_collection('losses'))
	learning_rate = tf.train.exponential_decay(LEARNING_RATE_DECAY, global_step, mnist.train.num_examples / BATCH_SIZE, LEARNING_RATE_DECAY)
	train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step = global_step)

	with tf.control_dependencies([train_step, variable_averages_op]):
		train_op = tf.no_op(name = 'train')

	saver = tf.train.Saver()
	with tf.Session() as sess:
		tf.initialize_all_variables().run()

		for i in range(TRAINING_STEPS):
			xs, ys = mnist.train.next_batch(BATCH_SIZE)
			_, loss_value, step = sess.run([train_op, loss, global_step], feed_dict = {x: xs, y_: ys})
			if i % 1000 == 0:
				print("After %d training steps, loss on training batch is %g" % (step, loss_value))
				saver.save(sess, os.path.join(MODEL_SAVE_PATH, MODEL_NAME), global_step = global_step)

def main(argv = None):
	createModelPath()
	mnist = input_data.read_data_sets("/tmp/data", one_hot = True)
	train(mnist)

if __name__ == '__main__':
	tf.app.run()

