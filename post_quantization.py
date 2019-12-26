#coding=utf-8
import re
import tensorflow as tf
import  tensorflow.contrib.slim as slim
import tensorflow.examples.tutorials.mnist.input_data as input_data

# 参数设置
KEEP_PROB = 0.5
LEARNING_RATE = 1e-5
BATCH_SIZE = 20
PARAMETER_FILE = "checkpoint/variable.ckpt-20000"
MAX_ITER = 20000

# Build LeNet

class Lenet:
    def __init__(self, is_train=True):
        self.raw_input_image = tf.placeholder(tf.float32, [None, 784], "inputs")
        self.input_images = tf.reshape(self.raw_input_image, [-1, 28, 28, 1])
        self.raw_input_label = tf.placeholder("float", [None, 10], "labels")
        self.input_labels = tf.cast(self.raw_input_label, tf.int32)
        self.dropout = KEEP_PROB
        self.is_train = is_train

        with tf.variable_scope("Lenet") as scope:
            self.train_digits = self.build(True)
            scope.reuse_variables()
            self.pred_digits = self.build(False)

        self.loss = slim.losses.softmax_cross_entropy(self.train_digits, self.input_labels)
        self.lr = LEARNING_RATE
        self.train_op = tf.train.AdamOptimizer(self.lr).minimize(self.loss)

        self.predictions = tf.arg_max(self.pred_digits, 1, name="predictions")
        self.correct_prediction = tf.equal(tf.argmax(self.pred_digits, 1), tf.argmax(self.input_labels, 1))
        self.train_accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, "float"))

    def build(self, is_trained=True):
        with slim.arg_scope([slim.conv2d], padding='VALID',
                            weights_initializer=tf.truncated_normal_initializer(stddev=0.01),
                            weights_regularizer=slim.l2_regularizer(0.0005)):
            net = slim.conv2d(self.input_images, 6, [5, 5], 1, padding='SAME', scope='conv1')
            net = slim.max_pool2d(net, [2, 2], scope='pool2')
            net = slim.conv2d(net, 16, [5, 5], 1, scope='conv3')
            net = slim.max_pool2d(net, [2, 2], scope='pool4')
            net = slim.conv2d(net, 120, [5, 5], 1, scope='conv5')
            net = slim.flatten(net, scope='flat6')
            net = slim.fully_connected(net, 84, scope='fc7')
            net = slim.dropout(net, self.dropout, is_training=is_trained, scope='dropout8')
            digits = slim.fully_connected(net, 10, scope='fc9')
        return digits


def main():
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
    test_images = mnist.test.images
    test_labels = mnist.test.labels
    sess = tf.Session()
    batch_size = BATCH_SIZE
    paramter_path = PARAMETER_FILE
    max_iter = MAX_ITER

    lenet = Lenet()
    variables = get_variables_to_restore()
    save_vars = [variable for variable in variables if not re.search("Adam", variable.name)]

    saver = tf.train.Saver(save_vars)
    sess.run(tf.initialize_all_variables())
    # 用来显示标量信息
    tf.summary.scalar("loss", lenet.loss)
    # merge_all 可以将所有summary全部保存到磁盘，以便tensorboard显示。如果没有特殊要求，
    # 一般用这一句就可一显示训练时的各种信息了。
    summary_op = tf.summary.merge_all()
    # 指定一个文件用来保存图
    train_summary_writer = tf.summary.FileWriter("logs", sess.graph)

    for i in range(max_iter):
        batch = mnist.train.next_batch(batch_size)
        if i % 100 == 0:
            train_accuracy, summary = sess.run([lenet.train_accuracy, summary_op], feed_dict={
                lenet.raw_input_image: batch[0],
                lenet.raw_input_label: batch[1]
            })
            train_summary_writer.add_summary(summary)
            print("step %d, training accuracy %g" % (i, train_accuracy))

        if i % 500 == 0:
            test_accuracy = sess.run(lenet.train_accuracy, feed_dict={lenet.raw_input_image: test_images,
                                                                      lenet.raw_input_label: test_labels})
            print("\n")
            print("step %d, test accuracy %g" % (i, test_accuracy))
            print("\n")
        sess.run(lenet.train_op, feed_dict={lenet.raw_input_image: batch[0],
                                            lenet.raw_input_label: batch[1]})
    saver.save(sess, paramter_path)
    print("saved model")

    # 保存为saved_model
    builder = tf.saved_model.builder.SavedModelBuilder("pb_model")
    inputs = {"inputs": tf.saved_model.utils.build_tensor_info(lenet.raw_input_image)}
    outputs = {"predictions": tf.saved_model.utils.build_tensor_info(lenet.predictions)}
    prediction_signature = tf.saved_model.signature_def_utils.build_signature_def(inputs=inputs, outputs=outputs,
                                                method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME)

    legacy_init_op = tf.group(tf.tables_initializer(), name="legacy_init_op")
    builder.add_meta_graph_and_variables(sess, [tf.saved_model.tag_constants.SERVING],
                                         signature_def_map={"serving_default": prediction_signature},
                                         legacy_init_op=legacy_init_op,
                                         saver=saver)
if __name__ == '__main__':
    main()


