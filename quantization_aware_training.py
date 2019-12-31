#coding=utf-8
import re
import time
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.contrib.slim import get_variables_to_restore
from tensorflow.python.framework import graph_util
import tensorflow.examples.tutorials.mnist.input_data as input_data

# 参数设置
KEEP_PROB = 0.5
LEARNING_RATE = 1e-5
BATCH_SIZE = 30
PARAMETER_FILE = "./checkpoint/variable.ckpt-100000"
MAX_ITER = 100000

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
        # 获取当前的计算图，用于后续的量化
        self.g = tf.get_default_graph()

        if self.is_train:
            # 在损失函数之后，优化器定义之前，在这里会自动选择计算图中的一些operation和activation做伪量化
            tf.contrib.quantize.create_training_graph(self.g, 80000)
            self.lr = LEARNING_RATE
            self.train_op = tf.train.AdamOptimizer(self.lr).minimize(self.loss)
        else:
            # 用于预测时，将之前训练时构造的伪量化的operation和activation实际量化，用于后续的推断
            tf.contrib.quantize.create_eval_graph(self.g)
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

# 将保存了训练时伪量化信息的checkpoint文件转换成freeze pb文件
def freeze():
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        le_net = Lenet(False)
        saver = tf.train.Saver()
        # 导出当前计算图的GraphDef部分
        saver.restore(sess, "checkpoint/variable.ckpt-100000")
        # 保存指定的节点，并将节点值保存为常数
        frozen_graph_def = graph_util.convert_variables_to_constants(sess, sess.graph_def, ['predictions'])
        # 将计算图写入到模型文件中
        tf.io.write_graph(frozen_graph_def, "pb_model", "freeze_eval_graph.pb", as_text=False)

# 将存储了伪量化信息的freeze pb文件转换成完全量化的tflite文件，可以看见量化完之后文件内存基本减小到1/4
def convert_to_tflite():
    converter = tf.lite.TFLiteConverter.from_frozen_graph("pb_model/freeze_eval_graph.pb", ["inputs"], ["predictions"])
    converter.inference_type = tf.lite.constants.QUANTIZED_UINT8
    converter.quantized_input_stats = {"inputs":(0., 1.)} # mean, std_dev，需要自己从训练集（增强后，输入网络之前的）统计出来
    converter.allow_custom_ops = True
    converter.default_ranges_stats = (0, 255)
    converter.post_training_quantize = True
    tflite_model = converter.convert()
    open("tflite_model/eval_graph.tflite", "wb").write(tflite_model)

# 使用原始的checkpoint进行预测
def origin_predict():
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
    sess = tf.Session()
    saver = tf.train.import_meta_graph("./checkpoint/variable.ckpt-100000.meta")
    saver.restore(sess, "./checkpoint/variable.ckpt-100000")

    input_node = sess.graph.get_tensor_by_name('inputs:0')
    pred = sess.graph.get_tensor_by_name('predictions:0')
    labels = [label.index(1) for label in mnist.test.labels.tolist()]
    predictions = []
    start_time = time.time()
    for i in range(10):
        for image in mnist.test.images:
            prediction = sess.run(pred, feed_dict={input_node: [image]}).tolist()[0]
            predictions.append(prediction)
    end_time = time.time()
    correct = 0
    for prediction, label in zip(predictions, labels):
        if prediction == label:
            correct += 1
    print(correct / len(labels))
    print((end_time - start_time))
    sess.close()

# 使用freeze pb文件进行预测
def freeze_pb_predict():
    mnist = input_data.read_data_sets('MNIST_data/', one_hot=True)
    with tf.Session() as sess:
        with tf.gfile.FastGFile("pb_model/freeze_eval_graph.pb", 'rb') as f:
            # 使用tf.GraphDef()定义一个空Graph
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            # 返回一个上下文管理器,使得这个Graph对象成为当前默认的graph.当你想在一个进程里面创建
            # 多个图的时候,就应该使用这个函数.为了方便起见,一个全局的图对象被默认提供,要是你没有
            # 显式创建一个新的图的话,所有的操作(ops)都会被添加到这个默认的图里面来.
            #sess.graph.as_default()
            # 导入Graph
            tf.import_graph_def(graph_def, name='')
            # tf.global_variables_initializer()添加节点用于初始化所有的变量(GraphKeys.VARIABLES)。
            # 返回一个初始化所有全局变量的操作（Op）。在你构建完整个模型并在会话中加载模型后，运行这个节点。
            sess.run(tf.global_variables_initializer())
            # 获取输出Tensor和输出Tensor
            input_node = sess.graph.get_tensor_by_name('inputs:0')
            pred = sess.graph.get_tensor_by_name('predictions:0')

            labels = [label.index(1) for label in mnist.test.labels.tolist()]
            predictions = []
            start_time = time.time()
            for i in range(10):
                for image in mnist.test.images:
                    prediction = sess.run(pred, feed_dict={input_node: [image]}).tolist()[0]
                    predictions.append(prediction)
            end_time = time.time()
            correct = 0
            for prediction, label in zip(predictions, labels):
                if prediction == label:
                    correct += 1
            print(correct / len(labels))
            print((end_time - start_time))

# 使用tflite进行预测
def tflite_predict():
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
    labels = [label.index(1) for label in mnist.test.labels.tolist()]
    images = mnist.test.images

    means = np.mean(images, axis=1).reshape([10000, 1])
    std = np.std(images, axis=1, ddof=1).reshape([10000, 1])
    images = (images - means) / std

    images = np.array(images, dtype="uint8")
    # 根据tflite文件生成解析器
    interpreter = tf.contrib.lite.Interpreter(model_path="./tflite_model/eval_graph.tflite")
    # 用allocate_tensors()分配内存
    interpreter.allocate_tensors()
    # 获取输入输出tensor
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    predictions = []
    start_time = time.time()
    for i in range(10):
        for image in images:
            # 填充输入tensor
            interpreter.set_tensor(input_details[0]['index'], [image])
            # 前向推理
            interpreter.invoke()
            # 获取输出tensor
            score = interpreter.get_tensor(output_details[0]['index'])[0][0]
            # # 结果去掉无用的维度
            # result = np.squeeze(score)
            # #print('result:{}'.format(result))
            # # 输出结果是长度为10（对应0-9）的一维数据，最大值的下标就是预测的数字
            predictions.append(score)
    end_time = time.time()
    correct = 0
    for prediction, label in zip(predictions, labels):
        if prediction == label:
            correct += 1
    print((end_time - start_time))
    print(correct / len(labels))


def train():
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

if __name__ == '__main__':
    #train()
    #freeze()
    convert_to_tflite()
    #origin_predict()
    #freeze_pb_predict()
    tflite_predict()

