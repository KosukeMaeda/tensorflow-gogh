import tensorflow as tf
import numpy as np
from PIL import Image

from vgg16 import Vgg16
import utils

tf.set_random_seed(20170425)
np.random.seed(20170425)

PATH_CONTENT = './data/img1.jpg'
PATH_STYLE = './data/img2.jpg'
PATH_VGG16_NPY = './data/vgg16.npy'

img_content = utils.load_image(PATH_CONTENT)
img_content = np.reshape(img_content, [1, 224, 224, 3])
img_style = utils.load_image(PATH_STYLE)
img_style = np.reshape(img_style, [1, 224, 224, 3])


def get_gram_matrix_tf(x):
    x = tf.squeeze(x, [0])
    return tf.einsum('xyi,xyj->ij', x, x)


with tf.Session() as sess:
    t_content = tf.placeholder(tf.float32, [1, 224, 224, 3])
    vgg16_content = Vgg16(PATH_VGG16_NPY)
    vgg16_content.build(t_content)

    t_style = tf.placeholder(tf.float32, [1, 224, 224, 3])
    vgg16_style = Vgg16(PATH_VGG16_NPY)
    vgg16_style.build(t_style)

    x = tf.Variable(tf.truncated_normal([1, 224, 224, 3], 0.5, 0.1))
    x = tf.reshape(tf.clip_by_value(tf.squeeze(x, [0]), 0, 1), shape=(1, 224, 224, 3))
    vgg16_out = Vgg16(PATH_VGG16_NPY)
    vgg16_out.build(x)

    f = vgg16_out.conv4_2
    p = vgg16_content.conv4_2
    l_content = tf.reduce_sum(tf.square(f - p)) / 2

    a_conv1 = get_gram_matrix_tf(vgg16_style.conv1_1)
    a_conv2 = get_gram_matrix_tf(vgg16_style.conv2_1)
    a_conv3 = get_gram_matrix_tf(vgg16_style.conv3_1)
    a_conv4 = get_gram_matrix_tf(vgg16_style.conv4_1)
    a_conv5 = get_gram_matrix_tf(vgg16_style.conv5_1)

    g_conv1 = get_gram_matrix_tf(vgg16_out.conv1_1)
    g_conv2 = get_gram_matrix_tf(vgg16_out.conv2_1)
    g_conv3 = get_gram_matrix_tf(vgg16_out.conv3_1)
    g_conv4 = get_gram_matrix_tf(vgg16_out.conv4_1)
    g_conv5 = get_gram_matrix_tf(vgg16_out.conv5_1)

    e_conv1 = tf.reduce_sum(tf.square(g_conv1 - a_conv1)) / (4 * 64 ** 2 * 224 ** 4)
    e_conv2 = tf.reduce_sum(tf.square(g_conv2 - a_conv2)) / (4 * 128 ** 2 * 128 ** 4)
    e_conv3 = tf.reduce_sum(tf.square(g_conv3 - a_conv3)) / (4 * 256 ** 2 * 56 ** 4)
    e_conv4 = tf.reduce_sum(tf.square(g_conv4 - a_conv4)) / (4 * 512 ** 2 * 28 ** 4)
    e_conv5 = tf.reduce_sum(tf.square(g_conv5 - a_conv5)) / (4 * 512 ** 2 * 14 ** 4)

    l_style = 0.2 * (e_conv1 + e_conv2 + e_conv3 + e_conv4 + e_conv5)

    param = 0.001  # param = alpha / beta
    l_total = param * l_content + l_style

    FEED_DICT = {t_content: img_content, t_style: img_style}

    train_step = tf.train.AdamOptimizer(0.01).minimize(l_total)

    saver = tf.train.Saver()
    ckpt = tf.train.get_checkpoint_state('./session/')
    if ckpt:
        last_model = ckpt.model_checkpoint_path
        print 'load ' + last_model
        saver.restore(sess, last_model)
        i = int(last_model.split('-')[-1])
    else:
        sess.run(tf.global_variables_initializer())
        i = 0

    for _ in range(5000):
        i += 1
        sess.run(train_step, feed_dict=FEED_DICT)
        print('Step: %d' % i)

        if i % 50 == 0:
            l, result = sess.run([l_total, x], feed_dict=FEED_DICT)
            print ('Loss: %f' % l)
            result = np.asarray(result[0])
            result = result * 255
            result = np.uint8(result)
            result_img = Image.fromarray(result)
            result_img.save('./out/gogh-{0}.jpg'.format(i))

            if i % 1000 == 0:
                saver.save(sess, './session/gogh.ckpt', global_step=i)
