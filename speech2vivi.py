import os
import time
from glob import glob
from six.moves import xrange

from my_test import imageName2VoiceName, getTrainImgFakeName
from my_test import getSampleImgName
from my_test import sample_voice_process
from my_test import ffmpegGenVideo
from my_test import getSampleImgNameHis
from my_test import train_voice_process
from ops import *

class speech2vivi(object):
    def __init__(self, sess, image_size=112,voice_dimen=13,voice_time=35,
                 batch_size=1, sample_size=1, output_size=112,
                 gf_dim=64, df_dim=64, L1_lambda=10,
                 input_c_dim=3, output_c_dim=3, dataset_name='facades',
                 checkpoint_dir=None, sample_dir=None, mode = "wgan-pn"):
        """

                Args:
                    sess: TensorFlow session
                    batch_size: The size of batch. Should be specified before training.
                    output_size: (optional) The resolution in pixels of the images. [256]
                    gf_dim: (optional) Dimension of gen filters in first conv layer. [64]
                    df_dim: (optional) Dimension of discrim filters in first conv layer. [64]
                    input_c_dim: (optional) Dimension of input image color. For grayscale input, set to 1. [3]
                    output_c_dim: (optional) Dimension of output image color. For grayscale input, set to 1. [3]
                """
        self.sess = sess
        self.is_grayscale = (input_c_dim == 1)
        self.batch_size = batch_size
        self.image_size = image_size
        self.voice_dimen = voice_dimen
        self.voice_time = voice_time
        self.sample_size = sample_size
        self.output_size = output_size

        self.gf_dim = gf_dim
        self.df_dim = df_dim

        self.input_c_dim = input_c_dim
        self.output_c_dim = output_c_dim

        self.L1_lambda = L1_lambda

        # batch normalization : deals with poor initialization helps gradient flow
        self.d_bn1 = batch_norm(name='d_bn1')
        self.d_bn2 = batch_norm(name='d_bn2')
        self.d_bn3 = batch_norm(name='d_bn3')

        self.g_bn_v_e1 = batch_norm(name='g_bn_v_e1')
        self.g_bn_v_e2 = batch_norm(name='g_bn_v_e2')
        self.g_bn_v_e3 = batch_norm(name='g_bn_v_e3')
        self.g_bn_v_e4 = batch_norm(name='g_bn_v_e4')
        self.g_bn_v_e5 = batch_norm(name='g_bn_v_e5')

        self.g_bn_i_e1 = batch_norm(name='g_bn_i_e1')
        self.g_bn_i_e2 = batch_norm(name='g_bn_i_e2')
        self.g_bn_i_e3 = batch_norm(name='g_bn_i_e3')
        self.g_bn_i_e4 = batch_norm(name='g_bn_i_e4')
        self.g_bn_i_e5 = batch_norm(name='g_bn_i_e5')


        self.g_bn_i_v_e1 = batch_norm(name='g_bn_i_v_e1')
        self.g_bn_i_v_e2 = batch_norm(name='g_bn_i_v_e2')
        self.g_bn_i_v_e3 = batch_norm(name='g_bn_i_v_e3')
        self.g_bn_i_v_e4 = batch_norm(name='g_bn_i_v_e4')
        self.g_bn_i_v_e5 = batch_norm(name='g_bn_i_v_e5')
        self.g_bn_i_v_e6 = batch_norm(name='g_bn_i_v_e6')
        self.g_bn_i_v_e7 = batch_norm(name='g_bn_i_v_e7')


        self.dataset_name = dataset_name
        self.checkpoint_dir = checkpoint_dir

        #for wgan mode
        self.mode = mode
        self.LAMBDA = 10

        self.build_model()


    def build_model(self):
        self.real_image = tf.placeholder(tf.float32,
                                         [self.batch_size, self.image_size, self.image_size, self.input_c_dim],
                                         name='mouth_image_real_data')

        self.real_voice = tf.placeholder(tf.float32,
                                         [self.batch_size, self.voice_dimen, self.voice_time, 1],
                                         name='mouth_voice_real_data')

        self.random_image = tf.placeholder(tf.float32,
                                         [self.batch_size, self.image_size, self.image_size, self.input_c_dim],
                                         name='random_image_real_data')

        self.sample_random_image = tf.placeholder(tf.float32,
                                           [1, self.image_size, self.image_size, self.input_c_dim],
                                           name='random_image_real_data')

        self.sample_real_voice = tf.placeholder(tf.float32,
                                         [1, self.voice_dimen, self.voice_time, 1],
                                         name='mouth_voice_real_data')

        print("random shape",self.random_image.shape)

        self.fake_image = self.generator(self.random_image, self.real_voice)

        self.D, self.D_logits = self.discriminator(self.real_image, reuse=False)
        self.D_, self.D_logits_ = self.discriminator(self.fake_image, reuse=True)

        self.fake_image_sample = self.sampler(self.sample_random_image, self.sample_real_voice)

        self.d_sum = tf.summary.histogram("d", self.D)
        self.d__sum = tf.summary.histogram("d_", self.D_)
        self.fake_B_sum = tf.summary.image("fake_image", self.fake_image)

        if self.mode == "wgan-pn":
            print("enter wgan-pn mode")
            self.d_loss_real = tf.reduce_mean(self.D)
            self.d_loss_fake = tf.reduce_mean(self.D_)
            self.g_loss = -tf.reduce_mean(self.D_) + self.L1_lambda * tf.reduce_mean(tf.abs(self.real_image - self.fake_image))

            self.d_loss_real_sum = tf.summary.scalar("d_loss_real", self.d_loss_real)
            self.d_loss_fake_sum = tf.summary.scalar("d_loss_fake", self.d_loss_fake)

            self.d_loss = self.d_loss_fake - self.d_loss_real

            alpha = tf.random_uniform(
                shape=[self.batch_size, 1],
                minval=0.,
                maxval=1.
            )
            temp_fake_image = tf.reshape(self.fake_image,[self.batch_size,-1])
            temp_real_image = tf.reshape(self.real_image,[self.batch_size,-1])

            differences = temp_fake_image - temp_real_image
            print("diff shape:",differences.shape)
            print("alpha shape:",alpha.shape)
            interpolates = temp_real_image + (alpha * differences)
            interpolates = tf.reshape(interpolates,[self.batch_size,112,112,3])
            intD, intD_ = self.discriminator(interpolates,reuse=True)
            gradients = tf.gradients(intD, [interpolates])[0]
            print("gradients:",gradients)
            print("gradients:",gradients.shape)
            # gradients = gradients[0]
            slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1]))
            gradient_penalty = tf.reduce_mean((slopes - 1.) ** 2)
            self.d_loss += self.LAMBDA * gradient_penalty

        else:
            print("enter original mode")
            self.d_loss_real = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_logits, labels=tf.ones_like(self.D)))
                # self.D_logits - tf.ones_like(self.D))

            self.d_loss_fake = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_logits_, labels=tf.zeros_like(self.D_)))
                # self.D_logits_ - tf.zeros(self.D_))

            self.g_loss_one = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_logits_, labels=tf.ones_like(self.D_)))
                # self.D_logits_ - tf.ones_like(self.D_))
            self.g_loss_two =  self.L1_lambda * tf.reduce_mean(tf.abs(self.real_image - self.fake_image))

            self.d_loss_real_sum = tf.summary.scalar("d_loss_real", self.d_loss_real)
            self.d_loss_fake_sum = tf.summary.scalar("d_loss_fake", self.d_loss_fake)

            self.d_loss = self.d_loss_real + self.d_loss_fake
            self.g_loss = self.g_loss_one + self.g_loss_two
            # self.g_loss = self.g_loss_one


        self.g_loss_sum = tf.summary.scalar("g_loss", self.g_loss)
        self.d_loss_sum = tf.summary.scalar("d_loss", self.d_loss)

        t_vars = tf.trainable_variables()

        self.d_vars = [var for var in t_vars if 'd_' in var.name]
        self.g_vars = [var for var in t_vars if 'g_' in var.name]

        self.saver = tf.train.Saver()



    def generator(self, image, voice, y=None):
        with tf.variable_scope("generator") as scope:
            # for audio
            # voice = tf.constant(1.0, shape=[self.batch_size, 13, 35, 1])

            v_conv1 = conv2d(voice, 64, 3, 3, 1, 1, name="g_v_conv1")
            # v_e1 = lrelu(v_conv1)
            v_e1 = v_conv1
            v_conv2 = conv2d(v_e1, 128, 3, 3, 1, 1, name="g_v_conv2")
            # v_e2 = lrelu(self.g_bn_v_e2(v_conv2))
            # v_e2 = lrelu(v_conv2)
            v_e2 = v_conv2
            v_max_pool2 = max_pool_3x3_2t(v_e2)
            v_conv3 = conv2d(v_max_pool2, 256, 3, 3, 1, 1, name="g_v_conv3")
            # v_e3 = lrelu(self.g_bn_v_e3(v_conv3))
            # v_e3 = lrelu(v_conv3)
            v_e3 = v_conv3
            v_conv4 = conv2d(v_e3, 256, 3, 3, 1, 1, name="g_v_conv4")
            # v_e4 = lrelu(self.g_bn_v_e4(v_conv4))
            # v_e4 = lrelu(v_conv4)
            v_e4 = v_conv4
            v_conv5 = conv2d(v_e4, 512, 3, 3, 1, 1, name="g_v_conv5")
            # v_e5 = lrelu(self.g_bn_v_e5(v_conv5))
            # v_e5 = lrelu(v_conv5)
            v_e5 = v_conv5
            v_max_pool2 = max_pool_3x3_2t(v_e5)
            # v_max_pool2 = max_pool_3x3_2t(v_conv5)
            # v_fc6 = lrelu(linear(tf.reshape(v_max_pool2, [self.batch_size, -1]), 512, "g_v_fc6"))
            v_fc6 = linear(tf.reshape(v_max_pool2, [self.batch_size, -1]), 512, "g_v_fc6")
            # v_fc7 = lrelu(linear(tf.reshape(v_fc6, [self.batch_size, -1]), 256, "g_v_fc7"))
            v_fc7 = linear(tf.reshape(v_fc6, [self.batch_size, -1]), 256, "g_v_fc7")

            # for video
            # image = tf.constant(1.0, shape=[self.batch_size, 112, 112, 3])

            i_conv1 = conv2d(image, 96, 7, 7, 2, 2, name="g_i_conv1")
            # i_e1 = self.g_bn_i_e1(i_conv1)
            # i_e1 = lrelu(i_conv1, name="g_i_e1")
            i_e1 = i_conv1
            i_maxPool1 = max_pool_3x3_2(i_e1)

            i_conv2 = conv2d_valid(i_maxPool1, 256, 5, 5, 2, 2, name="g_i_conv2")
            # i_e2 = self.g_bn_i_e2(i_conv2)
            # i_e2 = lrelu(i_conv2)
            i_e2 = i_conv2
            i_maxPool2 = max_pool_3x3_2(i_e2)

            i_conv3 = conv2d(i_maxPool2, 512, 3, 3, 1, 1, name="g_i_conv3")
            # i_e3 = lrelu(self.g_bn_i_e3(i_conv3))
            # i_e3 = lrelu(i_conv3)
            i_e3 = i_conv3
            i_conv4 = conv2d(i_e3, 512, 3, 3, 1, 1, name="g_i_conv4")
            # i_e4 = lrelu(self.g_bn_i_e4(i_conv4))
            # i_e4 = lrelu(i_conv4)
            i_e4 = i_conv4
            i_conv5 = conv2d(i_e4, 512, 3, 3, 1, 1, name="g_i_conv5")
            # i_e5 = lrelu(self.g_bn_i_e5(i_conv5))
            # i_e5 = lrelu(i_conv5)
            i_e5 = i_conv5

            i_reshape1 = tf.reshape(i_e5, [self.batch_size, -1])
            # i_fc6 = lrelu(linear(i_reshape1, 512, "g_i_fc6"))
            i_fc6 = linear(i_reshape1, 512, "g_i_fc6")
            # i_fc7 = lrelu(linear(i_fc6, 256, "g_i_fc7"))
            i_fc7 = linear(i_fc6, 256, "g_i_fc7")

            # concat

            print("v_fc7_shape",v_fc7.shape)
            print("i_fc7_shape",i_fc7.shape)
            i_v_concat = tf.concat([v_fc7, i_fc7], 1)
            print("i_v_concat_shape",i_v_concat.shape)


            # generate new picture
            i_v_fc1 = linear(i_v_concat, 128, "g_i_v_fc1")
            # i_v_convT2 = deconv2d(tf.reshape(self.g_bn_i_v_e1(lrelu(i_v_fc1)), [-1, 2, 2, 32]), [self.batch_size, 4, 4, 512], 6, 6, 2, 2, name="g_i_v_convT2")
            i_v_convT2 = deconv2d(tf.reshape(i_v_fc1, [-1, 2, 2, 32]), [self.batch_size, 4, 4, 512], 6, 6, 2, 2, name="g_i_v_convT2")
            # i_v_e2 = tf.nn.dropout(self.g_bn_i_v_e2(lrelu(i_v_convT2)),0.5)
            # i_v_e2 = tf.nn.dropout(lrelu(i_v_convT2),0.5)
            # i_v_e2 = lrelu(i_v_convT2)
            i_v_e2 = i_v_convT2
            i_v_convT3 = deconv2d_valid(i_v_e2, [self.batch_size, 12, 12, 256], 5, 5, 2, 2, name="g_i_v_convT3")
            # i_v_e3 = tf.nn.dropout(self.g_bn_i_v_e3(lrelu(i_v_convT3)),0.5)
            # i_v_e3 = tf.nn.dropout(lrelu(i_v_convT3),0.5)
            # i_v_e3 = lrelu(i_v_convT3)
            i_v_e3 = i_v_convT3
            i_v_concat1 = tf.concat([i_conv2, i_v_e3], 3, name="g_i_v_concat1")
            i_v_convT4 = deconv2d_valid(i_v_concat1, [self.batch_size, 28, 28, 96], 5, 5, 2, 2, name="g_i_v_convT4")
            # i_v_e4 = tf.nn.dropout(self.g_bn_i_v_e4(lrelu(i_v_convT4)),0.5)
            # i_v_e4 = tf.nn.dropout(lrelu(i_v_convT4),0.5)
            # i_v_e4 = lrelu(i_v_convT4)
            i_v_e4 = i_v_convT4
            i_v_concat2 = tf.concat([i_maxPool1, i_v_e4], 3, name="g_i_v_concat2")
            # for convinient to compare with original size, change to 112*112
            i_v_convT5 = deconv2d(i_v_concat2, [self.batch_size, 56, 56, 96], 5, 5, 2, 2, name="g_i_v_convT5")
            # i_v_e5 = tf.nn.dropout(self.g_bn_i_v_e5(lrelu(i_v_convT5)),0.5)
            # i_v_e5 = tf.nn.dropout(lrelu(i_v_convT5),0.5)
            # i_v_e5 = lrelu(i_v_convT5)
            i_v_e5 = i_v_convT5
            i_v_convT6 = deconv2d(i_v_e5, [self.batch_size, 112, 112, 64], 5, 5, 2, 2, name="g_i_v_convT6")
            # i_v_e6 = tf.nn.dropout(self.g_bn_i_v_e6(lrelu(i_v_convT6)),0.5)
            # i_v_e6 = tf.nn.dropout(lrelu(i_v_convT6),0.5)
            # i_v_e6 = lrelu(i_v_convT6)
            i_v_e6 = i_v_convT6
            i_v_convT7 = deconv2d(i_v_e6, [self.batch_size, 112, 112, 3], 5, 5, 1, 1, name="g_i_v_convT7")
            # i_v_e7 = tf.nn.dropout(self.g_bn_i_v_e7(lrelu(i_v_convT7)),0.5)
            # i_v_e7 = tf.nn.dropout(lrelu(i_v_convT7),0.5)
            # i_v_e7 = lrelu(i_v_convT7)
            i_v_e7 = i_v_convT7

            # print("generator's shape:",i_v_e7.shape)
            return i_v_e7
            # return i_v_convT7



    def discriminator(self, image, y=None, reuse=False):

        with tf.variable_scope("discriminator") as scope:

            # image = tf.constant(1.0, shape=[self.batch_size, 112, 112, 3])

            if reuse:
                tf.get_variable_scope().reuse_variables()
            else:
                assert tf.get_variable_scope().reuse == False

            if self.mode == "wgan-pn":
                h0 = lrelu(conv2d(image, self.df_dim, name='d_h0_conv'))
                # h0 is (128 x 128 x self.df_dim)
                h1 = lrelu(conv2d(h0, self.df_dim*2, name='d_h1_conv'))
                # h1 is (64 x 64 x self.df_dim*2)
                h2 = lrelu(conv2d(h1, self.df_dim*4, name='d_h2_conv'))
                # h2 is (32x 32 x self.df_dim*4)
                h3 = lrelu(conv2d(h2, self.df_dim*8, d_h=1, d_w=1, name='d_h3_conv'))
                # h3 is (16 x 16 x self.df_dim*8)
                h4 = linear(tf.reshape(h3, [self.batch_size, -1]), 1, 'd_h3_lin')

                # h0 = conv2d(image, self.df_dim, name='d_h0_conv')
                # h0 is (128 x 128 x self.df_dim)
                # h1 = conv2d(h0, self.df_dim * 2, name='d_h1_conv')
                # h1 is (64 x 64 x self.df_dim*2)
                # h2 = conv2d(h1, self.df_dim * 4, name='d_h2_conv')
                # h2 is (32x 32 x self.df_dim*4)
                # h3 = conv2d(h2, self.df_dim * 8, d_h=1, d_w=1, name='d_h3_conv')
                # h3 is (16 x 16 x self.df_dim*8)
                # h4 = linear(tf.reshape(h3, [self.batch_size, -1]), 1, 'd_h3_lin')

            else:
                # h0 = lrelu(conv2d(image, self.df_dim, name='d_h0_conv'))
                # h0 is (128 x 128 x self.df_dim)
                # h1 = lrelu(self.d_bn1(conv2d(h0, self.df_dim*2, name='d_h1_conv')))
                # h1 is (64 x 64 x self.df_dim*2)
                # h2 = lrelu(self.d_bn2(conv2d(h1, self.df_dim*4, name='d_h2_conv')))
                # h2 is (32x 32 x self.df_dim*4)
                # h3 = lrelu(self.d_bn3(conv2d(h2, self.df_dim*8, d_h=1, d_w=1, name='d_h3_conv')))
                # h3 is (16 x 16 x self.df_dim*8)
                # h4 = linear(tf.reshape(h3, [self.batch_size, -1]), 1, 'd_h3_lin')
                h0 = conv2d(image, self.df_dim, name='d_h0_conv')
                # h0 is (128 x 128 x self.df_dim)
                h1 = self.d_bn1(conv2d(h0, self.df_dim * 2, name='d_h1_conv'))
                # h1 is (64 x 64 x self.df_dim*2)
                h2 = self.d_bn2(conv2d(h1, self.df_dim * 4, name='d_h2_conv'))
                # h2 is (32x 32 x self.df_dim*4)
                h3 = self.d_bn3(conv2d(h2, self.df_dim * 8, d_h=1, d_w=1, name='d_h3_conv'))
                # h3 is (16 x 16 x self.df_dim*8)
                h4 = linear(tf.reshape(h3, [self.batch_size, -1]), 1, 'd_h3_lin')

            if self.mode == "wgan-pn":
                return h4,h4
            else:
                return h4,h4
                # return tf.nn.sigmoid(h4), h4


    def train(self, args):
        """Train pix2pix"""
        if self.mode == "wgan-pn":
            d_optim = tf.train.AdamOptimizer(args.lr_d, beta1=0.5, beta2=0.9) \
                              .minimize(self.d_loss, var_list=self.d_vars)
            g_optim = tf.train.AdamOptimizer(args.lr_g, beta1=0.5, beta2=0.9) \
                .minimize(self.g_loss, var_list=self.g_vars)
        else:
            d_optim = tf.train.AdamOptimizer(args.lr_d, beta1=args.beta1) \
                              .minimize(self.d_loss, var_list=self.d_vars)
            g_optim = tf.train.AdamOptimizer(args.lr_g, beta1=args.beta1) \
                              .minimize(self.g_loss, var_list=self.g_vars)

        init_op = tf.global_variables_initializer()
        self.sess.run(init_op)

        self.g_sum = tf.summary.merge([self.d__sum,
            self.fake_B_sum, self.d_loss_fake_sum, self.g_loss_sum])
        self.d_sum = tf.summary.merge([self.d_sum, self.d_loss_real_sum, self.d_loss_sum])
        self.merged = tf.summary.merge_all()
        self.writer = tf.summary.FileWriter("./logs", self.sess.graph)


        counter = 0
        start_time = time.time()

        if self.load(self.checkpoint_dir):
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")

        # train_voice_process("./datasets/first_run/real_voice/","./datasets/first_run/sample/specified01.mp3");
        for epoch in xrange(args.epoch):
            # data = glob('./datasets/{}/train/*.jpg'.format(self.dataset_name))
            #np.random.shuffle(data)
            # specifiedImage = [imread("./datasets/first_run/random.jpg") for i in xrange(self.batch_size)]
            # specifiedImage = imread("./datasets/first_run/random.jpg")
            # specifiedImage = specifiedImage.reshape(1,112,112,3)
            # specifiedImage = np.array(specifiedImage).astype(np.float32)
            # specifiedImage = tf.reshape(specifiedImage,[1,112,112,3])
            # print("specifiedImage shape==>",specifiedImage.shape)
            data = glob('./datasets/first_run/real_image/*.jpg')
            batch_idxs = min(len(data), args.train_size) // self.batch_size
            print("batch_idxs:",batch_idxs)

            for idx in xrange(0, batch_idxs):
                batch_files = data[idx*self.batch_size:(idx+1)*self.batch_size]
                # batch = [load_data(batch_file) for batch_file in batch_files]
                # if (self.is_grayscale):
                #     batch_images = np.array(batch).astype(np.float32)[:, :, :, None]
                # else:"
                #     batch_images = np.array(batch).astype(np.float32)
                batch_images = [imread(batch_file) for batch_file in batch_files]
                voiceData = [np.loadtxt(imageName2VoiceName(batch_file)).reshape(13,35,1) for batch_file in batch_files]
                batch_specific_images = [imread(self.getSpecificImage(batch_file)) for batch_file in batch_files]
                # for batch_file in batch_files:
                #     batch_images = batch_images.reshape(1,112,112,3)
                #     batch_images = np.array(batch_images).astype(np.float32)
                    # batch_images = tf.reshape(batch_images,[1,112,112,3])
                batch_images = np.array(batch_images).astype(np.float32)
                batch_specific_images = np.array(batch_specific_images).astype(np.float32)
                print(batch_images.shape)
                print("real_image shape==>",batch_images.shape)
                # voiceName = imageName2VoiceName(batch_file)
                # print("voiceName ==>",voiceName)
                #     voiceData = np.loadtxt(voiceName)
                # voiceData = voiceData.reshape(self.batch_size,13,35,1)
                voiceData = np.array(voiceData).astype(np.float32)
                print("voiceData shape==>",voiceData.shape)
                    # print(voiceData)
                    # voiceData = tf.reshape(voiceData,[1,13,35])
                # Update D network
                _, summary_str = self.sess.run([d_optim, self.d_sum],
                                               feed_dict={ self.real_image: batch_images, self.real_voice:voiceData, self.random_image: batch_specific_images })
                self.writer.add_summary(summary_str, counter)

                # Update G network
                _, summary_str = self.sess.run([g_optim, self.merged],
                                               feed_dict={ self.real_image: batch_images, self.real_voice:voiceData, self.random_image: batch_specific_images })
                self.writer.add_summary(summary_str, counter)

                # Run g_optim twice to make sure that d_loss does not go to zero (different from paper)
                _, summary_str = self.sess.run([g_optim, self.merged],
                                               feed_dict={ self.real_image: batch_images, self.real_voice:voiceData, self.random_image: batch_specific_images })
                self.writer.add_summary(summary_str, counter)

                errD_fake = self.d_loss_fake.eval({self.real_image: batch_images, self.real_voice:voiceData, self.random_image: batch_specific_images})
                errD_real = self.d_loss_real.eval({self.real_image: batch_images, self.real_voice:voiceData, self.random_image: batch_specific_images})
                errG_1 = self.g_loss_one.eval({self.real_image: batch_images, self.real_voice:voiceData, self.random_image: batch_specific_images})
                errG_2 = self.g_loss_two.eval({self.real_image: batch_images, self.real_voice:voiceData, self.random_image: batch_specific_images})

                counter += 1
                print("Epoch: [%2d] [%4d/%4d] time: %4.4f, d_loss: %.8f, g_loss: %.8f g_loss: %.8f" \
                    % (epoch, idx, batch_idxs,
                        time.time() - start_time, errD_fake+errD_real, errG_1, errG_2))
                train_fake_dir = "/home/train_fake/";
                # so_save_image(self.fake_image, getTrainImgFakeName(train_fake_dir, counter))
                temp_fake_image = self.sess.run([self.fake_image],feed_dict={self.real_image: batch_images, self.real_voice:voiceData, self.random_image: batch_specific_images})#TODO
                # {self.sample_random_image: specifiedSmapleImage, self.sample_real_voice: voiceData})
                # temp_fake_image = np.reshape(temp_fake_image,[112,112,3])
                # so_save_image(temp_fake_image,"/home/jackform/workspace/meiliyuyin/fake.jpg")
                # temp_real_image = np.reshape(batch_images,[112,112,3])
                # so_save_image(temp_real_image,"/home/jackform/workspace/meiliyuyin/real.jpg")


                if np.mod(counter, 50) == 0:
                    print("start to sample")
                    self.sample_model2(args.sample_dir, epoch, idx)

                if np.mod(counter, 50) == 0:
                    self.save(args.checkpoint_dir, counter)



    # 根据当前image名字00xx00xx，获取第一帧00xx0001,第一帧没有，按顺序获取，不包含自己，没有则返会-1
    def getSpecificImage(self,trainImage):
        targetName = trainImage #FIXME
        print(" trainImage: " + trainImage)
        baseName = os.path.basename(trainImage)
        # dirName = os.path.dirname(trainImage)
        temp = baseName.split('.')
        high4 = temp[0][0:4]
        realImageDir = "./datasets/first_run/real_image"
        XXXXimages = glob(realImageDir + "/" + high4 + "*.jpg")
        # print(" cout: *.jpg" +len(XXXXimages).__str__())
        for tempImg in XXXXimages:
            if tempImg == baseName:
                continue
            else:
                targetName = tempImg
                break
        print(" targetName: " + targetName)
        return targetName

    def load_random_samples(self):
        data = np.random.choice(glob('./datasets/{}/val/*.jpg'.format(self.dataset_name)), self.batch_size)
        sample = [load_data(sample_file) for sample_file in data]

        if (self.is_grayscale):
            sample_images = np.array(sample).astype(np.float32)[:, :, :, None]
        else:
            sample_images = np.array(sample).astype(np.float32)
        return sample_images

    def load_sample_voices(self):
        data = glob('./datasets/first_run/sample_voice/*.txt')
        sample_voices = [load_void_data(sample_file) for sample_file in data]
        return sample_voices

    def sample_model(self, sample_dir, epoch, idx):
        sample_images = self.load_random_samples()
        samples, d_loss, g_loss = self.sess.run(
            [self.fake_image_sample, self.d_loss, self.g_loss],
            feed_dict={self.random_image: sample_images}
        )
        save_images(samples, [self.batch_size, 1],
                    './{}/train_{:02d}_{:04d}.png'.format(sample_dir, epoch, idx))
        print("[Sample] d_loss: {:.8f}, g_loss: {:.8f}".format(d_loss, g_loss))

    def sample_model2(self, sample_dir, epoch, idx):

        specifiedSmapleImagePath = "./datasets/first_run/sample/K.jpg"
        sample_mp3               = "./datasets/first_run/sample/K.mp3"
        out_mp4="./datasets/first_run/sample/K.mp4"

        gen_sample_voices = "./datasets/first_run/sample/gen_sample_voices/"#*.txt
        gen_sample_images = "./datasets/first_run/sample/gen_sample_images/"#*.jpg
       #gen_sample_images_his = "./datasets/first_run/sample/gen_sample_images_his/"  # *.jpg 历史数据
        #pic
        specifiedSmapleImage = imread(specifiedSmapleImagePath)
        specifiedSmapleImage = specifiedSmapleImage.reshape(1, 112, 112, 3)
        specifiedSmapleImage = np.array(specifiedSmapleImage).astype(np.float32)
        #voice
        # sample_voice_process(genSampleVoice=gen_sample_voices,SampleVoice=sample_mp3)
        data = glob(gen_sample_voices+'*.txt')
        for voicepath in data:
            print("deal with voice",voicepath)
            voiceData = np.loadtxt(voicepath)
            print("voice shape",voiceData.shape)
            voiceData = voiceData.reshape(1, 13, 35, 1)
            voiceData = np.array(voiceData).astype(np.float32)
            samples = self.sess.run([self.fake_image_sample],feed_dict={self.sample_random_image: specifiedSmapleImage, self.sample_real_voice: voiceData})
            # print(tf.shape(samples))
            samples = np.reshape(samples,[112,112,3])
            so_save_image(samples, getSampleImgName(voicepath,gen_sample_images))
            # so_save_image(samples, getSampleImgNameHis(voicepath, gen_sample_images_his, epoch, idx, ))#历史记录
            # print("[Sample] d_loss: {:.8f}, g_loss: {:.8f}".format(d_loss, g_loss))
        ffmpegGenVideo(imageSlicesDir=gen_sample_images,mp3SampleFile=sample_mp3,outfile=out_mp4)

    def save(self, checkpoint_dir, step):
        model_name = "pix2pix.model"
        model_dir = "%s_%s_%s" % (self.dataset_name, self.batch_size, self.output_size)
        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.saver.save(self.sess,
                        os.path.join(checkpoint_dir, model_name),
                        global_step=step)



    def sampler(self, image, voice, batch_size = 1):
        with tf.variable_scope("generator") as scope:
            scope.reuse_variables()
            # for audio
            # voice = tf.constant(1.0, shape=[self.batch_size, 13, 35, 1])

            v_conv1 = conv2d(voice, 64, 3, 3, 1, 1, name="g_v_conv1",no_summery=True)
            # v_e1 = self.g_bn_v_e1(v_conv1)
            # v_conv2 = conv2d(v_e1, 128, 3, 3, 1, 1, name="g_v_conv2")
            v_conv2 = conv2d(v_conv1, 128, 3, 3, 1, 1, name="g_v_conv2",no_summery=True)
            # v_e2 = self.g_bn_v_e2(v_conv2)
            # v_max_pool2 = max_pool_3x3_2t(v_e2)
            v_max_pool2 = max_pool_3x3_2t(v_conv2)
            v_conv3 = conv2d(v_max_pool2, 256, 3, 3, 1, 1, name="g_v_conv3",no_summery=True)
            # v_e3 = self.g_bn_v_e3(v_conv3)
            # v_conv4 = conv2d(v_e3, 256, 3, 3, 1, 1, name="g_v_conv4")
            v_conv4 = conv2d(v_conv3, 256, 3, 3, 1, 1, name="g_v_conv4",no_summery=True)
            # v_e4 = self.g_bn_v_e4(v_conv4)
            # v_conv5 = conv2d(v_e4, 512, 3, 3, 1, 1, name="g_v_conv5")
            v_conv5 = conv2d(v_conv4, 512, 3, 3, 1, 1, name="g_v_conv5",no_summery=True)
            # v_e5 = self.g_bn_v_e5(v_conv5)
            # v_max_pool2 = max_pool_3x3_2t(v_e5)
            v_max_pool2 = max_pool_3x3_2t(v_conv5)
            v_fc6 = linear(tf.reshape(v_max_pool2, [batch_size, -1]), 512, "g_v_fc6",no_summery=True)
            v_fc7 = linear(tf.reshape(v_fc6, [batch_size, -1]), 256, "g_v_fc7",no_summery=True)

            # for video
            # image = tf.constant(1.0, shape=[self.batch_size, 112, 112, 3])

            i_conv1 = conv2d(image, 96, 7, 7, 2, 2, name="g_i_conv1",no_summery=True)
            # i_e1 = self.g_bn_i_e1(i_conv1)
            i_maxPool1 = max_pool_3x3_2(i_conv1)

            i_conv2 = conv2d_valid(i_maxPool1, 256, 5, 5, 2, 2, name="g_i_conv2",no_summery=True)
            # i_e2 = self.g_bn_i_e2(i_conv2)
            i_maxPool2 = max_pool_3x3_2(i_conv2)

            i_conv3 = conv2d(i_maxPool2, 512, 3, 3, 1, 1, name="g_i_conv3",no_summery=True)
            # i_e3 = self.g_bn_i_e3(i_conv3)
            # i_conv4 = conv2d(i_e3, 512, 3, 3, 1, 1, name="g_i_conv4")
            i_conv4 = conv2d(i_conv3, 512, 3, 3, 1, 1, name="g_i_conv4",no_summery=True)
            # i_e4 = self.g_bn_i_e4(i_conv4)
            # i_conv5 = conv2d(i_e4, 512, 3, 3, 1, 1, name="g_i_conv5")
            i_conv5 = conv2d(i_conv4, 512, 3, 3, 1, 1, name="g_i_conv5",no_summery=True)
            # i_e5 = self.g_bn_i_e5(i_conv5)

            # i_reshape1 = tf.reshape(i_e5, [self.batch_size, -1])
            i_reshape1 = tf.reshape(i_conv5, [batch_size, -1])
            i_fc6 = linear(i_reshape1, 512, "g_i_fc6",no_summery=True)
            i_fc7 = linear(i_fc6, 256, "g_i_fc7",no_summery=True)

            # concat

            i_v_concat = tf.concat([v_fc7, i_fc7], 1)

            # generate new picture
            i_v_fc1 = linear(i_v_concat, 128, "g_i_v_fc1",no_summery=True)
            i_v_convT2 = deconv2d(tf.reshape(i_v_fc1, [-1, 2, 2, 32]), [batch_size, 4, 4, 512], 6, 6, 2, 2,
                                  name="g_i_v_convT2",no_summery=True)
            # i_v_e2 = self.g_bn_i_v_e2(i_v_convT2, 0.5)
            # i_v_e2 = tf.nn.dropout(self.g_bn_i_v_e2(i_v_convT2,0.5))
            # i_v_convT3 = deconv2d_valid(i_v_e2, [self.batch_size, 12, 12, 256], 5, 5, 2, 2, name="g_i_v_convT3")
            i_v_convT3 = deconv2d_valid(i_v_convT2, [batch_size, 12, 12, 256], 5, 5, 2, 2, name="g_i_v_convT3",no_summery=True)
            # i_v_e3 = self.g_bn_i_v_e3(i_v_convT3, 0.5)
            # i_v_e3 = tf.nn.dropout(self.g_bn_i_v_e3(i_v_convT3,0.5))
            # i_v_concat1 = tf.concat([i_conv2, i_v_e3], 3, name="g_i_v_concat1")
            # i_v_concat1 = tf.concat([i_conv2, i_v_convT3], 3, name="g_i_v_concat1")
            # i_v_concat1 = tf.concat([i_conv2, i_v_convT3], 3, name="g_i_v_concat1")
            i_v_concat1 = tf.concat([i_v_convT3, i_v_convT3], 3, name="g_i_v_concat1")
            i_v_convT4 = deconv2d_valid(i_v_concat1, [batch_size, 28, 28, 96], 5, 5, 2, 2, name="g_i_v_convT4",no_summery=True)
            # i_v_e4 = self.g_bn_i_v_e4(i_v_convT4)
            # i_v_e4 = tf.nn.dropout(self.g_bn_i_v_e4(i_v_convT4,0.5))
            # i_v_concat2 = tf.concat([i_maxPool1, i_v_e4], 3, name="g_i_v_concat2")
            # i_v_concat2 = tf.concat([i_maxPool1, i_v_convT4], 3, name="g_i_v_concat2")
            i_v_concat2 = tf.concat([i_v_convT4, i_v_convT4], 3, name="g_i_v_concat2")
            # for convinient to compare with original size, change to 112*112
            i_v_convT5 = deconv2d(i_v_concat2, [batch_size, 56, 56, 96], 5, 5, 2, 2, name="g_i_v_convT5",no_summery=True)
            # i_v_e5 = self.g_bn_i_v_e5(i_v_convT5)
            # i_v_e5 = tf.nn.dropout(self.g_bn_i_v_e5(i_v_convT5,0.5))
            # i_v_convT6 = deconv2d(i_v_e5, [self.batch_size, 112, 112, 64], 5, 5, 2, 2, name="g_i_v_convT6")
            i_v_convT6 = deconv2d(i_v_convT5, [batch_size, 112, 112, 64], 5, 5, 2, 2, name="g_i_v_convT6",no_summery=True)
            # i_v_e6 = self.g_bn_i_v_e6(i_v_convT6
            # i_v_e6 = tf.nn.dropout(self.g_bn_i_v_e6(i_v_convT6,0.5))
            # i_v_convT7 = deconv2d(i_v_e6, [self.batch_size, 112, 112, 3], 5, 5, 1, 1, name="g_i_v_convT7")
            i_v_convT7 = deconv2d(i_v_convT6, [batch_size, 112, 112, 3], 5, 5, 1, 1, name="g_i_v_convT7",no_summery=True)
            # i_v_e7 = self.g_bn_i_v_e7(i_v_convT7)
            # i_v_e7 = tf.nn.dropout(self.g_bn_i_v_e7(i_v_convT7,0.5))

            # print("generator's shape:",i_v_e7.shape)
            # return i_v_e7
            return i_v_convT7

    def load(self, checkpoint_dir):
        print(" [*] Reading checkpoint...")

        model_dir = "%s_%s_%s" % (self.dataset_name, self.batch_size, self.output_size)
        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            return True
        else:
            return False


    def test(self, args):
        """Test pix2pix"""
        init_op = tf.global_variables_initializer()
        self.sess.run(init_op)

        sample_files = glob('./datasets/{}/val/*.jpg'.format(self.dataset_name))

        # sort testing input
        n = [int(i) for i in map(lambda x: x.split('/')[-1].split('.jpg')[0], sample_files)]
        sample_files = [x for (y, x) in sorted(zip(n, sample_files))]

        # load testing input
        print("Loading testing images ...")
        sample = [load_data(sample_file, is_test=True) for sample_file in sample_files]

        if (self.is_grayscale):
            sample_images = np.array(sample).astype(np.float32)[:, :, :, None]
        else:
            sample_images = np.array(sample).astype(np.float32)

        sample_images = [sample_images[i:i+self.batch_size]
                         for i in xrange(0, len(sample_images), self.batch_size)]
        sample_images = np.array(sample_images)
        print(sample_images.shape)

        start_time = time.time()
        if self.load(self.checkpoint_dir):
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")

        for i, sample_image in enumerate(sample_images):
            idx = i+1
            print("sampling image ", idx)
            samples = self.sess.run(
                self.fake_image_sample,
                feed_dict={self.real_image: sample_image}
            )
            save_images(samples, [self.batch_size, 1],
                        './{}/test_{:04d}.png'.format(args.test_dir, idx))
