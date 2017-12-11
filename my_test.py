import os
import librosa
from glob import glob
import scipy.misc
from pydub import AudioSegment
from six.moves import xrange
from model import pix2pix
from ops import *


def load_data(image_path, flip=True, is_test=False):
    img_A, img_B = load_image(image_path)
    img_A, img_B = preprocess_A_and_B(img_A, img_B, flip=flip, is_test=is_test)

    img_A = img_A/127.5 - 1.
    img_B = img_B/127.5 - 1.

    img_AB = np.concatenate((img_A, img_B), axis=2)
    # img_AB shape: (fine_size, fine_size, input_c_dim + output_c_dim)
    return img_AB
    # return img_A

def imageName2VoiceName(imageName):
    fileName = imageName[imageName.rindex("/")+1:]
    temp = fileName.split('.')
    result = "./datasets/first_run/real_voice/"+temp[0]+".txt"
    return result

def getSampleImgNameHis(voicePath,sample_dir, epoch, idx):
    fileName = voicePath[voicePath.rindex("/")+1:]
    temp = fileName.split('.')
    result = '{}train_{:02d}_{:04d}_txt_{}.jpg'.format(sample_dir,epoch,idx,temp[0])
    return result

def getSampleImgName(voicePath,sample_dir):
    fileName = voicePath[voicePath.rindex("/")+1:]
    temp = fileName.split('.')
    result = '{}{}.jpg'.format(sample_dir,temp[0])
    return result

def getTrainImgFakeName(train_dir, index):
    if not os.path.exists(train_dir):
        os.makedirs(train_dir)
    result = '{}/{}_fake.jpg'.format(train_dir,index)

def load_image(image_path):
    input_img = imread(image_path)
    # print("input_image shape",input_img.shape)
    print("input_image shape",input_img)
    w = int(input_img.shape[1])
    w2 = int(w/2)
    img_A = input_img[:, 0:w2]
    img_B = input_img[:, w2:w]

    return img_A, img_B

def imread(path, is_grayscale = False):
    if (is_grayscale):
        return scipy.misc.imread(path, flatten = True).astype(np.float)
    else:
        return scipy.misc.imread(path).astype(np.float)

def preprocess_A_and_B(img_A, img_B, load_size=286, fine_size=256, flip=True, is_test=False):
    if is_test:
        img_A = scipy.misc.imresize(img_A, [fine_size, fine_size])
        img_B = scipy.misc.imresize(img_B, [fine_size, fine_size])
    else:
        img_A = scipy.misc.imresize(img_A, [load_size, load_size])
        img_B = scipy.misc.imresize(img_B, [load_size, load_size])

        h1 = int(np.ceil(np.random.uniform(1e-2, load_size-fine_size)))
        w1 = int(np.ceil(np.random.uniform(1e-2, load_size-fine_size)))
        img_A = img_A[h1:h1+fine_size, w1:w1+fine_size]
        img_B = img_B[h1:h1+fine_size, w1:w1+fine_size]

        if flip and np.random.random() > 0.5:
            img_A = np.fliplr(img_A)
            img_B = np.fliplr(img_B)

    return img_A, img_B

def max_pool_3x3_2(x):
    """max_pool_3x3 downsamples a feature map by 2X."""
    return tf.nn.max_pool(x, ksize=[1, 3, 3, 1],
                              strides=[1, 2, 2, 1], padding='SAME')

def max_pool_3x3_2t(x):
    """max_pool_3x3 downsamples a feature map by 2X."""
    return tf.nn.max_pool(x, ksize=[1, 3, 3, 1],
                          strides=[1, 2, 1, 1], padding='SAME')

def conv_out_size_same(size, stride):
    return int(math.ceil(float(size) / float(stride)))


def readImage():
    batch_size = 1
    data = glob('./datasets/yousaidthat/*.jpg')
    batch_idxs = len(data)
    print("batch_idxs",batch_idxs)

    for idx in xrange(0, batch_idxs):
        if idx == 0:
            batch_files = data[idx * batch_size:(idx+1)*batch_size]
            batch = [imread(batch_file) for batch_file in batch_files]
            batch_images = np.array(batch).astype(np.float32)
            print(batch_images.shape)

def readVoice():
    batch_size = 1
    data = glob('./datasets/yousaidthat/voice/*.txt')
    batch_idxs = len(data)
    print("batch_idxs",batch_idxs)


    for idx in xrange(0, batch_idxs):
        # if idx == 0:
        batch_files = data[idx * batch_size:(idx+1)*batch_size]
        for batch_file in batch_files:
            # a = np.loadtxt(batch_file)
            # print(a.shape)
            print(batch_file)


def main():
    image = tf.constant(1.0, shape=[1,112,112,3])

    with tf.Session() as sess:
        model = pix2pix(sess, image_size=args.fine_size, batch_size=args.batch_size,
                        output_size=args.fine_size, dataset_name=args.dataset_name,
                        checkpoint_dir=args.checkpoint_dir, sample_dir=args.sample_dir)




def you():

    #for audio
    voice = tf.constant(1.0, shape=[1,13,35,1])

    v_conv1 = conv2d(voice,64,3,3,1,1,name="v_conv1")
    v_conv2 = conv2d(v_conv1,128,3,3,1,1,name="v_conv2")
    v_max_pool2 = max_pool_3x3_2t(v_conv2)
    v_conv3 = conv2d(v_max_pool2,256,3,3,1,1,name="v_conv3")
    v_conv4 = conv2d(v_conv3,256,3,3,1,1,name="v_conv4")
    v_conv5 = conv2d(v_conv4,512,3,3,1,1,name="v_conv5")
    v_max_pool2 = max_pool_3x3_2t(v_conv5)
    v_fc6 = linear(tf.reshape(v_max_pool2,[1,-1]),512,"v_fc6")
    v_fc7 = linear(tf.reshape(v_fc6,[1,-1]),256,"v_fc7")


    #for video
    image = tf.constant(1.0, shape=[1,112,112,3])
    i_conv1 = conv2d(image,96,7,7,2,2,name="i_conv1")
    i_e1 = lrelu(i_conv1,name="i_e1")
    i_maxPool1 = max_pool_3x3_2(i_e1)

    i_conv2 = conv2d_valid(i_maxPool1,256,5,5,2,2,name="i_conv2")
    i_e2 = lrelu(i_conv2)
    i_maxPool2 = max_pool_3x3_2(i_e2)

    i_conv3 = conv2d(i_maxPool2,512,3,3,1,1,name="i_conv3")
    i_conv4 = conv2d(i_conv3,512,3,3,1,1,name="i_conv4")
    i_conv5 = conv2d(i_conv4,512,3,3,1,1,name="i_conv5")

    i_reshape1 = tf.reshape(i_conv5,[1,-1])
    i_fc6 = linear(i_reshape1,512,"fc6")
    i_fc7 = linear(i_fc6,256,"fc7")

    #concat

    i_v_concat = tf.concat([v_fc7,i_fc7],1)

    #generate new picture
    i_v_fc1= linear(i_v_concat,128,"i_v_fc1")
    i_v_convT2 = deconv2d(tf.reshape(i_v_fc1,[-1,2,2,32]),[1,4,4,512],6,6,2,2,name="i_v_convT2")
    i_v_convT3 = deconv2d_valid(i_v_convT2,[1,12,12,256],5,5,2,2,name="i_v_convT3")
    i_v_concat1 = tf.concat([i_conv2,i_v_convT3],3)
    i_v_convT4 = deconv2d_valid(i_v_convT3,[1,28,28,96],5,5,2,2,name="i_v_convT4")
    i_v_concat2 = tf.concat([i_maxPool1,i_v_convT4],3)
    # for convinient to compare with original size, change to 112*112
    i_v_convT5 = deconv2d(i_v_concat2,[1,56,56,96],5,5,2,2,name="i_v_convT5")
    i_v_convT6 = deconv2d(i_v_convT5,[1,112,112,64],5,5,2,2,name="i_v_convT6")
    i_v_convT7 = deconv2d(i_v_convT6,[1,112,112,3],5,5,1,1,name="i_v_convT7")

    # pager version
    # i_v_convT5 = deconv2d(i_v_concat2,[1,55,55,96],5,5,2,2,name="i_v_convT5")
    # i_v_convT6 = deconv2d(i_v_convT5,[1,109,109,64],5,5,2,2,name="i_v_convT6")
    # i_v_convT7 = deconv2d(i_v_convT6,[1,109,109,3],5,5,1,1,name="i_v_convT7")

    # op = i_v_convT5
    op = i_v_convT7



    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)

    sess.run(op)
    print(op.shape)


def conv2d_valid(input_, output_dim,
           k_h=5, k_w=5, d_h=2, d_w=2, stddev=0.02,
           name="conv2d"):
    with tf.variable_scope(name):
        w = tf.get_variable('w', [k_h, k_w, input_.get_shape()[-1], output_dim],
                            initializer=tf.truncated_normal_initializer(stddev=stddev))
        conv = tf.nn.conv2d(input_, w, strides=[1, d_h, d_w, 1], padding='VALID')

        biases = tf.get_variable('biases', [output_dim], initializer=tf.constant_initializer(0.0))
        conv = tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape())

        return conv

def deconv2d_valid(input_, output_shape,
             k_h=5, k_w=5, d_h=2, d_w=2, stddev=0.02,
             name="deconv2d", with_w=False):
    with tf.variable_scope(name):
        # filter : [height, width, output_channels, in_channels]
        w = tf.get_variable('w', [k_h, k_w, output_shape[-1], input_.get_shape()[-1]],
                            initializer=tf.random_normal_initializer(stddev=stddev))

        try:
            deconv = tf.nn.conv2d_transpose(input_, w, output_shape=output_shape,
                                            strides=[1, d_h, d_w, 1],padding="VALID")

        # Support for verisons of TensorFlow before 0.7.0
        except AttributeError:
            deconv = tf.nn.deconv2d(input_, w, output_shape=output_shape,
                                    strides=[1, d_h, d_w, 1])

        biases = tf.get_variable('biases', [output_shape[-1]], initializer=tf.constant_initializer(0.0))
        deconv = tf.reshape(tf.nn.bias_add(deconv, biases), deconv.get_shape())

        if with_w:
            return deconv, w, biases
        else:
            return deconv

#13*35[hop_length ubuntu机器192，widonws:164,mac:183]sr必须为标准值44.1k
def train_voice_process(genSampleVoice,SampleVoice):
    song = AudioSegment.from_mp3(SampleVoice)
    print("sample_voice_process")
    print(song.__len__())
    sum = int(song.__len__()/350)

    for i in range(sum):
        next = (i + 1) * 350
        first_10_seconds = song[i * 350:next]
        index =(i + 1);
        mp3name = genSampleVoice+'image{:04d}.mp3'.format(index)
        first_10_seconds.export( mp3name, format="mp3")
        print(first_10_seconds.__len__())
        y1, sr1 = librosa.load(mp3name, sr=16000)#16000 采样率，
        print("y1 length:"+str(len(y1)))
        mfccs = librosa.feature.mfcc(y=y1, sr=sr1, n_mfcc=13, hop_length=183, n_fft=2048)#13*35
        print("==>",len(mfccs).__str__() + "*" + len(mfccs[0]).__str__())
        np.savetxt(genSampleVoice+str(i)+'.txt',mfccs)

#13*35[hop_length ubuntu机器192，widonws:164,mac:183] sr必须为标准值44.1k
def sample_voice_process(genSampleVoice,SampleVoice):
    song = AudioSegment.from_mp3(SampleVoice)
    print("sample_voice_process")
    print(song.__len__())
    sum = int(song.__len__()/350)

    for i in range(sum):
        next = (i + 1) * 350
        first_10_seconds = song[i * 350:next]
        index =(i + 1);
        mp3name = genSampleVoice+'image{:04d}.mp3'.format(index)
        first_10_seconds.export( mp3name, format="mp3")
        y1, sr1 = librosa.load(mp3name, sr=16000)
        mfccs = librosa.feature.mfcc(y=y1, sr=sr1, n_mfcc=13, hop_length=183, n_fft=2048)
        print("==>"+str(i)+"<==",len(mfccs).__str__() + "*" + len(mfccs[0]).__str__())
        np.savetxt(genSampleVoice+'image{:04d}.txt'.format(index),mfccs)

def ffmpegGenVideo(imageSlicesDir,mp3SampleFile,outfile):
    # os.system("ffmpeg -threads2 -y -r 4 -i "+imageSlicesDir+"image%04d.jpg -i "+mp3SampleFile+" -absf aac_adtstoasc "+outfile)
    # -r 是frame rate
    os.system("ffmpeg -y -r 3 -i "+imageSlicesDir+"K%04d.jpg -i "+mp3SampleFile+" -absf aac_adtstoasc -strict -2 "+outfile)

def test_voice():
    sample_mp3 = "./datasets/first_run/sample/specified01.mp3"
    gen_sample_voices = "./datasets/first_run/sample/gen_sample_voices/"
    sample_voice_process(gen_sample_voices,sample_mp3)

if __name__ == '__main__':
    test_voice()
