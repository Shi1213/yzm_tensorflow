import tensorflow as tf
import csv
import glob
from tensorflow import keras
import string
from tensorflow.keras import layers
from PIL import Image
import numpy as np
zidian_1=string.digits+string.ascii_lowercase+string.ascii_uppercase
yzm_zidian={}
for i in range(len(zidian_1)):
    yzm_zidian[i]=zidian_1[i]
def normalize(x,mean,std):
    x=(x-mean)/std
    return x
def yzm_model(x):
    x=tf.cast(x,dtype=tf.float32)/255.

    #x = tf.convert_to_tensor(x, dtype=tf.float32)
    x = tf.image.resize(x, [24, 72])
    img_mean = tf.constant([0.485, 0.456, 0.406])
    img_std = tf.constant([0.229, 0.224, 0.225])
    x=normalize(x,img_mean,img_std)
    x = tf.expand_dims(x, axis=0)
    model_logits = tf.keras.models.load_model('yzm_logits.h5', compile=False)
    model_conv = tf.keras.models.load_model('yzm_conv.h5', compile=False)
    x=model_conv(x,training=False)
    x = layers.Flatten()(x)
    logits=model_logits(x,training=False)
    x1 = logits[0][:62]
    x2 = logits[0][62:124]
    x3 = logits[0][124:186]
    x4 = logits[0][186:248]
    x1=tf.argmax(x1)
    x2=tf.argmax(x2)
    x3=tf.argmax(x3)
    x4=tf.argmax(x4)

    return [yzm_zidian[x1.numpy()],yzm_zidian[x2.numpy()],yzm_zidian[x3.numpy()],yzm_zidian[x4.numpy()]]


def main():
    x = tf.io.read_file("6xAx.jpg")
    x = tf.image.decode_jpeg(x, channels=3)
    z=yzm_model(x)
    print(z)



if __name__=="__main__":
    main()




