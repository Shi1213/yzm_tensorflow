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
zidian={}#形成字典
for i in zidian_1:
    if ord(i)<=57 and ord(i)>=48:
        zidian[i]=ord(i)-48
    elif ord(i)<=122 and ord(i)>=97:
        zidian[i]=ord(i)-87
    else:
        zidian[i]=ord(i)-29
img_mean=tf.constant([0.485,0.456,0.406])
img_std=tf.constant([0.229,0.224,0.225])
def normalize(x,mean,std):
    x=(x-mean)/std
    return x
def unnomalize(x,mean,std):
    x=x*std+mean
    return x
def pre(x_1,y):
    x = tf.io.read_file(x_1)
    x=tf.image.decode_jpeg(x,channels=3)
    x=tf.image.resize(x,[24,72])
    x = tf.cast(x, dtype=tf.float32) / 255.
    x = normalize(x, img_mean, img_std)
    x = tf.cast(x, dtype=tf.float32)
    y = tf.cast(y, dtype=tf.int32)

    return x,y
def zidian_one_hot_connect(y):
    flag = 1
    for i_1 in y:
        a = zidian[i_1]
        a = tf.convert_to_tensor(a, dtype=tf.int32)
        if flag == 1:
            y_1 = tf.one_hot(a, depth=62)
            flag = 0
        else:
            y_1 = tf.concat([y_1, tf.one_hot(a, depth=62)], axis=0)
    return y_1
db_train_x=[]
db_train_y=[]
db_text_x=[]
db_text_y=[]
with open(r"yzm_text\000.csv","r",newline="") as f:
    read=csv.reader(f)
    for i in read:
        db_text_x.append(i[0])
        y=zidian_one_hot_connect(i[1])
        db_text_y.append(y)
f.close()
x_text=tf.data.Dataset.from_tensor_slices((db_text_x,db_text_y))
x_text=x_text.map(pre).batch(300)
model_logits = tf.keras.models.load_model('yzm_logits.h5', compile=False)
model_conv = tf.keras.models.load_model('yzm_conv.h5', compile=False)
ture_1 = 0
num_train = 0
ture_train = 0
num=0
for step, (x, y) in enumerate(x_text):
    logits = model_conv(x, training=False)
    logits = layers.Flatten()(logits)
    logits = model_logits(logits, training=False)
    for gg in range(logits.shape[0]):
        x1 = logits[gg][:62]
        x2 = logits[gg][62:124]
        x3 = logits[gg][124:186]
        x4 = logits[gg][186:248]
        y1 = y[gg][:62]
        y2 = y[gg][62:124]
        y3 = y[gg][124:186]
        y4 = y[gg][186:248]
        if tf.argmax(x1) == tf.argmax(y1):
            if tf.argmax(x2) == tf.argmax(y2):
                if tf.argmax(x3) == tf.argmax(y3):
                    if tf.argmax(x4) == tf.argmax(y4):
                        ture_1 = ture_1 + 1
                        print(yzm_zidian[tf.argmax(x1).numpy()],yzm_zidian[tf.argmax(x2).numpy()],yzm_zidian[tf.argmax(x3).numpy()],
                              yzm_zidian[tf.argmax(x4).numpy()])

    num = x.shape[0] + num
print("true_1", ture_1 / num)