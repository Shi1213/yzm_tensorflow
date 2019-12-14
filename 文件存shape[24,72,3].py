import csv
import tensorflow as tf
import numpy as np
import matplotlib.image as mpimg
import string
db_train_x=[]
db_train_y1=[]
db_train_y2=[]
db_train_y3=[]
db_train_y4=[]
zidian_1=string.digits+string.ascii_lowercase+string.ascii_uppercase
zidian={}#形成字典
for i in zidian_1:
    if ord(i)<=57 and ord(i)>=48:
        zidian[i]=ord(i)-48
    elif ord(i)<=122 and ord(i)>=97:
        zidian[i]=ord(i)-87
    else:
        zidian[i]=ord(i)-29
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
with open(r"yzm\000.csv","r",newline="") as f:
    read=csv.reader(f)
    for i in read:
        db_train_x.append(i[0])
        #y=zidian_one_hot_connect(i[1])
        db_train_y1.append(zidian[i[1][0]])
        db_train_y2.append(zidian[i[1][1]])
        db_train_y3.append(zidian[i[1][2]])
        db_train_y4.append(zidian[i[1][3]])
f.close()
img=[]
img_y=[]
print("3")
for i in db_train_x:
    x = open(i, 'rb').read()
    #x = tf.image.decode_jpeg(x, channels=3)
    img.append(x)
a=0
with tf.io.TFRecordWriter("train.tfrecords") as writer:

    for image, label1,label2,label3,label4 in zip(img, db_train_y1,db_train_y2,db_train_y3,db_train_y4):


        a=a+1
        feature = {                             # 建立 tf.train.Feature 字典

            'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[(image)])),  # 图片是一个 bytes 对象

            'label1': tf.train.Feature(int64_list=tf.train.Int64List(value=[label1])),   # lable是一个 bytes 对象
            'label2': tf.train.Feature(int64_list=tf.train.Int64List(value=[label2])),
            'label3': tf.train.Feature(int64_list=tf.train.Int64List(value=[label3])),
            'label4': tf.train.Feature(int64_list=tf.train.Int64List(value=[label4]))

        }

        example = tf.train.Example(features=tf.train.Features(feature=feature)) # 通过字典建立 Example

        writer.write(example.SerializeToString())   # 将Example序列化并写入 TFRecord 文件

    writer.close()
print(a)