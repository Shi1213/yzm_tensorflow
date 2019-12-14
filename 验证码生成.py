import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from captcha.image import ImageCaptcha
import string
import random
import csv
#随机生成验证码
yzm_1={}
def pan(kk):
   for i in range(len(kk)-1):
      if kk[i]>="0" and kk[i]<="9":
         if kk[i+1]=="e" or kk[i+1]=="E":
            return 0
   if kk[0]=="0" and kk[1]>="0"and k[2]>="0"and k[3]>="0"and kk[1]<="9"and k[2]<="9"and k[3]<="9":
      return 0
   return 1
for i in range(100000):
   image=ImageCaptcha()
   x=string.digits+string.ascii_uppercase+string.ascii_lowercase#生成字母
   k=[]
   for i in range(4):
      k.append(random.choice(x))
   m="".join(k)
   kk=m
   m=image.generate_image(m)
   m=np.array(m)
   m=tf.convert_to_tensor(m)
   m=tf.image.resize(m,[24,72])
   m=m.numpy().astype(np.uint8)
   im=Image.fromarray(m)
   if pan(kk)==1:
     try:
        im.save("yzm/%s.jpg" % kk)
        yzm_1["yzm/" + str(kk) + ".jpg"] = str(kk)
     except:
        print(1)
with open("yzm/000.csv","w",newline="") as f:
   writer=csv.writer(f)
   for i in yzm_1:
      writer.writerow([i,str(yzm_1[i])])
f.close()
#yzm_model的两个基本模型
# model_conv = keras.Sequential([
#     layers.Conv2D(16, [3, 3], strides=2, padding='same'),
#     layers.BatchNormalization(),
#     layers.Activation("relu"),
#     layers.MaxPool2D(pool_size=[2, 2], strides=2, padding="same"),
#     layers.Conv2D(32, [3, 3], strides=2, padding='same'),
#     layers.BatchNormalization(),
#     layers.Activation("relu"),
#     layers.MaxPool2D(pool_size=[2, 2], strides=1, padding="same"),
# ])
# model_logits = keras.Sequential([
#     layers.Dropout(0.6),
#     layers.Dense(150),
#     layers.Activation("relu"),
#     layers.Dense(62 * 4),
# ])
# model_conv.build(input_shape=(None, 24, 72, 3))
# model_logits.build(input_shape=(None, 864))