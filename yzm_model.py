import tensorflow as tf
import csv
import glob
from tensorflow import keras
import string
from tensorflow.keras import layers
from  PIL import  Image
import matplotlib.pyplot as plt
zidian_1=string.digits+string.ascii_lowercase+string.ascii_uppercase
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
with open("yzm/000.csv","r",newline="") as f:
    read=csv.reader(f)
    for i in read:
        if i[1][0] != "-" and i[1][1] != "-" and i[1][2] != "-" and i[1][3] != "-":#避免出现9—avg这种
            db_train_x.append(i[0])
            y = zidian_one_hot_connect(i[1])
            db_train_y.append(y)
f.close()
with open(r"yzm_text\000.csv","r",newline="") as f:
    read=csv.reader(f)
    for i in read:
        db_text_x.append(i[0])
        y=zidian_one_hot_connect(i[1])
        db_text_y.append(y)
f.close()
x_tarin=tf.data.Dataset.from_tensor_slices((db_train_x,db_train_y))
x_tarin=x_tarin.map(pre).shuffle(10000).batch(300)
x_text=tf.data.Dataset.from_tensor_slices((db_text_x,db_text_y))
x_text=x_text.map(pre).batch(300)
optimizer=tf.optimizers.Adam(0.0000001)
def main():
    # model_logits = tf.keras.models.load_model('yzm_logits.h5', compile=False)
    # model_conv=tf.keras.models.load_model('yzm_conv.h5', compile=False)
    model_logits = tf.keras.models.load_model('yzm_logits.h5', compile=False)
    model_conv = tf.keras.models.load_model('yzm_conv.h5', compile=False)
    # model_conv = keras.Sequential([
    #     layers.Conv2D(32, 3, padding='same'),
    #     layers.BatchNormalization(),
    #     layers.Activation("relu"),
    #     layers.MaxPool2D(pool_size=[2, 2], strides=2, padding="same"),
    #
    #
    #     layers.Conv2D(32, 3, padding='same'),
    #     layers.BatchNormalization(),
    #     layers.Activation("relu"),
    #     layers.MaxPool2D(pool_size=[2, 2], strides=2, padding="same"),
    #
    #      layers.Conv2D(32, 3, padding='same'),
    #     layers.BatchNormalization(),
    #      layers.Activation("relu"),
    #      layers.MaxPool2D(pool_size=[2, 2], strides=2, padding="same"),
    # ])
    # model_logits = keras.Sequential([
    #     layers.Dropout(0.2),
    #     layers.Dense(800),
    #     layers.Activation("relu"),
    #     layers.Dropout(0.4),
    #     layers.Dense(62 * 4),
    # ])
    # model_conv.build(input_shape=(None, 24, 72, 3))
    # model_logits.build(input_shape=(None, 864))

    variables=model_conv.trainable_variables+model_logits.trainable_variables
    lost_py=[]
    ture_save = 0.4325
    for epoch in range(300000):
        for step,(x,y) in enumerate(x_tarin):
            with tf.GradientTape() as tape:
                logits=model_conv(x,training=True)
                logits=layers.Flatten()(logits)
                logits=model_logits(logits,training=True)
                y=tf.cast(y,dtype=tf.float32)
                loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits,labels=y)
                loss_regular=[]
                for i in variables[::2]:#正则化
                    loss_regular.append((tf.nn.l2_loss(i)))
                loss=tf.reduce_mean(loss)#+0.0000005*tf.reduce_sum(loss_regular)

            grads=tape.gradient(loss,variables)
            optimizer.apply_gradients(zip(grads,variables))

        print(epoch)
        if  epoch %10==0:
            print("loss:",float(loss))
            num=0
            ture_1=0
            num_train=0
            ture_train=0
            for step,(x,y) in enumerate(x_text):
                logits=model_conv(x,training=False)
                logits=layers.Flatten()(logits)
                logits=model_logits(logits,training=False)
                for gg in range(logits.shape[0]):
                    x1 = logits[gg][:62]
                    x2 = logits[gg][62:124]
                    x3 = logits[gg][124:186]
                    x4 = logits[gg][186:248]
                    y1 = y[gg][:62]
                    y2 = y[gg][62:124]
                    y3 = y[gg][124:186]
                    y4 = y[gg][186:248]
                    if tf.argmax(x1)==tf.argmax(y1):
                        if tf.argmax(x2)==tf.argmax(y2):
                            if tf.argmax(x3)==tf.argmax(y3):
                               if tf.argmax(x4) == tf.argmax(y4):
                                    ture_1=ture_1+1

                num=x.shape[0]+num
            print("true_1",ture_1/num)
            if ture_1/num>ture_save:
                model_conv.save("yzm_conv.h5")
                model_logits.save("yzm_logits.h5")
                ture_save=ture_1/num
                print(ture_save)

            # for step,(x,y) in enumerate(x_tarin):
            #     logits=model_conv(x,training=False)
            #     logits=layers.Flatten()(logits)
            #     logits=model_logits(logits,training=False)
            #     for gg in range(logits.shape[0]):
            #         x1 = logits[gg][:62]
            #         x2 = logits[gg][62:124]
            #         x3 = logits[gg][124:186]
            #         x4 = logits[gg][186:248]
            #         y1 = y[gg][:62]
            #         y2 = y[gg][62:124]
            #         y3 = y[gg][124:186]
            #         y4 = y[gg][186:248]
            #         if tf.argmax(x1)==tf.argmax(y1):
            #             if tf.argmax(x2)==tf.argmax(y2):
            #                 if tf.argmax(x3)==tf.argmax(y3):
            #                    if tf.argmax(x4) == tf.argmax(y4):
            #                         ture_train=ture_train+1
            #
            #     num_train=x.shape[0]+num_train
            # print("true_train",ture_train/num_train)
            # plt.plot(lost_py)
            # plt.show()


if __name__=="__main__":
    main()
