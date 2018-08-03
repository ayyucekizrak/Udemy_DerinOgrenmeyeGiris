# -*- coding: utf-8 -*- 
'''
Deep Learning Türkiye topluluğu tarafından hazırlanmıştır.

Amaç: El yazısı rakamların tanınması.
Veriseti: MNIST (http://yann.lecun.com/exdb/mnist/)
Algoritma: Evrişimli Sinir Ağları (Convolutional Neural Networks)
Microsoft Azure Notebook: https://notebooks.azure.com/deeplearningturkiye/libraries/pratik-derin-ogrenme/html/rakam_tanima_CNN_MNIST.ipynb

Ağ Mimarisi:

- 32 x 3 x 3 CONV
- 64 x 3 x 4 CONV
- 2 x 2 MAX POOL
- DROPOUT (%25)
- 128 FC
- DROPOUT (%50)
- 10 FC


12 epoch sonunda 99.25% test doğruluk oranı elde ediliyor.
'''

from __future__ import print_function
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K

batch_size = 128 # her bir iterasyonda "128" resim alınsın
num_classes = 10 # ayırt etmek istediğimiz "10" rakam
epochs = 12 # eğitim 12 epoch sürsün

# giriş resimlerinin boyutları 28 x 28 piksel
img_rows, img_cols = 28, 28

# veri önce karıştırılıyor (shuffle) sonra da eğitim/test diye ayrılıyor
(x_train, y_train), (x_test, y_test) = mnist.load_data()

if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# sınıf vektörleri ikili (binary) formununa dönüştürülür
# "to_catogorical" fonksiyonu ile one-hot-encoding yapıyoruz
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

model = Sequential()

# 3x3 boyutunda 32 adet filtreden oluşan ReLU aktivasyonlu CONV katmanı ekleyelim. 
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape))

# 3x3 boyutunda 64 adet filtreden oluşan ReLU aktivasyonlu CONV katmanı ekleyelim. 
model.add(Conv2D(64, (3, 3), activation='relu'))

# 2x2 boyutlu çerçeveden oluşan MAXPOOL katmanı ekleyelim. 
model.add(MaxPooling2D(pool_size=(2, 2)))

# her seferinde nöronların %25'i atılsın (drop)
model.add(Dropout(0.25))

# Tam bağlantılı (fully connected) katmanına geçiş olacağı için düzleştirme yapalım 
model.add(Flatten())

# 128 nörondan oluşan ReLU aktivasyonu FC katmanı ekleyelim 
model.add(Dense(128, activation='relu'))

# Her seferinde %50'sini atalım (drop)
model.add(Dropout(0.5))

# Çıkış katmanına sınıf sayısı kadar (10) Softmax aktivasyonlu nöron ekleyelim
model.add(Dense(num_classes, activation='softmax'))

# Adadelta optimizasyon yöntemini ve cross entropy yitim (loss) fonksiyonunu kullanalım.
model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

# eğitim işlemini gerçekleştirelim
model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_test, y_test))

# test işlemini gerçekleştirelim ve sonuçları ekrana yazdıralım
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
