
import pickle

from tensorflow import optimizers
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import BatchNormalization, MaxPooling2D, Dense, Dropout, Flatten, Conv2D
import tensorflow
import matplotlib.pyplot as plt


def train_model(epoch: int = 40):
    gpus = tensorflow.config.experimental.list_physical_devices(device_type='GPU')
    cpus = tensorflow.config.experimental.list_physical_devices(device_type='CPU')
    print(gpus, cpus)

    train_dir = r'./image/train'
    val_dir = r'./image/validate'
    test_dir = r'./image/test'

    train_datagen = ImageDataGenerator(
        rescale=1. / 255,  # 重放缩因子，数值乘以1.0/255（归一化）
        shear_range=0.2,  # 剪切强度（逆时针方向的剪切变换角度）
        zoom_range=0.2,  # 随机缩放的幅度
        horizontal_flip=True  # 进行随机水平翻转
    )
    val_datagen = ImageDataGenerator(rescale=1. / 255)
    test_datagen = ImageDataGenerator(rescale=1. / 255)

    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(48, 48),
        batch_size=128,
        shuffle=True,
        class_mode='categorical'
    )

    validation_generator = test_datagen.flow_from_directory(
        val_dir,
        target_size=(48, 48),
        batch_size=128,
        shuffle=True,
        class_mode='categorical'
    )

    test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=(48, 48),
        batch_size=128,
        shuffle=True,
        class_mode='categorical'
    )

    model = Sequential()
    model.add(
        Conv2D(64, kernel_size=(5, 5), strides=(1, 1), activation='relu', padding='same', input_shape=(48, 48, 3)))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))  # 第一池化层，池化核大小为2×2，步长2
    model.add(BatchNormalization())
    model.add(Dropout(0.4))  # 随机丢弃40%的网络连接，防止过拟合
    # 第二段
    model.add(Conv2D(128, kernel_size=(5, 5), strides=(1, 1), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))
    # 第三段
    model.add(Conv2D(256, kernel_size=(5, 5), strides=(1, 1), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    model.add(Flatten())  # 过渡层
    model.add(Dropout(0.3))
    model.add(Dense(2048, activation='relu'))  # 全连接层
    model.add(Dropout(0.4))
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.4))
    model.add(Dense(512, activation='relu'))
    model.add(Dense(7, activation='softmax'))  # 分类输出层
    model.summary()

    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizers.Adam(),  # Adam优化器
                  # optimizer=optimizers.RMSprop(learning_rate=0.0001),  # rmsprop优化器
                  metrics=['accuracy'])

    history = model.fit(
        train_generator,  # 生成训练集生成器
        steps_per_epoch=243,  # train_num/batch_size=128
        epochs=epoch,  # 数据迭代轮数
        validation_data=validation_generator,  # 生成验证集生成器
        validation_steps=28  # valid_num/batch_size=128
    )

    test_loss, test_acc = model.evaluate(test_generator, steps=28)
    print("test_loss: %.4f - test_acc: %.4f" % (test_loss, test_acc * 100))

    # 保存模型
    model_json = model.to_json()
    with open('myModel_2_json.json', 'w') as json_file:
        json_file.write(model_json)
    model.save_weights('myModel_2_weight.h5')
    model.save('myModel_2.h5')

    with open('fit_2_log.txt', 'wb') as file_txt:
        pickle.dump(history.history, file_txt, 0)

    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(1, len(acc) + 1)

    plt.figure("acc")
    plt.plot(epochs, acc, 'r-', label='Training acc')
    plt.plot(epochs, val_acc, 'b', label='validation acc')
    plt.title('Accuracy curve')
    plt.legend()
    plt.savefig('acc_2.jpg')

    plt.figure("loss")
    plt.plot(epochs, loss, 'r-', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='validation loss')
    plt.title('Loss curve')
    plt.legend()
    plt.savefig('loss_2.jpg')
    plt.show()

