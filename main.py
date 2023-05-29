import sys

import cv2
import keras.saving.save
import numpy as np

import util.loading
from train import train_model

emotion = {
    0: "angry",
    1: "disgust",
    2: "fear",
    3: "happy",
    4: "sad",
    5: "surprised",
    6: "normal"
}


def preprocessing() -> None:
    util.loading.load_csv("data/fer2013.csv")                       # 将数据集拆解为测试集、验证集和测试集
    util.loading.convert_csv_to_image()                             # 将 csv 中的图像数据保存为图片格式
    util.loading.dataset_augmenting("./image/train/1")              # 注意到 disgust 分组数据较少，做数据增强


if __name__ == '__main__':
    # preprocessing()                                               # 数据预处理（只需要执行一次）
    # train_model(epoch=80)                                           # 训练模型（只需要执行一次）
    img_path = ""
    # 后端程序调用：python main.py -i #{path}
    # if sys.argv[0] is not "-i":
    #     print("Illegal command!")
    #     exit(-1)
    # else:
    #     img_path = sys.argv[1]
    model = keras.saving.save.load_model("./myModel_2.h5")          # 读取模型（训练结束以后再调用）
    # model.summary()
    img_input = cv2.imread("./my_faces/2503729901.jpg")
    img_input = np.reshape(img_input, (1,) + img_input.shape) / 255.
    # print(img_input.shape)
    predict = model.predict(img_input, batch_size=1)
    y_predict = np.argmax(predict)
    result_json = f"{{\n\t\"angry\": {predict[0, 0]}, \n\t\"disgust\": {predict[0, 1]}, \n\t\"fear\": {predict[0][2]}, \n\t\"happy\": {predict[0][3]}, \n\t\"sad\": {predict[0][4]}, \n\t\"surprised\": {predict[0][5]}, \n\t\"normal\": {predict[0][6]}, \n\t\"prediction\": \"{emotion[y_predict]}\"\n}}"
    print(result_json)
