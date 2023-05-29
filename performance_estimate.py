
import os

import keras.saving.saving_api
import cv2
import numpy as np

emotion = {
    0: "angry",
    1: "disgust",
    2: "fear",
    3: "happy",
    4: "sad",
    5: "surprised",
    6: "normal"
}

performance_matrix = [[0 for i in range(7)] for i in range(7)]
print(performance_matrix)

if __name__ == '__main__':
    model = keras.saving.saving_api.load_model("./myModel_2.h5")    # 读取模型（训练结束以后再调用）
    # model.summary()
    dir = "./image/test/"
    for cls in range(7):
        cls_dir = os.path.join(dir, str(cls))
        img_list = os.listdir(cls_dir)
        for file in img_list:
            cur_path = os.path.join(cls_dir, file)
            img_input = cv2.imread(cur_path)
            img_input = np.reshape(img_input, (1,) + img_input.shape) / 255.
            # print(img_input.shape)
            predict = model.predict(img_input, batch_size=1)
            y_predict = np.argmax(predict)
            performance_matrix[cls][y_predict] += 1
    for cls in range(7):
        base = sum(performance_matrix[cls])
        for col in range(7):
            performance_matrix[cls][col] /= base
    print(performance_matrix)
