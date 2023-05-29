import cv2 as cv
import os
import random


# 图片输出路径
out_dir = './my_faces'
if not os.path.exists(out_dir):
    os.makedirs(out_dir)


def get_img_face(img, haar):
    """
    从照片中提取人脸
    :param img: 人脸提取目标图片
    :param haar: 分类器
    :return:
    """
    gray_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    faces = haar.detectMultiScale(img, 1.3, 5)
    # 图片输出名字前缀
    prefix = random.randint(1, 100000)
    if len(faces) == 0:
        return
    for f_x, f_y, f_w, f_h in faces:
        face = gray_img[f_y: f_y + f_h, f_x: f_x + f_w]
        face = cv.resize(face, (size_m, size_n))
        # 图片输出名字后缀
        suffix = random.randint(1, 100000)
        cv.imwrite(out_dir + '/' + str(prefix) + str(suffix) + '.jpg', face)


def get_camera_face(n, haar):
    """
    从摄像头中提取人脸
    :param n: 提取照片数量
    :param haar: 分类器
    :return:
    """
    camera = cv.VideoCapture(0)
    # 图片输出名字前缀
    order = random.randint(1, 10000)
    cur = 1
    while cur <= n:
        print('It`s processing %s image.' % cur)
        success, cur_img = camera.read()
        gray_img = cv.cvtColor(cur_img, cv.COLOR_RGB2GRAY)
        faces = haar.detectMultiScale(gray_img, 1.3, 5)
        for f_x, f_y, f_w, f_h in faces:
            face = gray_img[f_y: f_y + f_h, f_x: f_x + f_w]
            face = cv.resize(face, (size_m, size_n))
            cv.imwrite(out_dir + '/' + str(order) + str(cur) + '.jpg', face)
            cur += 1


if __name__ == '__main__':

    # 裁剪尺寸大小
    size_m = 48
    size_n = 48

    # 读入图片
    img = cv.imread("./face1.jpg")
    # 获取分类器
    haar = cv.CascadeClassifier('./haarcascade_frontalface_default.xml')
    # 测试
    get_img_face(img, haar)
