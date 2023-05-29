import csv
import os
import numpy as np
from PIL import Image
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import img_to_array, load_img

train_csv_file = "./data/preprocess/train.csv"
validate_csv_file = "./data/preprocess/validate.csv"
test_csv_file = "./data/preprocess/test.csv"

train_image_dir = "./image/train"
validate_image_dir = "./image/validate"
test_image_dir = "./image/test"


def load_csv(csv_file: str) -> None:
    with open(csv_file) as file:
        csv_data = csv.reader(file)
        header = next(csv_data)
        rows = [row for row in csv_data]
        train = [row[:-1] for row in rows if row[-1] == "Training"]
        csv.writer(open(train_csv_file, "w+"), lineterminator='\n').writerows([header[:-1]] + train)
        validate = [row[:-1] for row in rows if row[-1] == "PublicTest"]
        csv.writer(open(validate_csv_file, "w+"), lineterminator='\n').writerows([header[:-1]] + validate)
        test = [row[:-1] for row in rows if row[-1] == "PrivateTest"]
        csv.writer(open(test_csv_file, "w+"), lineterminator='\n').writerows([header[:-1]] + test)


def convert_csv_to_image() -> None:
    for save_path, csv_file in ((train_image_dir, train_csv_file), (validate_image_dir, validate_csv_file), (test_image_dir, test_csv_file)):
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        with open(csv_file) as file:
            csv_data = csv.reader(file)
            header = next(csv_data)
            for i, (label, pixel) in enumerate(csv_data):
                pixel = np.asarray([float(p) for p in pixel.split()]).reshape(48, 48)
                sub_dir = os.path.join(save_path, label)
                if not os.path.exists(sub_dir):
                    os.makedirs(sub_dir)
                img = Image.fromarray(pixel).convert('L')
                image_name = os.path.join(sub_dir, f"{i:05d}.jpg")
                print(image_name)
                img.save(image_name)


def dataset_augmenting(augment_dir: str):
    datagen = ImageDataGenerator(
        rotation_range=20,  # 旋转范围
        width_shift_range=0.1,  # 水平平移范围
        height_shift_range=0.1,  # 垂直平移范围
        shear_range=0.1,  # 透视变换的范围
        zoom_range=0.1,  # 缩放范围
        horizontal_flip=True,  # 水平反转
        fill_mode='nearest')

    for filename in os.listdir(augment_dir):
        print(filename)
        img = load_img(f"{augment_dir}/{filename}")
        x = img_to_array(img)
        x = x.reshape((1,) + x.shape)
        i = 0
        for batch in datagen.flow(x, batch_size=1, save_to_dir=augment_dir, save_prefix="augment", save_format="jpg"):
            i += 1
            if i > 5:
                break
