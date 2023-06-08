import os
import shutil
import random


def split_data(data_path, ration=0.2):
    if not os.path.exists("train"):
        os.mkdir("train")
    if not os.path.exists("val"):
        os.mkdir("val")
    for folder in os.listdir(data_path):
        for fn in os.listdir(os.path.join(data_path, folder)):
            if random.random() > ration:
                if not os.path.exists(os.path.join("train", folder)):
                    os.mkdir(os.path.join("train", folder))
                shutil.copyfile(os.path.join(data_path, folder, fn), os.path.join("train", folder, fn))
                print(f"{fn}已写入{os.path.join('train', folder, fn)}")
            else:
                if not os.path.exists(os.path.join("val", folder)):
                    os.mkdir(os.path.join("val", folder))
                shutil.copyfile(os.path.join(data_path, folder, fn), os.path.join("val", folder, fn))
                print(f"{fn}已写入{os.path.join('val', folder, fn)}")


if __name__ == "__main__":
    split_data("data")
