from mmpretrain import ImageClassificationInferencer
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


inference = ImageClassificationInferencer("./resnet50_8xb32_in1k.py", "./exp/epoch_100.pth")
inference("test.jpeg", show_dir="./visualize")
