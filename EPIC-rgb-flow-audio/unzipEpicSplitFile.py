from mmaction.datasets.pipelines import Compose
import torch.utils.data
import pandas as pd
import soundfile as sf
from scipy import signal
import numpy as np

base_path="D:\ML\Dataset\EPIC_KITCHENS"
source_dom1 = "D3"
split="test"
train_file = pd.read_pickle(base_path + '/' + 'MM-SADA_Domain_Adaptation_Splits/'+source_dom1+"_"+split+".pkl")
source_1 = [0, 1, 2, 4, 5, 6, 7]
# 帮我创建一个csv文件，逐行保存data1中的数据
with open(base_path + '/' + 'MM-SADA_Domain_Adaptation_Splits/'+source_dom1+"_"+split+".csv", 'w') as f:
    for _, line in train_file.iterrows():
        image = [source_dom1 + '/' + line['video_id'], line['start_frame'], line['stop_frame'], line['start_timestamp'], line['stop_timestamp']]
        labels = line['verb_class']
        if int(labels) in source_1: # source_1=[0, 1, 2, 4, 5, 6, 7]
            if int(labels) == 7:
                labels = 3 # 将编号为7的重映射为了3。为什么要这样操作，训练时直接取source_1=[0, 1, 2, 3, 4, 5, 6]不是更简单么？
            # data1.append((image[0], image[1], image[2], image[3], image[4], int(labels)))
            f.write(f"{image[0]},{image[1]},{image[2]},{image[3]},{image[4]},{int(labels)}\n")
print("CSV文件已保存") 