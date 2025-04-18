from mmaction.datasets.pipelines import Compose
import torch.utils.data
import pandas as pd
import soundfile as sf
from scipy import signal
import numpy as np

source_1 = [0, 1, 2, 4, 5, 6, 7]
source_2 = [0, 1, 2, 4, 5, 6, 7]
source_all = [0, 1, 2, 4, 5, 6, 7]
target_all = [0, 1, 2, 3, 4, 5, 6, 7]
# EPIC数据集
class EPICDOMAIN(torch.utils.data.Dataset):
    def __init__(self, split='train', domain_name='source', domain=['D1'], modality='rgb', cfg=None, cfg_flow=None, sample_dur=10, use_video=True, use_flow=True, use_audio=True, datapath='/path/to/EPIC-KITCHENS/'):
        self.base_path = datapath
        self.split = split
        self.modality = modality
        self.interval = 9
        self.sample_dur = sample_dur
        self.use_video = use_video
        self.use_audio = use_audio
        self.use_flow = use_flow

        # build the data pipeline
        if split == 'train':
            if self.use_video:
                train_pipeline = cfg.data.train.pipeline
                self.pipeline = Compose(train_pipeline)
            if self.use_flow:
                train_pipeline_flow = cfg_flow.data.train.pipeline
                self.pipeline_flow = Compose(train_pipeline_flow)
        else:
            if self.use_video:
                val_pipeline = cfg.data.val.pipeline
                self.pipeline = Compose(val_pipeline) # Compose 是MMAction2 框架的一部分，用于将多个数据处理步骤组合成一个数据处理管道。
            if self.use_flow:
                val_pipeline_flow = cfg_flow.data.val.pipeline
                self.pipeline_flow = Compose(val_pipeline_flow)
        # 样本量统计：(可进一步统计，哪些类别样本量较少，哪些样本量较多；哪些样本容易分类错误)
        # D1_train: 1415   D1_test: 404
        # D2_train: 2281   D2_test: 694
        # D3_train: 3530   D3_test: 903
        data1 = []
        # 加载源域数据,EPIC共3个域，其中2个域作为source,另外1个域作为target
        # 这里重点是加载视频模态切分的数据
        if domain_name == 'source':
            source_dom1 = domain[0]#源域1
            train_file = pd.read_pickle(self.base_path + '/' + 'MM-SADA_Domain_Adaptation_Splits/'+source_dom1+"_"+split+".pkl")
            for _, line in train_file.iterrows():
                # 获取该domain下视频数据对应的：视频id、开始帧、结束帧、开始时间、结束时间
                image = [source_dom1 + '/' + line['video_id'], line['start_frame'], line['stop_frame'], line['start_timestamp'], line['stop_timestamp']]
                # 获取该domain下视频数据对应的标签
                labels = line['verb_class']
                # 如果标签7在source_1中，则将标签重映射为source_1中的标签3
                if int(labels) in source_1: # source_1=[0, 1, 2, 4, 5, 6, 7]
                    if int(labels) == 7:
                        labels = 3 # 将编号为7的重映射为了3。测试时，将标签3与7进行了互换
                    data1.append((image[0], image[1], image[2], image[3], image[4], int(labels)))

            source_dom2 = domain[1]#源域2
            train_file = pd.read_pickle(self.base_path +  '/' +'MM-SADA_Domain_Adaptation_Splits/'+source_dom2+"_"+split+".pkl")
            for _, line in train_file.iterrows():
                image = [source_dom2 + '/' + line['video_id'], line['start_frame'], line['stop_frame'], line['start_timestamp'], line['stop_timestamp']]
                labels = line['verb_class']
                if int(labels) in source_2:
                    if int(labels) == 7:
                        labels = 3
                    data1.append((image[0], image[1], image[2], image[3], image[4], int(labels)))
        else:
            target_dom = domain[0]
            train_file = pd.read_pickle(self.base_path + '/' + 'MM-SADA_Domain_Adaptation_Splits/'+target_dom+"_"+split+".pkl")
            for _, line in train_file.iterrows():
                image = [target_dom + '/' + line['video_id'], line['start_frame'], line['stop_frame'], line['start_timestamp'], line['stop_timestamp']]
                labels = line['verb_class']
                if int(labels) in target_all:#测试时，原来为3和7的标签进行了互换
                    if int(labels) == 3:
                        labels = 7
                    elif int(labels) == 7:
                        labels = 3
                    data1.append((image[0], image[1], image[2], image[3], image[4], int(labels)))
        # 将视频模态切分的数据保存到self.samples中
        self.samples = data1
        self.cfg = cfg
        self.cfg_flow = cfg_flow

    def __getitem__(self, index):
        video_path = self.base_path +'/rgb/'+self.split + '/'+self.samples[index][0] # 视频路径，self.samples[index][0] 是视频id
        flow_path = self.base_path +'/flow/'+self.split + '/'+self.samples[index][0]
        # 加载index对应的视频模态数据
        if self.use_video:
            filename_tmpl = self.cfg.data.train.get('filename_tmpl', 'frame_{:010}.jpg') # 获取 'filename_tmpl' 的值，如果找不到这个键，则返回 'frame_{:010}.jpg'
            modality = self.cfg.data.train.get('modality', 'RGB') # 在 MMAction2 框架中，所有地方都使用 'RGB' 作为标准表示，因此这里返回 'RGB'
            start_index = self.cfg.data.train.get('start_index', int(self.samples[index][1])) # 视频模态切分的开始帧
            # MMAction2 提供了 RawFrameDataset 类来处理视频帧数据，进行自动加载，提供如下格式数据即可
            #   data = dict(
            #     frame_dir=video_path,      # 视频帧所在的目录
            #     total_frames=total_frames, # 总帧数
            #     start_index=start_index,   # 开始帧
            #     filename_tmpl=filename_tmpl # 文件名模板
            # )
            data = dict(
                frame_dir=video_path,#视频路径
                total_frames=int(self.samples[index][2] - self.samples[index][1]),#视频模态切分的总帧数
                label=-1,#这里的 label=-1 是一个占位符，表示在数据加载和预处理（当前阶段）暂时不需要标签信息，实际的标签会在后续的处理阶段使用
                start_index=start_index,#开始帧
                filename_tmpl=filename_tmpl,#文件名模板
                modality=modality)# 在 MMAction2 框架中，所有地方都使用 'RGB' 作为标准表示，因此这里返回 'RGB'
            data = self.pipeline(data)#通过pipeline对数据进行了预处理，其中视频模态处理为32帧
        '''
        train_pipeline = [
                dict(type='SampleFrames', clip_len=32, frame_interval=2, num_clips=1),# 参数说明：clip_len=32：每个片段包含32帧，frame_interval=2：每隔2帧采样一次，num_clips=1：每个视频只采样一个片段，实际效果：从视频中提取32帧，总共覆盖约64帧的时间范围
                dict(type='RawFrameDecode'),#参数说明：解码原始视频帧。功能：将压缩的视频帧转换为可处理的图像数据
                dict(type='Resize', scale=(-1, 256)),#目的：统一输入尺寸，同时保持图像比例。保持宽高比，将短边调整为256像素
                dict(type='RandomResizedCrop'),
                dict(type='Resize', scale=(224, 224), keep_ratio=False),#目的：统一输入尺寸，符合模型要求
                dict(type='Flip', flip_ratio=0.5),#随机水平翻转图像
                dict(type='Normalize', **img_norm_cfg),#对图像进行标准化处理
                dict(type='FormatShape', input_format='NCTHW'),#NCTHW：批次大小(N) x 通道数(C) x 时间维度(T) x 高度(H) x 宽度(W)
                dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),#收集需要的字段.keys=['imgs', 'label']：收集图像和标签.meta_keys=[]：不收集元数据
                dict(type='ToTensor', keys=['imgs', 'label'])#转换为张量
            ]
        '''

        # MMAction2 框架会自动处理光流数据的加载
        # 1.光流数据的存储结构：
        #   u 文件夹：存储 x 方向的光流图
        #   v 文件夹：存储 y 方向的光流图
        # 2.MMAction2 的自动处理：
        #   当设置 modality='Flow' 时，MMAction2 会自动：识别 u 和 v 文件夹，加载对应帧的 x 和 y 方向光流图，将两个方向的光流图组合成完整的光流信息
        if self.use_flow:
            filename_tmpl_flow = self.cfg_flow.data.train.get('filename_tmpl', 'frame_{:010}.jpg')#光流文件模板名
            modality_flow = self.cfg_flow.data.train.get('modality', 'Flow')#光流模态
            # 研究表明，光流下采样后识别准确率几乎不变。在多模态融合中，光流模态主要提供运动信息，而不需要与RGB帧一样高的时间精度，所以使用更低的帧率是合理的设计决策。并且其采样是EPIC标准数据集中已经整理好的
            start_index_flow = self.cfg_flow.data.train.get('start_index', int(np.ceil(self.samples[index][1] / 2)))#光流模态切分的开始帧，为视频模态开始帧的一半（在下载的数据中光流数据的文件已进行了重新编码）
            flow = dict(
                frame_dir=flow_path,
                total_frames=int((self.samples[index][2] - self.samples[index][1])/2),#光流模态切分的总帧数，为视频帧总帧数的一半
                label=-1,
                start_index=start_index_flow,
                filename_tmpl=filename_tmpl_flow,
                modality=modality_flow)
            flow = self.pipeline_flow(flow)
        label1 = self.samples[index][-1] # 真实标签

        if self.use_audio:
            audio_path = self.base_path + '/audio/' + self.split + '/' + self.samples[index][0] + '.wav'
            samples, samplerate = sf.read(audio_path)
            # samples：音频数据，通常是一个 NumPy 数组对于单声道音频：形状为 (n_samples,)，采样率samplerate=16000，表示每秒采样的次数（单位：Hz）
            duration = len(samples) / samplerate
            # 从成员属性self.samples中获取动作的开始时间
            fr_sec = self.samples[index][3].split(':')#动作的开始时间
            hour1 = float(fr_sec[0])
            minu1 = float(fr_sec[1])
            sec1 = float(fr_sec[2])
            fr_sec = (hour1 * 60 + minu1) * 60 + sec1#开始的秒数

            stop_sec = self.samples[index][4].split(':')#动作的结束时间
            hour1 = float(stop_sec[0])
            minu1 = float(stop_sec[1])
            sec1 = float(stop_sec[2])
            stop_sec = (hour1 * 60 + minu1) * 60 + sec1#结束的秒数

            start1 = fr_sec / duration * len(samples)#开始的索引
            end1 = stop_sec / duration * len(samples)#结束的索引
            start1 = int(np.round(start1))
            end1 = int(np.round(end1))
            samples = samples[start1:end1]#音频对应的数据，为一维数组

            resamples = samples[:160000]#截取前10秒的数据
            while len(resamples) < 160000:#数据不够时，进行重复填充
                resamples = np.tile(resamples, 10)[:160000]
            # 数据进行截断处理，保证在正常范围内
            resamples[resamples > 1.] = 1.
            resamples[resamples < -1.] = -1.
            frequencies, times, spectrogram = signal.spectrogram(resamples, samplerate, nperseg=512, noverlap=353)#生成音频频谱图（spectrogram）。nperseg=512：窗口大小为512个样本，noverlap=353：窗口重叠为353个样本。返回值spectrogram：频谱图数据（二维数组）
            spectrogram = np.log(spectrogram + 1e-7)#对频谱图进行对数变换，便于后续处理。spectrogram：(257, 1004)
            # 数据归一化处理
            mean = np.mean(spectrogram)
            std = np.std(spectrogram)
            spectrogram = np.divide(spectrogram - mean, std + 1e-9)
            if self.split == 'train':#训练时，添加噪声（数据增强）
                noise = np.random.uniform(-0.05, 0.05, spectrogram.shape)#加入的是均匀分布的噪声，将噪声添加到频谱图中。目的是模拟真实环境中的背景声，增加数据的多样性
                spectrogram = spectrogram + noise
                start1 = np.random.choice(256 - self.interval, (1,))[0]#随机遮挡（Random Masking）：随机选择一个起始位置，将一段连续的频率区域设置为0（遮挡）。起始位置在[0, 256 - interval]范围内随机选择，模拟音频信号中的缺失或干扰
                spectrogram[start1:(start1 + self.interval), :] = 0#对这个范围内的数据进行遮挡. self.interval=9

        if self.use_video and self.use_flow and self.use_audio:
            return data, flow, spectrogram.astype(np.float32), label1
        elif self.use_video and self.use_flow:
            return data, flow, 0, label1
        elif self.use_video and self.use_audio:
            return data, 0, spectrogram.astype(np.float32), label1
        elif self.use_flow and self.use_audio:
            return 0, flow, spectrogram.astype(np.float32), label1

    def __len__(self):
        return len(self.samples)

