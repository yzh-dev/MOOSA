from mmaction.apis import init_recognizer
import torch
import argparse
import tqdm
import os
import numpy as np
import torch.nn as nn
import random
from VGGSound.model import AVENet
from VGGSound.models.resnet import AudioAttGenModule
from VGGSound.test import get_arguments
from dataloader_video_flow_audio_EPIC_MOOSA import EPICDOMAIN
import torch.nn.functional as F
from losses import SupConLoss
from torch.distributions import Categorical
import itertools
import math
source_1 = [0, 1, 2, 4, 5, 6, 7]  # EPIC-Kitchens dataset includes eight actions (‘put’, ‘take’, ‘open’, ‘close’, ‘wash’, ‘cut’, ‘mix’, and ‘pour’)
source_2 = [0, 1, 2, 4, 5, 6, 7]
source_all = [0, 1, 2, 4, 5, 6, 7]  # 训练集的类别少了一个类别3（数据加载模块中对类别3和类别7进行了重映射）
target_all = [0, 1, 2, 3, 4, 5, 6, 7]

def train_one_step(clip, labels, flow, spectrogram):
    labels = labels.cuda()
    print(labels)
    if args.use_video:
        clip = clip['imgs'].cuda().squeeze(1) # clip输入帧率较高（如32帧）， shape是[batch_size, 3, 32, 224, 224]，维度：[batch_size, channels, num_frames, height, width]

    # 光流（Optical Flow）表示图像中像素点的运动
    # 每个像素点都有两个方向的运动分量：
    #   x方向：水平运动
    #   y方向：垂直运动
    if args.use_flow:# 光流数据已经编码了运动信息,不需要高帧率来捕捉快速运动
        flow = flow['imgs'].cuda().squeeze(1) # 这里的flow 输入帧率较低（如8帧），shape是 [batch_size, 2, 8, 224, 224]，维度：[batch_size, channels, num_frames, height, width]

    # 1. batch_size: 一次处理的音频样本数
    # 2. 1: 单通道音频
    # 3. 257: 频率维度（FFT点数/2 + 1）
    # 4. 1004: 时间步数    
    if args.use_audio:
        spectrogram = spectrogram.unsqueeze(1).cuda() # 音频，shape是 [batch_size, 1, 257,1004]，维度:

    with torch.no_grad():
        if args.use_flow:
            f_feat = model_flow.module.backbone.get_feature(flow)
        if args.use_video:
            x_slow, x_fast = model.module.backbone.get_feature(clip)  #输出layer3输出的中间层特征，低级特征，包含更多细节
            v_feat = (x_slow.detach(), x_fast.detach()) # 元组
        if args.use_audio:#频率维度 257 对应音频的频域特征，时间维度 1004 对应音频的时域特征
            _, audio_feat, _ = audio_model(spectrogram) # audio_feat shape是 [batch_size, 1, 257,1004]

    if args.use_video:
        # 输入v_feat包含慢速特征x_slow：torch.Size([16, 1280, 8, 14, 14])和快速特征x_fast：torch.Size([16, 128, 32, 14, 14])
        v_feat = model.module.backbone.get_predict(v_feat)# layer3->layer4的输出高级特征，更具语义信息，输出v_feat为元组，v_feat[0] shape是torch.Size([16, 2048, 8, 7, 7])，v_feat[1] shape是torch.Size([16, 256, 32, 7, 7])
        predict1, v_emd = model.module.cls_head(v_feat)#predict1:torch.Size([16, 7]),v_emd shape是 [batch_size, 2304]
        v_dim = int(v_emd.shape[1] / 2)  # 视频的一半特征作为domain-shared特征
        entropyp = Categorical(probs = nn.Softmax(dim=1)(predict1)).entropy().reshape(-1,1) #这段代码的作用是对预测结果 predict1 进行 Softmax 操作，得到概率分布，然后计算该概率分布的熵，并将熵值调整为列向量形式。最终得到的 entropyp 是一个包含每个样本熵值的张量
        output_loss1 = criterion(predict1, labels)#利用全特征计算video模态下的损失
        video_parts = torch.split(v_emd, v_emd.shape[1] // args.jigsaw_num_splits, dim=1)  # 将视频domain-shared特征，切分为P=jigsaw_num_splits=2个部分，用于随机打乱
        output_u_v = F.softmax(predict1, 1)
        loss_u_v = (-output_u_v * torch.log(output_u_v + 1e-5)).sum(1).mean() #计算预测结果的熵最小化损失

    if args.use_flow:
        f_feat = model_flow.module.backbone.get_predict(f_feat.detach())
        f_predict, f_emd = model_flow.module.cls_head(f_feat)
        f_dim = int(f_emd.shape[1] / 2)
        entropyf = Categorical(probs = nn.Softmax(dim=1)(f_predict)).entropy().reshape(-1,1)
        output_loss3 = criterion(f_predict, labels)
        flow_parts = torch.split(f_emd, f_emd.shape[1] // args.jigsaw_num_splits, dim=1)
        output_u_f = F.softmax(f_predict, 1)
        loss_u_f = (-output_u_f * torch.log(output_u_f + 1e-5)).sum(1).mean()

    if args.use_audio:    
        audio_predict, audio_emd = audio_cls_model(audio_feat.detach())#audio_predict:torch.Size([16, 7]),audio_emd:torch.Size([16, 512]),audio_feat:torch.Size([16, 256, 17, 63])
        a_dim = int(audio_emd.shape[1] / 2)
        entropya = Categorical(probs = nn.Softmax(dim=1)(audio_predict)).entropy().reshape(-1,1)
        output_loss2 = criterion(audio_predict, labels)
        audio_parts = torch.split(audio_emd, audio_emd.shape[1] // args.jigsaw_num_splits, dim=1)
        output_u_a = F.softmax(audio_predict, 1)
        loss_u_a = (-output_u_a * torch.log(output_u_a + 1e-5)).sum(1).mean()

    if args.use_video and args.use_flow and args.use_audio:
        feat = torch.cat((v_emd, audio_emd, f_emd), dim=1)  # 拼接各个模态domain-shared的特征
        parts = video_parts + audio_parts + flow_parts
    elif args.use_video and args.use_flow:
        feat = torch.cat((v_emd, f_emd), dim=1)
        parts = video_parts + flow_parts
    elif args.use_video and args.use_audio:
        feat = torch.cat((v_emd, audio_emd), dim=1)
        parts = video_parts + audio_parts 
    elif args.use_flow and args.use_audio:
        feat = torch.cat((f_emd, audio_emd), dim=1)
        parts = flow_parts + audio_parts

    predict = mlp_cls(feat)
    output_loss4 = criterion(predict, labels)
    entropypa = Categorical(probs = nn.Softmax(dim=1)(predict)).entropy().reshape(-1,1)
    output_u = F.softmax(predict, 1)#全模态的预测结果
    loss_u = (-output_u * torch.log(output_u + 1e-5)).sum(1).mean()

    # Entropy Weighting
    if args.use_video and args.use_flow and args.use_audio:
        entropy = -torch.cat((entropyp, entropya, entropyf, entropypa), 1)  # 拼接各个模态的熵值，[batch,4(模态数)]
        output = nn.Softmax(dim=1)(entropy/args.entropy_weight_temp)#计算各个模态损失权重
        loss = torch.mean(output[:,0]*output_loss1+output[:,1]*output_loss2+output[:,2]*output_loss3+output[:,3]*output_loss4)
        loss_ent_min = (loss_u + loss_u_v + loss_u_a + loss_u_f) * args.entropy_min_weight / 4
    elif args.use_video and args.use_flow:
        entropy = -torch.cat((entropyp, entropyf, entropypa), 1)
        output = nn.Softmax(dim=1)(entropy/args.entropy_weight_temp)
        loss = torch.mean(output[:,0]*output_loss1+output[:,1]*output_loss3+output[:,2]*output_loss4)
        loss_ent_min = (loss_u + loss_u_v + loss_u_f) * args.entropy_min_weight / 3
    elif args.use_video and args.use_audio:
        entropy = -torch.cat((entropyp, entropya, entropypa), 1)
        output = nn.Softmax(dim=1)(entropy/args.entropy_weight_temp)
        loss = torch.mean(output[:,0]*output_loss1+output[:,1]*output_loss2+output[:,2]*output_loss4)
        loss_ent_min = (loss_u + loss_u_v + loss_u_a) * args.entropy_min_weight / 3
    elif args.use_flow and args.use_audio:
        entropy = -torch.cat((entropyf, entropya, entropypa), 1)
        output = nn.Softmax(dim=1)(entropy/args.entropy_weight_temp)
        loss = torch.mean(output[:,0]*output_loss3+output[:,1]*output_loss2+output[:,2]*output_loss4)
        loss_ent_min = (loss_u + loss_u_a + loss_u_f) * args.entropy_min_weight / 3

    # Multimodal Jigsaw Puzzles
    # 生成多模态拼图的组合，并计算拼图分类的损失
    all_combinations = list(itertools.permutations(parts, len(parts))) #parts为拼接好的多模态特征，生成多模态拼图的所有的可能组合(M*P)!
    all_combinations = [all_combinations[ji] for ji in jigsaw_indices] #根据 jigsaw_indices 选择特定的拼图组合
    jigsaw_labels = []## 初始化拼图标签和组合的列表
    combinations = []
    for label, all_parts in enumerate(all_combinations):## 遍历所有选择的拼图组合
        concatenated = torch.cat(all_parts, dim=1)# 将每个组合的所有部分连接起来，concatenated：[batch,v+f+a]
        jigsaw_labels.append(torch.tensor([label]).repeat(concatenated.shape[0], 1))# 为每个组合生成对应的标签
        combinations.append(concatenated) # 将连接后的组合添加到组合列表中
    combinations = torch.cat(combinations, dim=0)# 将所有组合连接成一个大的张量
    jigsaw_labels = torch.cat(jigsaw_labels, dim=0).squeeze(1).type(torch.LongTensor).cuda()# 将所有标签连接成一个大的张量，并转换为 LongTensor 类型并移动到 GPU 上
    predict_jigsaw = jigsaw_cls(combinations)# 使用 jigsaw_cls 模型对组合进行预测
    loss_jigsaw = nn.CrossEntropyLoss()(predict_jigsaw, jigsaw_labels)# 计算拼图分类的交叉熵损失
    loss = loss + loss_jigsaw*args.jigsaw_ratio

    # Entropy Minimization
    loss = loss + loss_ent_min

    # Masked Cross-modal Translation 
    if args.use_video and args.use_flow and args.use_audio:
        mask_v = torch.rand_like(v_emd) < args.mask_ratio  # mask70%的位置
        v_emd_masked = v_emd.clone()  
        v_emd_masked[mask_v] = 0 

        mask_a = torch.rand_like(audio_emd) < args.mask_ratio
        audio_emd_masked = audio_emd.clone()  
        audio_emd_masked[mask_a] = 0 

        mask_f = torch.rand_like(f_emd) < args.mask_ratio
        f_emd_masked = f_emd.clone()  
        f_emd_masked[mask_f] = 0 

        a_emd_t = mlp_v2a(v_emd_masked)
        v_emd_t = mlp_a2v(audio_emd_masked)
        f_emd_t = mlp_v2f(v_emd_masked)
        v_emd_t2 = mlp_f2v(f_emd_masked)
        a_emd_t2 = mlp_f2a(f_emd_masked)
        f_emd_t2 = mlp_a2f(audio_emd_masked)
        a_emd_t = a_emd_t/torch.norm(a_emd_t, dim=1, keepdim=True)
        v_emd_t = v_emd_t/torch.norm(v_emd_t, dim=1, keepdim=True)
        f_emd_t = f_emd_t/torch.norm(f_emd_t, dim=1, keepdim=True)
        a_emd_t2 = a_emd_t2/torch.norm(a_emd_t2, dim=1, keepdim=True)
        v_emd_t2 = v_emd_t2/torch.norm(v_emd_t2, dim=1, keepdim=True)
        f_emd_t2 = f_emd_t2/torch.norm(f_emd_t2, dim=1, keepdim=True)
        v2a_loss = torch.mean(torch.norm(a_emd_t-audio_emd/torch.norm(audio_emd, dim=1, keepdim=True), dim=1))
        a2v_loss = torch.mean(torch.norm(v_emd_t-v_emd/torch.norm(v_emd, dim=1, keepdim=True), dim=1))
        v2f_loss = torch.mean(torch.norm(f_emd_t-f_emd/torch.norm(f_emd, dim=1, keepdim=True), dim=1))
        f2a_loss = torch.mean(torch.norm(a_emd_t2-audio_emd/torch.norm(audio_emd, dim=1, keepdim=True), dim=1))
        f2v_loss = torch.mean(torch.norm(v_emd_t2-v_emd/torch.norm(v_emd, dim=1, keepdim=True), dim=1))
        a2f_loss = torch.mean(torch.norm(f_emd_t2-f_emd/torch.norm(f_emd, dim=1, keepdim=True), dim=1))
        loss = loss + args.alpha_trans*(v2a_loss + a2v_loss+ v2f_loss+ f2a_loss+ f2v_loss+ a2f_loss)/6
    elif args.use_video and args.use_flow:
        mask_v = torch.rand_like(v_emd) < args.mask_ratio
        v_emd_masked = v_emd.clone()  
        v_emd_masked[mask_v] = 0 

        mask_f = torch.rand_like(f_emd) < args.mask_ratio
        f_emd_masked = f_emd.clone()  
        f_emd_masked[mask_f] = 0 

        f_emd_t = mlp_v2f(v_emd_masked)
        v_emd_t2 = mlp_f2v(f_emd_masked)
        f_emd_t = f_emd_t/torch.norm(f_emd_t, dim=1, keepdim=True)
        v_emd_t2 = v_emd_t2/torch.norm(v_emd_t2, dim=1, keepdim=True)
        v2f_loss = torch.mean(torch.norm(f_emd_t-f_emd/torch.norm(f_emd, dim=1, keepdim=True), dim=1))
        f2v_loss = torch.mean(torch.norm(v_emd_t2-v_emd/torch.norm(v_emd, dim=1, keepdim=True), dim=1))
        loss = loss + args.alpha_trans*(v2f_loss+ f2v_loss)/2
    elif args.use_video and args.use_audio:
        mask_v = torch.rand_like(v_emd) < args.mask_ratio
        v_emd_masked = v_emd.clone()  
        v_emd_masked[mask_v] = 0 

        mask_a = torch.rand_like(audio_emd) < args.mask_ratio
        audio_emd_masked = audio_emd.clone()  
        audio_emd_masked[mask_a] = 0 

        a_emd_t = mlp_v2a(v_emd_masked)
        v_emd_t = mlp_a2v(audio_emd_masked)
        a_emd_t = a_emd_t/torch.norm(a_emd_t, dim=1, keepdim=True)
        v_emd_t = v_emd_t/torch.norm(v_emd_t, dim=1, keepdim=True)
        v2a_loss = torch.mean(torch.norm(a_emd_t-audio_emd/torch.norm(audio_emd, dim=1, keepdim=True), dim=1))
        a2v_loss = torch.mean(torch.norm(v_emd_t-v_emd/torch.norm(v_emd, dim=1, keepdim=True), dim=1))
        loss = loss + args.alpha_trans*(v2a_loss + a2v_loss)/2
    elif args.use_flow and args.use_audio:
        mask_a = torch.rand_like(audio_emd) < args.mask_ratio
        audio_emd_masked = audio_emd.clone()  
        audio_emd_masked[mask_a] = 0 

        mask_f = torch.rand_like(f_emd) < args.mask_ratio
        f_emd_masked = f_emd.clone()  
        f_emd_masked[mask_f] = 0 

        a_emd_t2 = mlp_f2a(f_emd_masked)
        f_emd_t2 = mlp_a2f(audio_emd_masked)
        a_emd_t2 = a_emd_t2/torch.norm(a_emd_t2, dim=1, keepdim=True)
        f_emd_t2 = f_emd_t2/torch.norm(f_emd_t2, dim=1, keepdim=True)
        f2a_loss = torch.mean(torch.norm(a_emd_t2-audio_emd/torch.norm(audio_emd, dim=1, keepdim=True), dim=1))
        a2f_loss = torch.mean(torch.norm(f_emd_t2-f_emd/torch.norm(f_emd, dim=1, keepdim=True), dim=1))
        loss = loss + args.alpha_trans*(f2a_loss + a2f_loss)/2

    # Supervised Contrastive Learning
    if args.use_video:
        v_emd_proj = v_proj(v_emd[:, :v_dim])#取前一半作为domain-shared特征，用于计算对应的对比损失
    if args.use_audio:
        a_emd_proj = a_proj(audio_emd[:, :a_dim])
    if args.use_flow:
        f_emd_proj = f_proj(f_emd[:, :f_dim])
    if args.use_video and args.use_flow and args.use_audio:
        emd_proj = torch.stack([v_emd_proj, a_emd_proj, f_emd_proj], dim=1)
    elif args.use_video and args.use_flow:
        emd_proj = torch.stack([v_emd_proj, f_emd_proj], dim=1)
    elif args.use_video and args.use_audio:
        emd_proj = torch.stack([v_emd_proj, a_emd_proj], dim=1)
    elif args.use_flow and args.use_audio:
        emd_proj = torch.stack([f_emd_proj, a_emd_proj], dim=1)
    # 对比损失，未在Moosa论文中进行显式表述，出现在了Simmmdg中
    loss_contrast = criterion_contrast(emd_proj, labels)
    loss = loss + args.alpha_contrast*loss_contrast
  
    # Feature Splitting with Distance
    loss_e = 0
    num_loss = 0
    if args.use_video:
        loss_e = loss_e - F.mse_loss(v_emd[:, :v_dim], v_emd[:, v_dim:])
        num_loss = num_loss + 1
    if args.use_audio:
        loss_e = loss_e - F.mse_loss(audio_emd[:, :a_dim], audio_emd[:, a_dim:])
        num_loss = num_loss + 1
    if args.use_flow:
        loss_e = loss_e - F.mse_loss(f_emd[:, :f_dim], f_emd[:, f_dim:])
        num_loss = num_loss + 1
    
    loss = loss + args.explore_loss_coeff * loss_e/num_loss

    optim.zero_grad()
    loss.backward()
    optim.step()
    return predict, loss

def validate_one_step(clip, labels, flow, spectrogram):
    if args.use_video:
        clip = clip['imgs'].cuda().squeeze(1)
    labels = labels.cuda()
    if args.use_flow:
        flow = flow['imgs'].cuda().squeeze(1)
    if args.use_audio:
        spectrogram = spectrogram.unsqueeze(1).type(torch.FloatTensor).cuda()
    
    with torch.no_grad():
        if args.use_video:
            x_slow, x_fast = model.module.backbone.get_feature(clip) 
            v_feat = (x_slow.detach(), x_fast.detach())  

            v_feat = model.module.backbone.get_predict(v_feat)
            predict1, v_emd = model.module.cls_head(v_feat)
        if args.use_audio:
            _, audio_feat, _ = audio_model(spectrogram)
            audio_predict, audio_emd = audio_cls_model(audio_feat.detach())
        if args.use_flow:
            f_feat = model_flow.module.backbone.get_feature(flow)  
            f_feat = model_flow.module.backbone.get_predict(f_feat)
            f_predict, f_emd = model_flow.module.cls_head(f_feat)

        if args.use_video and args.use_flow and args.use_audio:
            feat = torch.cat((v_emd, audio_emd, f_emd), dim=1)
        elif args.use_video and args.use_flow:
            feat = torch.cat((v_emd, f_emd), dim=1)
        elif args.use_video and args.use_audio:
            feat = torch.cat((v_emd, audio_emd), dim=1)
        elif args.use_flow and args.use_audio:
            feat = torch.cat((f_emd, audio_emd), dim=1)

        predict = mlp_cls(feat)

    loss = torch.mean(criterion(predict, labels))

    return predict, loss

class Encoder(nn.Module):
    def __init__(self, input_dim=2816, out_dim=8, hidden=512):
        super(Encoder, self).__init__()
        self.enc_net = nn.Sequential(
          nn.Linear(input_dim, hidden),
          nn.ReLU(),
          nn.Dropout(p=0.5),
          nn.Linear(hidden, out_dim)
        )
        
    def forward(self, feat):
        return self.enc_net(feat)

class EncoderTrans(nn.Module):
    def __init__(self, input_dim=2816, out_dim=8, hidden=512):
        super(EncoderTrans, self).__init__()
        self.enc_net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(hidden, out_dim)
        )
        
    def forward(self, feat):
        feat = self.enc_net(feat)
        return feat

class EncoderJigsaw(nn.Module):
    def __init__(self, input_dim=2816, out_dim=8, hidden=512):
        super(EncoderJigsaw, self).__init__()
        self.enc_net = nn.Sequential(
          nn.Linear(input_dim, hidden),
          nn.ReLU(),
          nn.Linear(hidden, out_dim)
        )
        
    def forward(self, feat):
        return self.enc_net(feat)

class ProjectHead(nn.Module):
    def __init__(self, input_dim=2816, hidden_dim=2048, out_dim=128):
        super(ProjectHead, self).__init__()
        self.head = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim, out_dim)
            )
        
    def forward(self, feat):
        feat = F.normalize(self.head(feat), dim=1)
        return feat

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('-s','--source_domain', nargs='+', help='<Required> Set source_domain', required=True)
    parser.add_argument('-t','--target_domain', nargs='+', help='<Required> Set target_domain', required=True)
    parser.add_argument('--datapath', type=str, default='/path/to/EPIC-KITCHENS/', help='datapath')
    parser.add_argument('--lr', type=float, default=1e-4, help='lr')
    parser.add_argument('--bsz', type=int, default=16, help='batch_size')
    parser.add_argument("--nepochs", type=int, default=15)
    parser.add_argument('--save_checkpoint', action='store_true')
    parser.add_argument('--save_best', action='store_true')
    parser.add_argument('--alpha_trans', type=float, default=0.1,  help='alpha_trans')
    parser.add_argument("--trans_hidden_num", type=int, default=2048)
    parser.add_argument("--hidden_dim", type=int, default=2048)
    parser.add_argument("--out_dim", type=int, default=128)
    parser.add_argument('--temp', type=float, default=0.1, help='temp')
    parser.add_argument('--alpha_contrast', type=float, default=3.0, help='alpha_contrast')
    parser.add_argument('--resumef', action='store_true')
    parser.add_argument('--explore_loss_coeff', type=float, default=0.7,  help='explore_loss_coeff')
    parser.add_argument("--BestEpoch", type=int, default=0)
    parser.add_argument('--BestAcc', type=float, default=0, help='BestAcc')
    parser.add_argument('--BestTestAcc', type=float, default=0,  help='BestTestAcc')
    parser.add_argument('--BestTestHscore', type=float, default=0,  help='BestTestHscore')
    parser.add_argument("--appen", type=str, default='')
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument('--use_video', action='store_true')
    parser.add_argument('--use_audio', action='store_true')
    parser.add_argument('--use_flow', action='store_true')
    parser.add_argument('--entropy_weight_temp', type=float, default=1.0, help='entropy_weight_temp')
    parser.add_argument('--entropy_min_weight', type=float, default=0.001, help='entropy_min_weight')
    parser.add_argument('--jigsaw_ratio', type=float, default=1.0,  help='jigsaw_ratio')
    parser.add_argument("--jigsaw_num_splits", type=int, default=4)
    parser.add_argument("--jigsaw_samples", type=int, default=128)
    parser.add_argument("--jigsaw_hidden", type=int, default=512)
    parser.add_argument('--mask_ratio', type=float, default=0.3, help='mask_ratio')
    parser.add_argument('--train_round', type=str, default='_r1', help='the round of training')
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    if args.use_video and args.use_flow and args.use_audio:
        jigsaw_indices = random.sample(range(math.factorial(3*args.jigsaw_num_splits)), args.jigsaw_samples)  # 随机生成jigsaw_samples=128个索引
    else:
        jigsaw_indices = random.sample(range(math.factorial(2*args.jigsaw_num_splits)), args.jigsaw_samples)
    print('jigsaw_indices: ', jigsaw_indices)

    # 视频模态的配置文件
    config_file_video = 'configs/recognition/slowfast/slowfast_r101_8x8x1_256e_kinetics400_rgb.py'
    checkpoint_file_video = 'pretrained_models/slowfast_r101_8x8x1_256e_kinetics400_rgb_20210218-0dd54025.pth'
    # 光流模态的配置文件
    config_file_flow = 'configs/recognition/slowonly/slowonly_r50_8x8x1_256e_kinetics400_flow.py'
    checkpoint_file_flow = 'pretrained_models/slowonly_r50_8x8x1_256e_kinetics400_flow_20200704-6b384243.pth'

    # assign the desired device.
    device = 'cuda:0' # or 'cpu'
    device = torch.device(device)

    num_class = len(source_all)
    input_dim = 0

    cfg = None
    cfg_flow = None
    
    if args.use_video:#3d模型
        # 初始化模型，init_recognizer 是 MMAction2 框架的一部分，用于初始化模型
        model = init_recognizer(config_file_video, checkpoint_file_video, device=device, use_frames=True)
        model.cls_head.fc_cls = nn.Linear(2304, num_class).cuda()#修改了最后的预测头
        cfg = model.cfg
        model = torch.nn.DataParallel(model)# 如果是多GPU训练，需要使用DataParallel进行加速。实际上我们这里是单GPU的，所以不需要使用。
        # ProjectHead为3层MLP，input_dim->hidden=512->hidden=512->out_dim
        v_proj = ProjectHead(input_dim=1152, hidden_dim=args.hidden_dim, out_dim=args.out_dim).cuda()
        input_dim = input_dim + 2304

    if args.use_flow:#3d模型
        model_flow = init_recognizer(config_file_flow, checkpoint_file_flow, device=device, use_frames=True)
        model_flow.cls_head.fc_cls = nn.Linear(2048, num_class).cuda()
        cfg_flow = model_flow.cfg
        model_flow = torch.nn.DataParallel(model_flow)

        f_proj = ProjectHead(input_dim=1024, hidden_dim=args.hidden_dim, out_dim=args.out_dim).cuda()
        input_dim = input_dim + 2048

    if args.use_audio:#2d模型
        audio_args = get_arguments()# 获取音频模态的参数
        # audio_model功能：将输入的频谱图转换为高级特征，不进行具体的分类任务，输出中间层特征用于后续处理
        # 对应代码：_, audio_feat, _ = audio_model(spectrogram)  # 提取音频特征
        audio_model = AVENet(audio_args)# 默认使用res18处理音频，包含conv1，bn1，relu，maxpool，layer1，layer2，layer3，layer4，avgpool，fc
        checkpoint = torch.load("pretrained_models/vggsound_avgpool.pth.tar")
        audio_model.load_state_dict(checkpoint['model_state_dict'])
        audio_model = audio_model.cuda()
        audio_model.eval()#此模型不进行训练，只提取音频初步模型

        # audio_cls_model功能：对提取的特征进行分类，包含注意力机制，输出动作类别的预测
        # 对应代码：audio_predict, audio_emd = audio_cls_model(audio_feat.detach())
        audio_cls_model = AudioAttGenModule()# 只保留了layer4，avgpool，fc
        audio_cls_model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        audio_cls_model.fc = nn.Linear(512, num_class)#替换fc
        audio_cls_model = audio_cls_model.cuda()

        a_proj = ProjectHead(input_dim=256, hidden_dim=args.hidden_dim, out_dim=args.out_dim).cuda()
        input_dim = input_dim + 512
    # 2层MLP，input_dim->hidden=512->输出维度为num_class
    mlp_cls = Encoder(input_dim=input_dim, out_dim=num_class)
    mlp_cls = mlp_cls.cuda()
    # 2层MLP，input_dim->hidden=512->128
    jigsaw_cls = EncoderJigsaw(input_dim=input_dim, out_dim=args.jigsaw_samples, hidden=args.jigsaw_hidden)
    jigsaw_cls = jigsaw_cls.cuda()

    if args.use_video and args.use_flow and args.use_audio:
        # EncoderTrans为3层MLP，input_dim->hidden=512->hidden=512->out_dim
        mlp_v2f = EncoderTrans(input_dim=2304, hidden=args.trans_hidden_num, out_dim=2048).cuda()
        mlp_f2v = EncoderTrans(input_dim=2048, hidden=args.trans_hidden_num, out_dim=2304).cuda()
        mlp_v2a = EncoderTrans(input_dim=2304, hidden=args.trans_hidden_num, out_dim=512).cuda()
        mlp_a2v = EncoderTrans(input_dim=512, hidden=args.trans_hidden_num, out_dim=2304).cuda()
        mlp_f2a = EncoderTrans(input_dim=2048, hidden=args.trans_hidden_num, out_dim=512).cuda()
        mlp_a2f = EncoderTrans(input_dim=512, hidden=args.trans_hidden_num, out_dim=2048).cuda()
    elif args.use_video and args.use_flow:
        mlp_v2f = EncoderTrans(input_dim=2304, hidden=args.trans_hidden_num, out_dim=2048).cuda()
        mlp_f2v = EncoderTrans(input_dim=2048, hidden=args.trans_hidden_num, out_dim=2304).cuda()
    elif args.use_video and args.use_audio:
        mlp_v2a = EncoderTrans(input_dim=2304, hidden=args.trans_hidden_num, out_dim=512).cuda()
        mlp_a2v = EncoderTrans(input_dim=512, hidden=args.trans_hidden_num, out_dim=2304).cuda()
    elif args.use_flow and args.use_audio:
        mlp_f2a = EncoderTrans(input_dim=2048, hidden=args.trans_hidden_num, out_dim=512).cuda()
        mlp_a2f = EncoderTrans(input_dim=512, hidden=args.trans_hidden_num, out_dim=2048).cuda()


    base_path = "checkpoints/"
    if not os.path.exists(base_path):
        os.mkdir(base_path)
    base_path_model = "models/"
    if not os.path.exists(base_path_model):
        os.mkdir(base_path_model)

    log_name = "log%s2%s_MOOSA"%(args.source_domain, args.target_domain)
    if args.use_video:
        log_name = log_name + '_video'
    if args.use_flow:
        log_name = log_name + '_flow'
    if args.use_audio:
        log_name = log_name + '_audio'

    log_name = log_name + '_entropy_min' + '_' + str(args.entropy_min_weight)
    log_name = log_name + '_entropy_weight' + '_' + str(args.entropy_weight_temp)
    log_name = log_name + '_trans_mask_%s_'%(str(args.trans_hidden_num)) + str(args.alpha_trans)+ '_' + str(args.mask_ratio)
    log_name = log_name + '_jigsaw_' + str(args.jigsaw_num_splits) + '_' + str(args.jigsaw_samples) + '_' + str(args.jigsaw_ratio)+ '_' + str(args.jigsaw_hidden)

    log_name = log_name + args.appen + args.train_round#添加训练轮次
    log_path = base_path + log_name + '.csv'
    print(log_path)

    criterion = nn.CrossEntropyLoss(reduction='none')
    criterion = criterion.cuda()
    batch_size = args.bsz
    # 对比损失动机：实际应用效果：使相同动作的不同模态特征更接近，使不同动作的特征更远离，促进模态间的特征对齐，提高模型对多模态数据的理解能力
    # SupConLoss具体使用示例
    # 假设batch_size=16，使用所有三种模态
    # 输入特征形状：
    # emd_proj: [16, 3, 128]  # 16个样本，3个模态，每个模态128维特征
    # labels: [16]  # 16个样本的类别标签
    criterion_contrast = SupConLoss(temperature=args.temp)
    criterion_contrast = criterion_contrast.cuda()
    # 训练mlp_cls分类器（含三种模态的特征作为输入）
    params = list(mlp_cls.parameters())
    # 训练视频模态的fast_path.layer4
    if args.use_video:
        # fast_path：layer1->layer2->layer3->layer4
        # slow_path：layer1->layer2->layer3->layer4
        # cls_head：全连接层
        # v_proj：视频模态的投影头(单独训练)
        params = params + list(model.module.backbone.fast_path.layer4.parameters()) + list(
        model.module.backbone.slow_path.layer4.parameters()) + list(model.module.cls_head.parameters()) + list(v_proj.parameters())
    # 训练光流模态的layer4，model_flow为slowonly模型，只包含slowpath
    if args.use_flow:
        params = params + list(model_flow.module.backbone.layer4.parameters()) +list(model_flow.module.cls_head.parameters()) + list(f_proj.parameters())
    if args.use_audio:
    # 训练音频模态的cls_head，a_proj为音频模态的投影头(单独训练)
        params = params + list(audio_cls_model.parameters()) + list(a_proj.parameters())
    # 训练模态转换器
    if args.use_video and args.use_flow and args.use_audio:
        params = params + list(mlp_v2a.parameters())+list(mlp_a2v.parameters())
        params = params + list(mlp_v2f.parameters())+list(mlp_f2v.parameters())
        params = params + list(mlp_f2a.parameters())+list(mlp_a2f.parameters())
    elif args.use_video and args.use_flow:
        params = params + list(mlp_v2f.parameters())+list(mlp_f2v.parameters())
    elif args.use_video and args.use_audio:
        params = params + list(mlp_v2a.parameters())+list(mlp_a2v.parameters())
    elif args.use_flow and args.use_audio:
        params = params + list(mlp_f2a.parameters())+list(mlp_a2f.parameters())
    # 训练jigsaw_cls分类器（含三种模态的特征作为输入，预测其排列重组后的索引）
    params = params + list(jigsaw_cls.parameters())

    optim = torch.optim.Adam(params, lr=args.lr, weight_decay=1e-4)
    
    BestLoss = float("inf")
    BestEpoch = args.BestEpoch
    BestAcc = args.BestAcc
    BestTestAcc = args.BestTestAcc
    BestTestHscore = args.BestTestHscore
    BestTestOS = 0
    BestTestUNK = 0

    if args.resumef:
        resume_file = base_path_model + log_name + '.pt'
        print("Resuming from ", resume_file)
        checkpoint = torch.load(resume_file)
        starting_epoch = checkpoint['epoch']+1
    
        BestLoss = checkpoint['BestLoss']
        BestEpoch = checkpoint['BestEpoch']
        BestAcc = checkpoint['BestAcc']
        BestTestAcc = checkpoint['BestTestAcc']
        BestTestHscore = checkpoint['BestTestHscore']

        if args.use_video:
            model.load_state_dict(checkpoint['model_state_dict'])
            v_proj.load_state_dict(checkpoint['v_proj_state_dict'])
        if args.use_flow:
            model_flow.load_state_dict(checkpoint['model_flow_state_dict'])
            f_proj.load_state_dict(checkpoint['f_proj_state_dict'])
        if args.use_audio:
            audio_model.load_state_dict(checkpoint['audio_model_state_dict'])
            audio_cls_model.load_state_dict(checkpoint['audio_cls_model_state_dict'])
            a_proj.load_state_dict(checkpoint['a_proj_state_dict'])
        optim.load_state_dict(checkpoint['optimizer'])
        if args.use_video and args.use_flow and args.use_audio:
            mlp_v2a.load_state_dict(checkpoint['mlp_v2a_state_dict'])
            mlp_a2v.load_state_dict(checkpoint['mlp_a2v_state_dict'])
            mlp_v2f.load_state_dict(checkpoint['mlp_v2f_state_dict'])
            mlp_f2v.load_state_dict(checkpoint['mlp_f2v_state_dict'])
            mlp_f2a.load_state_dict(checkpoint['mlp_f2a_state_dict'])
            mlp_a2f.load_state_dict(checkpoint['mlp_a2f_state_dict'])
        elif args.use_video and args.use_flow:
            mlp_v2f.load_state_dict(checkpoint['mlp_v2f_state_dict'])
            mlp_f2v.load_state_dict(checkpoint['mlp_f2v_state_dict'])
        elif args.use_video and args.use_audio:
            mlp_v2a.load_state_dict(checkpoint['mlp_v2a_state_dict'])
            mlp_a2v.load_state_dict(checkpoint['mlp_a2v_state_dict'])
        elif args.use_flow and args.use_audio:
            mlp_f2a.load_state_dict(checkpoint['mlp_f2a_state_dict'])
            mlp_a2f.load_state_dict(checkpoint['mlp_a2f_state_dict'])
        mlp_cls.load_state_dict(checkpoint['mlp_cls_state_dict'])
        jigsaw_cls.load_state_dict(checkpoint['jigsaw_cls_state_dict'])
    else:
        print("Training From Scratch ..." )
        starting_epoch = 0

    print("starting_epoch: ", starting_epoch)

    train_dataset = EPICDOMAIN(split='train', domain_name='source', domain=args.source_domain, cfg=cfg, cfg_flow=cfg_flow, datapath=args.datapath, use_video=args.use_video, use_flow=args.use_flow, use_audio=args.use_audio)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, num_workers=args.num_workers, shuffle=True,
                                                   pin_memory=(device.type == "cuda"), drop_last=True)

    val_dataset = EPICDOMAIN(split='test', domain_name='source', domain=args.source_domain, cfg=cfg, cfg_flow=cfg_flow, datapath=args.datapath, use_video=args.use_video, use_flow=args.use_flow, use_audio=args.use_audio)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, num_workers=args.num_workers, shuffle=False,
                                                   pin_memory=(device.type == "cuda"), drop_last=False)

    test_dataset = EPICDOMAIN(split='test', domain_name='target', domain=args.target_domain, cfg=cfg, cfg_flow=cfg_flow, datapath=args.datapath, use_video=args.use_video, use_flow=args.use_flow, use_audio=args.use_audio)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, num_workers=args.num_workers, shuffle=False,
                                                   pin_memory=(device.type == "cuda"), drop_last=False)
    dataloaders = {'train': train_dataloader, 'val': val_dataloader, 'test': test_dataloader}
    with open(log_path, "a") as f:
        for epoch_i in range(starting_epoch, args.nepochs):
            print("Epoch: %02d" % epoch_i)
            for split in ['train', 'val', 'test']:
                acc = 0
                count = 0
                total_loss = 0
                print(split)
                # 如果split == 'train'，设置模型为训练模式。根据模态的不同，选择需要训练的模型
                mlp_cls.train(split == 'train')# nn.Module 的 train() 方法接受一个布尔值参数：当参数为 True 时，模型进入训练模式；当参数为 False 时，模型进入评估模式
                jigsaw_cls.train(split == 'train')
                if args.use_video:
                    model.train(split == 'train')
                    v_proj.train(split == 'train')
                if args.use_flow:
                    model_flow.train(split == 'train')
                    f_proj.train(split == 'train')
                if args.use_audio:
                    audio_cls_model.train(split == 'train')
                    a_proj.train(split == 'train')
                if args.use_video and args.use_flow and args.use_audio:
                    mlp_v2a.train(split == 'train')
                    mlp_a2v.train(split == 'train')
                    mlp_v2f.train(split == 'train')
                    mlp_f2v.train(split == 'train')
                    mlp_f2a.train(split == 'train')
                    mlp_a2f.train(split == 'train')
                elif args.use_video and args.use_flow:
                    mlp_v2f.train(split == 'train')
                    mlp_f2v.train(split == 'train')
                elif args.use_video and args.use_audio:
                    mlp_v2a.train(split == 'train')
                    mlp_a2v.train(split == 'train')
                elif args.use_flow and args.use_audio:
                    mlp_f2a.train(split == 'train')
                    mlp_a2f.train(split == 'train')
                with tqdm.tqdm(total=len(dataloaders[split])) as pbar:
                    # 如果是测试环节
                    if split=='test':
                        output_sum = []
                        target_sum = []

                        with torch.no_grad():
                            for (i, (clip, flow, spectrogram, labels)) in enumerate(dataloaders[split]):
                                target = labels  # 获取测试集中的标签
                                if args.use_video:
                                    clip = clip['imgs'].cuda().squeeze(1)
                                    x_slow, x_fast = model.module.backbone.get_feature(clip)  
                                    v_feat = (x_slow.detach(), x_fast.detach()) 
                                    v_feat = model.module.backbone.get_predict(v_feat)
                                    predict1, v_emd = model.module.cls_head(v_feat)
                                if args.use_flow:
                                    flow = flow['imgs'].cuda().squeeze(1)
                                    f_feat = model_flow.module.backbone.get_feature(flow)
                                    f_feat = model_flow.module.backbone.get_predict(f_feat.detach())
                                    f_predict, f_emd = model_flow.module.cls_head(f_feat)
                                if args.use_audio:
                                    spectrogram = spectrogram.unsqueeze(1).cuda()
                                    _, audio_feat, _ = audio_model(spectrogram)
                                    audio_predict, audio_emd = audio_cls_model(audio_feat.detach())

                                if args.use_video and args.use_flow and args.use_audio:
                                    feat = torch.cat((v_emd, audio_emd, f_emd), dim=1)
                                elif args.use_video and args.use_flow:
                                    feat = torch.cat((v_emd, f_emd), dim=1)
                                elif args.use_video and args.use_audio:
                                    feat = torch.cat((v_emd, audio_emd), dim=1)
                                elif args.use_flow and args.use_audio:
                                    feat = torch.cat((f_emd, audio_emd), dim=1)

                                predict = mlp_cls(feat)#
                                # 训练集中标签编号：[0,num_class - 1]
                                # 测试集中出现新标签，即target > (num_class - 1)。
                                outlier_flag = (target > (num_class - 1)).float()  # outlier设置为1
                                # 使用 outlier_flag 更新目标标签 target：
                                # 对于正常值（outlier_flag 为 0），保持原值不变。
                                # 对于异常值（outlier_flag 为 1），将其值设置为 num_clas
                                target = target * (1 - outlier_flag) + num_class * outlier_flag
                                target = target.long()
                                output_sum.append(predict)
                                target_sum.append(target)
                        output_sum = torch.cat(output_sum)
                        target_sum = torch.cat(target_sum)
                        # 对模型的输出进行 Softmax 操作，计算每个样本的最大概率值，并生成从最小最大概率值到最大最大概率值的等间隔阈值范围。这些阈值可以用于后续的异常值检测
                        tsm_output = F.softmax(output_sum, dim=1)
                        outlier_indis, max_index = torch.max(tsm_output, 1)
                        thd_min = torch.min(outlier_indis)
                        thd_max = torch.max(outlier_indis)
                        # 生成一个包含 10 个元素的列表 outlier_range，表示从 thd_min 到 thd_max 的等间隔值。每个值表示一个可能的异常值阈值
                        outlier_range = [thd_min + (thd_max - thd_min) * k / 9 for k in range(10)]

                        best_overall_acc = 0.0
                        best_thred_acc = 0.0
                        best_overall_Hscore = 0.0
                        best_thred_Hscore = 0.0
                        best_acc_insider = 0.0
                        best_acc_outsider = 0.0
                        # 通过不同的异常值阈值来评估模型的整体准确率和Hscore，并找到最佳的阈值，使得模型在正常值和异常值上的表现最优
                        for outlier_thred in outlier_range:
                            # 成一个标志张量 outlier_pred。如果 outlier_indis 中的值小于 outlier_thred，则标记为 1，否则标记为 0
                            outlier_pred = (outlier_indis < outlier_thred).float()
                            outlier_pred = outlier_pred.view(-1, 1)
                            output = torch.cat((tsm_output, outlier_pred.cuda()), dim=1)

                            _, predict = torch.max(output.detach().cpu(), dim=1)
                            overall_acc = (predict == target_sum).sum().item() / target_sum.shape[0]
                            #
                            indices_outsider = torch.where(target_sum == num_class)[0]
                            indices_insider = torch.where(target_sum != num_class)[0]
                            # 计算正常值的准确率 acc_insider 和异常值的准确率 acc_outsider
                            acc_insider = (predict[indices_insider] == target_sum[indices_insider]).sum().item() / target_sum[indices_insider].shape[0]
                            acc_outsider = (predict[indices_outsider] == target_sum[indices_outsider]).sum().item() / target_sum[indices_outsider].shape[0]
                            overall_Hscore = 2.0 * acc_insider * acc_outsider / (acc_insider + acc_outsider)

                            if overall_acc > best_overall_acc:
                                best_overall_acc = overall_acc
                                best_thred_acc = outlier_thred
                            if overall_Hscore > best_overall_Hscore:
                                best_overall_Hscore = overall_Hscore
                                best_thred_Hscore = outlier_thred
                                best_acc_insider = acc_insider
                                best_acc_outsider = acc_outsider
                    # 如果是训练或验证环节    
                    else:
                        # clip, flow, spectrogram, labels分别对应视频、光流、音频和标签数据
                        # 
                        for (i, (clip, flow, spectrogram, labels)) in enumerate(dataloaders[split]):  # 迭代一个batch的数据
                            if split=='train':
                                predict1, loss = train_one_step(clip, labels, flow, spectrogram)
                            else:
                                predict1, loss = validate_one_step(clip, labels, flow, spectrogram)

                            total_loss += loss.item() * batch_size
                            _, predict = torch.max(predict1.detach().cpu(), dim=1)

                            acc1 = (predict == labels).sum().item()
                            acc += int(acc1)
                            count += predict1.size()[0]
                            pbar.set_postfix_str(
                                "Average loss: {:.4f}, Current loss: {:.4f}, Accuracy: {:.4f}".format(total_loss / float(count), loss.item(), acc / float(count)))
                            pbar.update()

                    if split == 'val':
                        currentvalAcc = acc / float(count)
                        if currentvalAcc >= BestAcc:
                            BestLoss = total_loss / float(count)
                            BestEpoch = epoch_i
                            BestAcc = acc / float(count)
                            
                    if split == 'test':
                        if currentvalAcc >= BestAcc:
                            BestTestAcc = best_overall_acc
                            BestTestHscore = best_overall_Hscore
                            BestTestOS = best_acc_insider
                            BestTestUNK = best_acc_outsider
                            if args.save_best:
                                save = {
                                    'epoch': epoch_i,
                                    'BestLoss': BestLoss,
                                    'BestEpoch': BestEpoch,
                                    'BestAcc': BestAcc,
                                    'BestTestAcc': BestTestAcc,
                                    'BestTestHscore': BestTestHscore,
                                    'optimizer': optim.state_dict(),
                                }
                                save['mlp_cls_state_dict'] = mlp_cls.state_dict()
                                save['jigsaw_cls_state_dict'] = jigsaw_cls.state_dict()
                                
                                if args.use_video:
                                    save['v_proj_state_dict'] = v_proj.state_dict()
                                    save['model_state_dict'] = model.state_dict()
                                if args.use_flow:
                                    save['f_proj_state_dict'] = f_proj.state_dict()
                                    save['model_flow_state_dict'] = model_flow.state_dict()
                                if args.use_audio:
                                    save['a_proj_state_dict'] = a_proj.state_dict()
                                    save['audio_model_state_dict'] = audio_model.state_dict()
                                    save['audio_cls_model_state_dict'] = audio_cls_model.state_dict()
                                if args.use_video and args.use_flow and args.use_audio:
                                    save['mlp_v2a_state_dict'] = mlp_v2a.state_dict()
                                    save['mlp_a2v_state_dict'] = mlp_a2v.state_dict()
                                    save['mlp_v2f_state_dict'] = mlp_v2f.state_dict()
                                    save['mlp_f2v_state_dict'] = mlp_f2v.state_dict()
                                    save['mlp_f2a_state_dict'] = mlp_f2a.state_dict()
                                    save['mlp_a2f_state_dict'] = mlp_a2f.state_dict()
                                elif args.use_video and args.use_flow:
                                    save['mlp_v2f_state_dict'] = mlp_v2f.state_dict()
                                    save['mlp_f2v_state_dict'] = mlp_f2v.state_dict()
                                elif args.use_video and args.use_audio:
                                    save['mlp_v2a_state_dict'] = mlp_v2a.state_dict()
                                    save['mlp_a2v_state_dict'] = mlp_a2v.state_dict()
                                elif args.use_flow and args.use_audio:
                                    save['mlp_f2a_state_dict'] = mlp_f2a.state_dict()
                                    save['mlp_a2f_state_dict'] = mlp_a2f.state_dict()

                                torch.save(save, base_path_model + log_name + '_best_%s.pt'%(str(epoch_i)))

                        if args.save_checkpoint:
                            save = {
                                    'epoch': epoch_i,
                                    'BestLoss': BestLoss,
                                    'BestEpoch': BestEpoch,
                                    'BestAcc': BestAcc,
                                    'BestTestAcc': BestTestAcc,
                                    'BestTestHscore': BestTestHscore,
                                    'optimizer': optim.state_dict(),
                                }
                            save['mlp_cls_state_dict'] = mlp_cls.state_dict()
                            save['jigsaw_cls_state_dict'] = jigsaw_cls.state_dict()
                            
                            if args.use_video:
                                save['v_proj_state_dict'] = v_proj.state_dict()
                                save['model_state_dict'] = model.state_dict()
                            if args.use_flow:
                                save['f_proj_state_dict'] = f_proj.state_dict()
                                save['model_flow_state_dict'] = model_flow.state_dict()
                            if args.use_audio:
                                save['a_proj_state_dict'] = a_proj.state_dict()
                                save['audio_model_state_dict'] = audio_model.state_dict()
                                save['audio_cls_model_state_dict'] = audio_cls_model.state_dict()
                            if args.use_video and args.use_flow and args.use_audio:
                                save['mlp_v2a_state_dict'] = mlp_v2a.state_dict()
                                save['mlp_a2v_state_dict'] = mlp_a2v.state_dict()
                                save['mlp_v2f_state_dict'] = mlp_v2f.state_dict()
                                save['mlp_f2v_state_dict'] = mlp_f2v.state_dict()
                                save['mlp_f2a_state_dict'] = mlp_f2a.state_dict()
                                save['mlp_a2f_state_dict'] = mlp_a2f.state_dict()
                            elif args.use_video and args.use_flow:
                                save['mlp_v2f_state_dict'] = mlp_v2f.state_dict()
                                save['mlp_f2v_state_dict'] = mlp_f2v.state_dict()
                            elif args.use_video and args.use_audio:
                                save['mlp_v2a_state_dict'] = mlp_v2a.state_dict()
                                save['mlp_a2v_state_dict'] = mlp_a2v.state_dict()
                            elif args.use_flow and args.use_audio:
                                save['mlp_f2a_state_dict'] = mlp_f2a.state_dict()
                                save['mlp_a2f_state_dict'] = mlp_a2f.state_dict()

                            torch.save(save, base_path_model + log_name + '.pt')
                        
                    if split == 'test':
                        f.write("{},{},{},{},{},{},{},{}\n".format(epoch_i, split, best_thred_acc, best_thred_Hscore, best_overall_acc, best_acc_insider, best_acc_outsider, best_overall_Hscore))
                    else:
                        f.write("{},{},{},{}\n".format(epoch_i, split, total_loss / float(count), acc / float(count)))
                    f.flush()

                    print('acc on epoch ', epoch_i)
                    print('BestValAcc ', BestAcc)
                    print('BestTestAcc ', BestTestAcc)
                    print('BestTestHscore ', BestTestHscore)
                    print('BestTestOS ', BestTestOS)
                    print('BestTestUNK ', BestTestUNK)
                    
                    if split == 'test':
                        f.write("CurrentBestEpoch,{},BestLoss,{},BestValAcc,{},BestTestAcc,{},OS,{},UNK,{},BestTestHscore,{} \n".format(BestEpoch, BestLoss, BestAcc, BestTestAcc, BestTestOS, BestTestUNK, BestTestHscore))
                        f.flush()

        f.write("BestEpoch,{},BestLoss,{},BestValAcc,{},BestTestAcc,{},OS,{},UNK,{},BestTestHscore,{} \n".format(BestEpoch, BestLoss, BestAcc, BestTestAcc, BestTestOS, BestTestUNK, BestTestHscore))
        f.flush()

        print('BestValAcc ', BestAcc)
        print('BestTestAcc ', BestTestAcc)
        print('BestTestHscore ', BestTestHscore)

    f.close()
