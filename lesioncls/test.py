import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
import json
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, precision_score, recall_score
from sklearn.metrics import confusion_matrix

import numpy as np
import logging
import matplotlib.pyplot as plt
import PIL
from PIL import Image
import torch.nn.functional as F
import argparse
import os
import csv
import random
import re
import seaborn as sns

from Datasets.skin_data import SkinDataset

parser = argparse.ArgumentParser()
parser.add_argument('--device', type=str, default='cuda')
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--bs', type=int, default=32)
parser.add_argument('--epochs', type=int, default=200)
parser.add_argument('--fold', type=int, default=8)
parser.add_argument('--json_path', type=str, default='./new_ZCH_test.json')
parser.add_argument('--root_path', default='./ZCH-Data-test',
                help='root directory of data')
parser.add_argument('--seg_path', default='./ZCH-Data-test-seg',
                help='seg directory of data')
parser.add_argument('--which_model', type=str, default='resnet50', help=['densenet161', 'densenet121', 'mobilenetv3', 'vgg19', 'resnet101', 'resnet50', 'efficientnet_b1'])
args = parser.parse_args()



def collate_fn(batch):
    images = []
    labels = []
    pat_ids = []
    img_paths = []
    for img, _, label, pat_id, img_path in batch: # here, 修改了 getitem 的返回值
        label = torch.tensor(label, dtype=torch.long)  # Convert labels to Tensor
        images.append(img)
        labels.append(label)
        pat_ids.append(pat_id)
        img_paths.append(img_path)
    images = torch.stack(images)  
    labels = torch.stack(labels)
    return images, '', labels, pat_ids, img_paths



# 计算特异性的函数
def calculate_specificity(cm, class_label):
    tn, fp, fn, tp = cm.ravel()
    if class_label == 0:
        specificity = tn / (tn + fp) if (tn + fp) != 0 else 0
    else:
        specificity = tn / (tn + fn) if (tn + fn) != 0 else 0
    return specificity


def test(model, test_loader, args):
    # 在测试集上评估
    # 加载最佳模型权重，训练权重的位置保持不变
    best_model_path = f'ckpts_new_splites/{args.which_model}_best.pth'
    model.load_state_dict(torch.load(best_model_path))
    # model.eval()  
    model.eval().to(args.device)
    all_labels = []
    all_probs = []
    all_preds = []
    pat_pl = {}
    with torch.no_grad():
        for i, (inputs, _, labels, pat_id, img_paths) in enumerate(test_loader): # here, images, '', lesion_label, pat_id, full_path
            inputs = inputs.to(args.device)
            labels = labels.to(args.device)
            print("pat_id : ", pat_id) #  ('02290',)
            print("img_path length : ", len(img_paths))

            logits = model(inputs)
            max_prob = torch.argmax(logits, dim=1)
            Y_prob = F.softmax(logits, dim=1)
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(Y_prob.cpu().numpy()[:, 1])
            all_preds.extend(max_prob.cpu().numpy())


            for j in range(len(labels)):
                pat_id_j = pat_id[j]
                if pat_id_j not in pat_pl:
                    pat_pl[pat_id_j] = {'pred': [], 'label': []}
                pat_pl[pat_id_j]['pred'].append(max_prob[j].item())
                pat_pl[pat_id_j]['label'].append(labels[j].item())


    # Lesion-level metrics
    cm = confusion_matrix(all_labels, all_preds) # here

    auc = roc_auc_score(all_labels, all_probs)
    acc = accuracy_score(all_labels, all_preds)
    f1_class0 = f1_score(all_labels, all_preds, pos_label=0)
    f1_class1 = f1_score(all_labels, all_preds, pos_label=1)
    f1_avg = f1_score(all_labels, all_preds, average='weighted')
    recall_class0 = recall_score(all_labels, all_preds, pos_label=0)
    recall_class1 = recall_score(all_labels, all_preds, pos_label=1)
    recall_avg = recall_score(all_labels, all_preds, average='weighted')
    # 计算每个类别的特异性
    specificity_class0 = calculate_specificity(cm, 0)
    specificity_class1 = calculate_specificity(cm, 1)
    # 计算平均特异性
    specificity_avg = (specificity_class0 + specificity_class1) / 2 if (specificity_class0 is not None and specificity_class1 is not None) else None
    print('Lesion-level')
    print(f'AUC: {auc}, Accuracy: {acc}, F1_class0: {f1_class0}, F1_class1: {f1_class1}, F1_avg: {f1_avg}, Recall_class0: {recall_class0}, Recall_class1: {recall_class1}, Recall_avg: {recall_avg}, Specificity_class0: {specificity_class0}, Specificity_class1: {specificity_class1}, Specificity_avg: {specificity_avg}')

    # here, 绘制并保存病灶级混淆矩阵
    save_cm_path = f'results_new_splits_open_ZCH_test/{args.which_model}'
    os.makedirs(save_cm_path, exist_ok=True)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=range(2), yticklabels=range(2), annot_kws={"size": 16})
    plt.xlabel('Predicted', fontsize=16)
    plt.ylabel('Label', fontsize=16)
    plt.title('Lesion-level Confusion Matrix', fontsize=16)
    # 设置坐标轴刻度的字体大小
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.tight_layout()  # 紧凑布局
    plt.savefig(os.path.join(save_cm_path, f'{args.which_model}_lesion_cm.png'))
    plt.close()



    # Patient-level metrics
    pat_labels = []
    pat_preds = []
    for pat_id in pat_pl:
        pat_labels.append(1 if any(pat_pl[pat_id]['label']) else 0)
        pat_preds.append(1 if any(pat_pl[pat_id]['pred']) else 0)
    
    # 计算混淆矩阵
    pat_cm = confusion_matrix(pat_labels, pat_preds)

    pat_auc = roc_auc_score(pat_labels, pat_preds)
    pat_acc = accuracy_score(pat_labels, pat_preds)
    pat_f1_class0 = f1_score(pat_labels, pat_preds, pos_label=0)
    pat_f1_class1 = f1_score(pat_labels, pat_preds, pos_label=1)
    pat_f1_avg = f1_score(pat_labels, pat_preds, average='weighted')
    pat_recall_class0 = recall_score(pat_labels, pat_preds, pos_label=0)
    pat_recall_class1 = recall_score(pat_labels, pat_preds, pos_label=1)
    pat_recall_avg = recall_score(pat_labels, pat_preds, average='weighted')
    # 计算每个类别的特异性
    pat_specificity_class0 = calculate_specificity(pat_cm, 0)
    pat_specificity_class1 = calculate_specificity(pat_cm, 1)
    # 计算平均特异性
    pat_specificity_avg = (pat_specificity_class0 + pat_specificity_class1) / 2 if (pat_specificity_class0 is not None and pat_specificity_class1 is not None) else None
    print('Patient-level')
    print(f'AUC: {pat_auc}, Accuracy: {pat_acc}, F1_class0: {pat_f1_class0}, F1_class1: {pat_f1_class1}, F1_avg: {pat_f1_avg}, Recall_class0: {pat_recall_class0}, Recall_class1: {pat_recall_class1}, Recall_avg: {pat_recall_avg}, Specificity_class0: {pat_specificity_class0}, Specificity_class1: {pat_specificity_class1}, Specificity_avg: {pat_specificity_avg}')

    
    # 绘制并保存病人级混淆矩阵
    plt.figure(figsize=(8, 6))
    sns.heatmap(pat_cm, annot=True, fmt='d', cmap='Blues', xticklabels=range(2), yticklabels=range(2), annot_kws={"size": 16})
    plt.xlabel('Predicted', fontsize=16)
    plt.ylabel('Label', fontsize=16)
    plt.title('Patient-level Confusion Matrix', fontsize=16)
    # 设置坐标轴刻度的字体大小
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.tight_layout()  
    plt.savefig(os.path.join(save_cm_path, f'{args.which_model}_patient_cm.png'))
    plt.close()
    
    
    
    savepath = f'results_new_splits_open_ZCH_test/{args.which_model}/{args.which_model}_test_metrics.csv'
    file_exist = os.path.exists(savepath)
    print("savepath : ", savepath)
    row = [
        args.fold,
        round(acc, 4), round(f1_class0, 4), round(f1_class1, 4), round(f1_avg, 4), round(recall_class0, 4), round(recall_class1, 4), round(recall_avg, 4),
        round(specificity_class0, 4), round(specificity_class1, 4), round(specificity_avg, 4), round(auc, 4),
        round(pat_acc, 4), round(pat_f1_class0, 4), round(pat_f1_class1, 4), round(pat_f1_avg, 4), round(pat_recall_class0, 4), round(pat_recall_class1, 4), round(pat_recall_avg, 4),
        round(pat_specificity_class0, 4), round(pat_specificity_class1, 4), round(pat_specificity_avg, 4)
    ]
    with open(savepath, 'a', newline='') as f:
        csv_writer = csv.writer(f)
        if not file_exist:
            csv_writer.writerow(['fold', 'acc', 'lesion_f1_0', 'lesion_f1_1', 'lesion_f1_avg', 'lesion_recall_0', 'lesion_recall_1', 'lesion_recall_avg',
                               'lesion_specificity_0', 'lesion_specificity_1', 'lesion_specificity_avg', 'auc', 'pat_acc',
                               'pat_f1_0', 'pat_f1_1', 'pat_f1_avg', 'pat_recall_0', 'pat_recall_1', 'pat_recall_avg',
                               'pat_specificity_0', 'pat_specificity_1', 'pat_specificity_avg'])
        csv_writer.writerow(row)
    

    
def main(args):
    
    # 数据预处理
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


    # 加载测试集,here、新的测试集ZCH
    test_file_path = args.json_path #f'/mnt/LiverGroup/liuyy/nevi_classification/new_ZCH_test.json' # open source version
    test_dataset = SkinDataset(test_file_path, root_path=args.root_path, seg_path=args.seg_path, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=args.bs, shuffle=False, collate_fn=collate_fn)

    # 选择模型
    # help=["densenet161", "densenet121", "mobilenetv3", "vgg19", "resnet101", "resnet50", "efficientnet_b0", "efficientnet_b1", "vit_b_16"]
    if args.which_model == 'densenet161':
        model = models.densenet161(pretrained=True).to(args.device)
        num_ftrs = model.classifier.in_features
        model.classifier = nn.Linear(num_ftrs, 2)  

    elif args.which_model == 'densenet121':
        model = models.densenet121(pretrained=True).to(args.device)
        num_ftrs = model.classifier.in_features
        model.classifier = nn.Linear(num_ftrs, 2)  

    elif args.which_model == 'resnet101':
        model = models.resnet101(pretrained=True).to(args.device)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, 2)

    elif args.which_model == 'resnet50':
        model = models.resnet50(pretrained=True).to(args.device)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, 2).to(args.device)

    elif args.which_model == 'efficientnet_b0': 
        model = models.efficientnet_b0(pretrained=True).to(args.device)
        num_ftrs = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(num_ftrs, 2)  

    elif args.which_model == 'efficientnet_b1': 
        model = models.efficientnet_b1(pretrained=True).to(args.device)
        num_ftrs = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(num_ftrs, 2)  

    elif args.which_model == 'vit_b_16':
        model = models.vit_b_16(pretrained=True).to(args.device)
        num_ftrs = model.heads.head.in_features
        model.heads.head = nn.Linear(num_ftrs, 2)  
    else:
        raise ValueError("Unsupported model type")


    # 测试模型
    test(model, test_loader, args)

if __name__ == "__main__":
    args = parser.parse_args()
    main(args)