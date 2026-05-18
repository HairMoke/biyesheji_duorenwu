# Author: Tammie li
# Description: Define the evaluation metrics
# FilePath: \DRL\Utils\evaluate.py
import numpy as np
from sklearn.metrics import roc_curve, confusion_matrix, precision_recall_curve, auc, roc_auc_score
import matplotlib.pyplot as plt
import copy


class EvaluateMetric:
    def __init__(self) -> None:
        self.KAPPA = None
        self.AUC = None
        self.BA = None
        self.TPR = None
        self.FPR = None
        self.F1 = None
        self.ACC = None
        self.P = None


class EvaluateManage:
    def __init__(self, subject_id, dataset_name, model_name):
        self.subject_id = subject_id
        self.dataset_name = dataset_name
        self.model_name = model_name
        self.pred = np.load(f'PredictionResult/{self.dataset_name}_{self.model_name}_S{self.subject_id:>02d}_preds.npy')
        self.y = np.load(f'PredictionResult/{self.dataset_name}_{self.model_name}_S{self.subject_id:>02d}_y.npy')
        self.y_pred = np.load(f'PredictionResult/{self.dataset_name}_{self.model_name}_S{self.subject_id:>02d}_y_pred.npy')

        self.score = EvaluateMetric()
    
    def calculate_metric_score(self):
        # 计算

        sum_num_non_tar = 0 # 所有预测为0的样本数量
        cor_num_non_tar = 0 # 所有预测为0， 真实也为0的样本数量
        sum_num_tar = 0  # 所有预测为1的样本数量
        cor_num_tar = 0  # 预测为1， 真实也为1的样本数量

        for idx, label in enumerate(self.y):
            if label == 0:
                if self.pred[idx] == label:
                    cor_num_non_tar += 1
                sum_num_non_tar += 1
            elif label == 1:
                if self.pred[idx] == label:
                    cor_num_tar += 1
                sum_num_tar += 1

        print(cor_num_tar, cor_num_non_tar, sum_num_tar, sum_num_non_tar)


        TP = cor_num_tar
        TN = cor_num_non_tar
        FP = sum_num_non_tar - cor_num_non_tar
        FN = sum_num_tar - cor_num_tar
        '''
        # 初始化统计量,更新一下混淆矩阵的算法
        sum_num_non_tar = 0  # 真实为0的样本总数
        cor_num_non_tar = 0  # 真实为0且预测为0的样本数（TN）
        sum_num_tar = 0  # 真实为1的样本总数
        cor_num_tar = 0  # 真实为1且预测为1的样本数（TP）
        FP = 0  # 预测为1但真实为0的样本数
        FN = 0  # 预测为0但真实为1的样本数

        for idx, label in enumerate(self.y):
            pred_label = self.pred[idx]
            if label == 0:
                sum_num_non_tar += 1
                if pred_label == 0:
                    cor_num_non_tar += 1
                elif pred_label == 1:
                    FP += 1  # 预测为1但真实为0
            elif label == 1:
                sum_num_tar += 1
                if pred_label == 1:
                    cor_num_tar += 1
                elif pred_label == 0:
                    FN += 1  # 预测为0但真实为1

        # 验证计算结果
        TP = cor_num_tar
        TN = cor_num_non_tar
        # FP 和 FN 已在循环中直接统计
        '''
        print(f"TP = {TP}, TN = {TN}, FP = {FP}, FN = {FN} ")

        ACC = round((TP + TN) / (TP + TN + FP + FN), 4)
        BA = round((TP / (TP + FN) + TN / (TN + FP)) / 2, 4)
        TPR = round(TP / (TP + FN), 4)
        FPR = round(FP / (TN + FP), 4)
        P = round(TP / (TP + FP), 4)
        # 这里计算的是正样本的F1,论文中一般计算宏平均，会对正负两类分别计算F1，再取平均值；
        # F1 = round(2 * P * TPR / (P + TPR), 4)
        # 计算正样本的精确率(Precision)和召回率(Recall)，并由此得到F1
        precision_pos = TP / (TP + FP) if TP + FP != 0 else 0
        recall_pos = TP / (TP + FN) if TP + FN != 0 else 0
        f1_pos = 2 * precision_pos * recall_pos / (precision_pos + recall_pos) if precision_pos + recall_pos != 0 else 0

        # 计算负样本的精确率(Precision)和召回率(Recall)，并由此得到F1
        precision_neg = TN / (TN + FN) if TN + FN != 0 else 0
        recall_neg = TN / (TN + FP) if TN + FP != 0 else 0
        f1_neg = 2 * precision_neg * recall_neg / (precision_neg + recall_neg) if precision_neg + recall_neg != 0 else 0

        # 宏平均F1分数
        F1_macro = (f1_pos + f1_neg) / 2
        F1 = round(F1_macro, 4)


        P_o, P_e = (TP + TN) / (TP + TN + FP + FN), (cor_num_tar**2 + cor_num_non_tar**2) / (sum_num_tar + sum_num_non_tar)**2
        KAPPA = (P_o - P_e) / (1 - P_e)

        precision, recall, _thresholds = precision_recall_curve(self.y, self.y_pred[:, 1])
        AUC = auc(recall, precision)

        AUC = roc_auc_score(self.y, self.y_pred[:, 1])
        
        self.score.ACC, self.score.BA, self.score.TPR, self.score.FPR, self.score.P, self.score.F1, \
        self.score.KAPPA, self.score.AUC = ACC, BA , TPR, FPR, P , F1, KAPPA, AUC
        print(f"ACC = {ACC}, BA = {BA}, TPR = {TPR}, P = {P}, F1 = {F1}, KAPPA = {KAPPA}, AUC = {AUC} ")
        return self.score



    














