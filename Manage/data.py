'''
Author: Tammie li
Description: 完成数据装载工作 numpy格式for传统算法 dataloaderfor深度学习模型
FilePath: \data.py
'''
import os
import gc
import copy
import random
import numpy as np
from Utils.preprocess import DataProcess
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from torchvision import transforms
from module_test.medgnn.data_provider.uea import collate_fn


class MEDGNNDataGenerate(Dataset):
    def __init__(self, Dataset, x, y, sub_id, mode):  # x(256, 64, 256)  # y (256,)
        # 数据集类型，输入数据，输入标签
        self.dataset = Dataset  # 'THU'
        self.x = x
        self.y = y
        field = "train" if mode == True else "test"
        try:
            self.x_total = np.load(
                os.path.join(os.getcwd(), 'Dataset', self.dataset, f'S{sub_id:>02d}', f'x_medgnn_{field}.npy'))
            self.y_total = np.load(
                os.path.join(os.getcwd(), 'Dataset', self.dataset, f'S{sub_id:>02d}', f'y_medgnn_{field}.npy'))
        except:
            # self.x_mtr, self.y_mtr = self._generate_by_mtr(x, y)  # (2304, 64, 256), (2304, )
            # self.x_msr, self.y_msr = self._generate_by_msr(x, y)  # (2048, 64, 256)  (2048,)
            # self.x_total = np.concatenate((self.x_mtr, self.x_msr, self.x), axis=0)
            # self.y_total = np.concatenate((self.y_mtr, self.y_msr, self.y), axis=0)
            self.x_total = np.transpose(self.x, (0, 2, 1))
            self.y_total = self.y
            np.save(os.path.join(os.getcwd(), 'Dataset', self.dataset, f'S{sub_id:>02d}', f'x_medgnn_{field}.npy'),
                    self.x_total)
            np.save(os.path.join(os.getcwd(), 'Dataset', self.dataset, f'S{sub_id:>02d}', f'y_medgnn_{field}.npy'),
                    self.y_total)
            print(self.x_total.shape, self.y_total.shape)  # (4608, 64, 256) (4608,)

    def __getitem__(self, index):
        x = self.x_total[index, ...]
        y = self.y_total[index, ...]
        return x, y

    def __len__(self):
        return len(self.x_total)

    def _draw(self, data):
        # data->(C, T)
        for i in range(data.shape[0]):
            plt.plot(data[i] + i * 0.2)
        plt.show()

    def _generate_by_mtr(self, data, label):  # (256, 64, 256)
        # 生成 masked temporal recongnition task 数据集
        masked_temporal_martrix = np.array([[0, 28], [29, 34], [35, 40], [41, 51], [52, 64], [65, 85],
                                            [86, 115], [116, 137], [138, 255]])  # (9,2)
        [N, C, T] = data.shape  # (256, 64, 256)

        # step1：扩充
        expand_data = []
        for i in range(N):  # 对于每一个样本
            for j in range(masked_temporal_martrix.shape[0]):  # 对于每一个时间段
                expand_data.append(data[i])  # 每一个样本要连着放9份
        expand_data = np.array(expand_data)  # (2304,) 里面每一个都是(64,256)
        # step2: 修改
        for i in range(N * masked_temporal_martrix.shape[0]):  # 对于现在的2304个样本
            sequence = i % masked_temporal_martrix.shape[0]  # sequence是第几个时间段
            for j in range(C):  # 对于64个通道
                raw_signal = expand_data[i, j,
                             masked_temporal_martrix[sequence][0]: masked_temporal_martrix[sequence][1]]
                mean, var = np.mean(raw_signal), np.var(raw_signal)
                for m in range(masked_temporal_martrix[sequence][0], masked_temporal_martrix[sequence][1]):
                    expand_data[i, j, m] = random.gauss(mean, var)
        x_mtr = np.array(expand_data)  # (2304, 64, 256)

        # y_mtr = [[i for i in range(masked_temporal_martrix.shape[0])] for i in range(N)]   # (256, ) 每一个都是[0,1,2,3,4,5,6,7,8]
        # 我要把每个样本的标签维持原样，起到扩充数据的作用
        # y_mtr = [[label[i] for i in range(masked_temporal_martrix.shape[0])] for i in range(N)]  # (256, ) 每一个都是这扩充的9个样本的便签
        # y_mtr = np.array(y_mtr).reshape(N * masked_temporal_martrix.shape[0])  # (2304, ) 把每一行展开了
        y_mtr = np.repeat(label, masked_temporal_martrix.shape[0])  # 这将会把 label 中的每一个元素复制 9 次

        return x_mtr, y_mtr  # (2304, 64, 256), (2304, )

    def _generate_by_msr(self, data, label):  # (256, 64, 256)
        # 生成 masked spatial recongnition task 数据集
        BiosemiRegion = [[0, 1, 2, 3, 4, 5, 6, 62],
                         [7, 8, 9, 10, 11, 16, 17, 18],
                         [13, 15, 20, 22, 24, 29, 31, 33],
                         [36, 59, 38, 42, 43, 45, 47, 60],
                         [37, 39, 41, 44, 46, 48, 61, 63],
                         [50, 51, 52, 53, 54, 55, 56, 57],
                         [12, 14, 19, 21, 23, 28, 30, 32],
                         [25, 26, 27, 34, 35, 42, 49, 58]]  # (8, 8)

        NeuralScanRegion = [[0, 1, 2, 3, 4, 59, 62, 63],
                            [7, 8, 9, 10, 11, 17, 18, 19],
                            [12, 13, 20, 21, 22, 29, 30, 31],
                            [32, 33, 34, 42, 43, 44, 41, 58],
                            [26, 27, 28, 35, 36, 37, 45, 53],
                            [38, 39, 40, 46, 47, 48, 49, 61],
                            [50, 51, 52, 54, 56, 57, 60, 55],
                            [5, 6, 14, 15, 16, 23, 24, 25]]  # (8, 8)
        region = NeuralScanRegion if self.dataset == "CAS" else BiosemiRegion
        region = np.array(region)  # (8,8)

        [N, C, T] = data.shape  # (256, 64, 256)

        expand_data = []
        # step1: 扩充
        for i in range(N):  # 对于256个样本
            for j in range(region.shape[0]):  # 对于8个脑区
                expand_data.append(data[i])  # 每一个样本要连着放8份
        expand_data = np.array(expand_data)  # (2048, 64, 256)

        # step2: 修改
        for i in range(N * region.shape[0]):  # 对于2048个样本
            sequence = i % region.shape[0]  # sequence是第几个脑区
            for j in region[sequence]:  # 每个脑区里有8个通道
                raw_signal = expand_data[i, j, :]
                mean, var = np.mean(raw_signal), np.var(raw_signal)
                for m in range(T):
                    expand_data[i, j, m] = random.gauss(mean, var)
            # self._draw(expand_data[i])
        x_msr = np.array(expand_data)  # (2048, 64, 256)

        # y_msr = [[i for i in range(region.shape[0])] for i in range(N)] #[256,8]
        # y_msr = [[label[i] for i in range(region.shape[0])] for i in range(N)]  # [256,8]
        # y_msr = np.array(y_msr).reshape(N * region.shape[0])  # (2048，)
        y_msr = np.repeat(label, region.shape[0])  # 这将会把 label 中的每一个元素复制 9 次
        return x_msr, y_msr  # (2048, 64, 256)  (2048,)


class VisionEagleDataGenerate(Dataset):
    def __init__(self, Dataset, x, y, sub_id, mode):  # x(256, 64, 256)  # y (256,)
        # 数据集类型，输入数据，输入标签
        self.dataset = Dataset  # 'THU'
        self.x = x
        self.y = y
        field = "train" if mode == True else "test"
        try:
            self.x_total = np.load(
                os.path.join(os.getcwd(), 'Dataset', self.dataset, f'S{sub_id:>02d}', f'x_total_{field}.npy'))
            self.y_total = np.load(
                os.path.join(os.getcwd(), 'Dataset', self.dataset, f'S{sub_id:>02d}', f'y_total_{field}.npy'))
        except:
            self.x_mtr, self.y_mtr = self._generate_by_mtr(x, y)  # (2304, 64, 256), (2304, )
            self.x_msr, self.y_msr = self._generate_by_msr(x, y)  # (2048, 64, 256)  (2048,)
            self.x_total = np.concatenate((self.x_mtr, self.x_msr, self.x), axis=0)
            self.y_total = np.concatenate((self.y_mtr, self.y_msr, self.y), axis=0)
            np.save(os.path.join(os.getcwd(), 'Dataset', self.dataset, f'S{sub_id:>02d}', f'x_total_{field}.npy'),
                    self.x_mtr)
            np.save(os.path.join(os.getcwd(), 'Dataset', self.dataset, f'S{sub_id:>02d}', f'y_total_{field}.npy'),
                    self.y_mtr)
            print(self.x_total.shape, self.y_total.shape)  # (4608, 64, 256) (4608,)

    def __getitem__(self, index):
        x = self.x_total[index, ...]
        y = self.y_total[index, ...]
        return x, y

    def __len__(self):
        return len(self.x_total)

    def _draw(self, data):
        # data->(C, T)
        for i in range(data.shape[0]):
            plt.plot(data[i] + i * 0.2)
        plt.show()

    def _generate_by_mtr(self, data, label):  # (256, 64, 256)
        # 生成 masked temporal recongnition task 数据集
        masked_temporal_martrix = np.array([[0, 28], [29, 34], [35, 40], [41, 51], [52, 64], [65, 85],
                                            [86, 115], [116, 137], [138, 255]])  # (9,2)
        [N, C, T] = data.shape  # (256, 64, 256)

        # step1：扩充
        expand_data = []
        for i in range(N):  # 对于每一个样本
            for j in range(masked_temporal_martrix.shape[0]):  # 对于每一个时间段
                expand_data.append(data[i])  # 每一个样本要连着放9份
        expand_data = np.array(expand_data)  # (2304,) 里面每一个都是(64,256)
        # step2: 修改
        for i in range(N * masked_temporal_martrix.shape[0]):  # 对于现在的2304个样本
            sequence = i % masked_temporal_martrix.shape[0]  # sequence是第几个时间段
            for j in range(C):  # 对于64个通道
                raw_signal = expand_data[i, j,
                             masked_temporal_martrix[sequence][0]: masked_temporal_martrix[sequence][1]]
                mean, var = np.mean(raw_signal), np.var(raw_signal)
                for m in range(masked_temporal_martrix[sequence][0], masked_temporal_martrix[sequence][1]):
                    expand_data[i, j, m] = random.gauss(mean, var)
        x_mtr = np.array(expand_data)  # (2304, 64, 256)

        # y_mtr = [[i for i in range(masked_temporal_martrix.shape[0])] for i in range(N)]   # (256, ) 每一个都是[0,1,2,3,4,5,6,7,8]
        # 我要把每个样本的标签维持原样，起到扩充数据的作用
        # y_mtr = [[label[i] for i in range(masked_temporal_martrix.shape[0])] for i in range(N)]  # (256, ) 每一个都是这扩充的9个样本的便签
        # y_mtr = np.array(y_mtr).reshape(N * masked_temporal_martrix.shape[0])  # (2304, ) 把每一行展开了
        y_mtr = np.repeat(label, masked_temporal_martrix.shape[0])  # 这将会把 label 中的每一个元素复制 9 次

        return x_mtr, y_mtr  # (2304, 64, 256), (2304, )

    def _generate_by_msr(self, data, label):  # (256, 64, 256)
        # 生成 masked spatial recongnition task 数据集
        BiosemiRegion = [[0, 1, 2, 3, 4, 5, 6, 62],
                         [7, 8, 9, 10, 11, 16, 17, 18],
                         [13, 15, 20, 22, 24, 29, 31, 33],
                         [36, 59, 38, 42, 43, 45, 47, 60],
                         [37, 39, 41, 44, 46, 48, 61, 63],
                         [50, 51, 52, 53, 54, 55, 56, 57],
                         [12, 14, 19, 21, 23, 28, 30, 32],
                         [25, 26, 27, 34, 35, 42, 49, 58]]  # (8, 8)

        NeuralScanRegion = [[0, 1, 2, 3, 4, 59, 62, 63],
                            [7, 8, 9, 10, 11, 17, 18, 19],
                            [12, 13, 20, 21, 22, 29, 30, 31],
                            [32, 33, 34, 42, 43, 44, 41, 58],
                            [26, 27, 28, 35, 36, 37, 45, 53],
                            [38, 39, 40, 46, 47, 48, 49, 61],
                            [50, 51, 52, 54, 56, 57, 60, 55],
                            [5, 6, 14, 15, 16, 23, 24, 25]]  # (8, 8)
        region = NeuralScanRegion if self.dataset == "CAS" else BiosemiRegion
        region = np.array(region)  # (8,8)

        [N, C, T] = data.shape  # (256, 64, 256)

        expand_data = []
        # step1: 扩充
        for i in range(N):  # 对于256个样本
            for j in range(region.shape[0]):  # 对于8个脑区
                expand_data.append(data[i])  # 每一个样本要连着放8份
        expand_data = np.array(expand_data)  # (2048, 64, 256)

        # step2: 修改
        for i in range(N * region.shape[0]):  # 对于2048个样本
            sequence = i % region.shape[0]  # sequence是第几个脑区
            for j in region[sequence]:  # 每个脑区里有8个通道
                raw_signal = expand_data[i, j, :]
                mean, var = np.mean(raw_signal), np.var(raw_signal)
                for m in range(T):
                    expand_data[i, j, m] = random.gauss(mean, var)
            # self._draw(expand_data[i])
        x_msr = np.array(expand_data)  # (2048, 64, 256)

        # y_msr = [[i for i in range(region.shape[0])] for i in range(N)] #[256,8]
        # y_msr = [[label[i] for i in range(region.shape[0])] for i in range(N)]  # [256,8]
        # y_msr = np.array(y_msr).reshape(N * region.shape[0])  # (2048，)
        y_msr = np.repeat(label, region.shape[0])  # 这将会把 label 中的每一个元素复制 9 次
        return x_msr, y_msr  # (2048, 64, 256)  (2048,)

class MTCNPicDataGenerate(Dataset):
    def __init__(self, Dataset, x, y, sub_id, mode):  # x(256, 64, 256)  # y (256,)
        # 数据集类型，输入数据，输入标签
        self.dataset = Dataset  # 'THU'
        self.x = x
        self.y = y
        field = "train" if mode == True else "test"
        try:
            self.x_mtr = np.load(
                os.path.join(os.getcwd(), 'Dataset', self.dataset, f'S{sub_id:>02d}', f'x_mtr_{field}.npy'))
            self.y_mtr = np.load(
                os.path.join(os.getcwd(), 'Dataset', self.dataset, f'S{sub_id:>02d}', f'y_mtr_{field}.npy'))
            self.x_msr = np.load(
                os.path.join(os.getcwd(), 'Dataset', self.dataset, f'S{sub_id:>02d}', f'x_msr_{field}.npy'))
            self.y_msr = np.load(
                os.path.join(os.getcwd(), 'Dataset', self.dataset, f'S{sub_id:>02d}', f'y_msr_{field}.npy'))
            self.x_pic = np.load(
                os.path.join(os.getcwd(), 'Dataset', self.dataset, f'S{sub_id:>02d}', f'x_pic_{field}.npy'))
            self.y_pic = y
            self.x_ftr = np.load(
                os.path.join(os.getcwd(), 'Dataset', self.dataset, f'S{sub_id:>02d}', f'x_ftr_{field}.npy'))
            self.y_ftr = np.load(
                os.path.join(os.getcwd(), 'Dataset', self.dataset, f'S{sub_id:>02d}', f'y_ftr_{field}.npy'))
        except:
            self.x_mtr, self.y_mtr = self._generate_by_mtr(x)
            self.x_msr, self.y_msr = self._generate_by_msr(x)
            self.x_pic = self._generate_by_pic(x)
            self.y_pic = y
            self.x_ftr, self.y_ftr = self._generate_by_ftr(x)
            np.save(os.path.join(os.getcwd(), 'Dataset', self.dataset, f'S{sub_id:>02d}', f'x_mtr_{field}.npy'),
                    self.x_mtr)
            np.save(os.path.join(os.getcwd(), 'Dataset', self.dataset, f'S{sub_id:>02d}', f'y_mtr_{field}.npy'),
                    self.y_mtr)
            np.save(os.path.join(os.getcwd(), 'Dataset', self.dataset, f'S{sub_id:>02d}', f'x_msr_{field}.npy'),
                    self.x_msr)
            np.save(os.path.join(os.getcwd(), 'Dataset', self.dataset, f'S{sub_id:>02d}', f'y_msr_{field}.npy'),
                    self.y_msr)
            np.save(os.path.join(os.getcwd(), 'Dataset', self.dataset, f'S{sub_id:>02d}', f'x_pic_{field}.npy'),
                    self.x_pic)
            np.save(os.path.join(os.getcwd(), 'Dataset', self.dataset, f'S{sub_id:>02d}', f'y_pic_{field}.npy'),
                    self.y_pic)
            np.save(os.path.join(os.getcwd(), 'Dataset', self.dataset, f'S{sub_id:>02d}', f'x_ftr_{field}.npy'),
                    self.x_ftr)
            np.save(os.path.join(os.getcwd(), 'Dataset', self.dataset, f'S{sub_id:>02d}', f'y_ftr_{field}.npy'),
                    self.y_ftr)
            print(self.x_mtr.shape, self.y_mtr.shape, self.x_msr.shape, self.y_msr.shape, self.x_pic.shape,
                  self.y_pic.shape, self.x_ftr.shape, self.y_ftr.shape)

    def __getitem__(self, index):
        x = self.x[index, ...]
        y = self.y[index, ...]
        x_mtr = self.x_mtr[9 * index: 9 * index + 9, ...]
        y_mtr = self.y_mtr[9 * index: 9 * index + 9, ...]
        x_msr = self.x_msr[8 * index: 8 * index + 8, ...]
        y_msr = self.y_msr[8 * index: 8 * index + 8, ...]
        x_pic = self.x_pic[index, ...]
        y_pic = self.y_pic[index, ...]
        x_ftr = self.x_ftr[10 * index:10 * index + 10, ...]
        y_ftr = self.y_ftr[10 * index:10 * index + 10, ...]
        return x, y, x_mtr, y_mtr, x_msr, y_msr, x_pic, y_pic, x_ftr, y_ftr

    def __len__(self):
        return len(self.x)

    def _draw(self, data):
        # data->(C, T)
        for i in range(data.shape[0]):
            plt.plot(data[i] + i * 0.2)
        plt.show()

    def _generate_by_mtr(self, data):  # (256, 64, 256)
        # 生成 masked temporal recongnition task 数据集
        masked_temporal_martrix = np.array([[0, 28], [29, 34], [35, 40], [41, 51], [52, 64], [65, 85],
                                            [86, 115], [116, 137], [138, 255]])  # (9,2)
        [N, C, T] = data.shape  # (256, 64, 256)

        # step1：扩充
        expand_data = []
        for i in range(N):  # 对于每一个样本
            for j in range(masked_temporal_martrix.shape[0]):  # 对于每一个时间段
                expand_data.append(data[i])  # 每一个样本要连着放9份
        expand_data = np.array(expand_data)  # (2304,) 里面每一个都是(64,256)
        # step2: 修改
        for i in range(N * masked_temporal_martrix.shape[0]):  # 对于现在的2304个样本
            sequence = i % masked_temporal_martrix.shape[0]  # sequence是第几个时间段
            for j in range(C):  # 对于64个通道
                raw_signal = expand_data[i, j,
                             masked_temporal_martrix[sequence][0]: masked_temporal_martrix[sequence][1]]
                mean, var = np.mean(raw_signal), np.var(raw_signal)
                for m in range(masked_temporal_martrix[sequence][0], masked_temporal_martrix[sequence][1]):
                    expand_data[i, j, m] = random.gauss(mean, var)
        x_mtr = np.array(expand_data)  # (2304, 64, 256)

        y_mtr = [[i for i in range(masked_temporal_martrix.shape[0])] for i in
                 range(N)]  # (256, ) 每一个都是[0,1,2,3,4,5,6,7,8]
        y_mtr = np.array(y_mtr).reshape(N * masked_temporal_martrix.shape[0])  # (2304, ) 把每一行展开了

        return x_mtr, y_mtr  # (2304, 64, 256), (2304, )

    def _generate_by_msr(self, data):  # (256, 64, 256)
        # 生成 masked spatial recongnition task 数据集
        BiosemiRegion = [[0, 1, 2, 3, 4, 5, 6, 62],
                         [7, 8, 9, 10, 11, 16, 17, 18],
                         [13, 15, 20, 22, 24, 29, 31, 33],
                         [36, 59, 38, 42, 43, 45, 47, 60],
                         [37, 39, 41, 44, 46, 48, 61, 63],
                         [50, 51, 52, 53, 54, 55, 56, 57],
                         [12, 14, 19, 21, 23, 28, 30, 32],
                         [25, 26, 27, 34, 35, 42, 49, 58]]  # (8, 8)

        NeuralScanRegion = [[0, 1, 2, 3, 4, 59, 62, 63],
                            [7, 8, 9, 10, 11, 17, 18, 19],
                            [12, 13, 20, 21, 22, 29, 30, 31],
                            [32, 33, 34, 42, 43, 44, 41, 58],
                            [26, 27, 28, 35, 36, 37, 45, 53],
                            [38, 39, 40, 46, 47, 48, 49, 61],
                            [50, 51, 52, 54, 56, 57, 60, 55],
                            [5, 6, 14, 15, 16, 23, 24, 25]]  # (8, 8)
        region = NeuralScanRegion if self.dataset == "CAS" else BiosemiRegion
        region = np.array(region)  # (8,8)

        [N, C, T] = data.shape  # (256, 64, 256)

        expand_data = []
        # step1: 扩充
        for i in range(N):  # 对于256个样本
            for j in range(region.shape[0]):  # 对于8个脑区
                expand_data.append(data[i])  # 每一个样本要连着放8份
        expand_data = np.array(expand_data)  # (2048, 64, 256)

        # step2: 修改
        for i in range(N * region.shape[0]):  # 对于2048个样本
            sequence = i % region.shape[0]  # sequence是第几个脑区
            for j in region[sequence]:  # 每个脑区里有8个通道
                raw_signal = expand_data[i, j, :]
                mean, var = np.mean(raw_signal), np.var(raw_signal)
                for m in range(T):
                    expand_data[i, j, m] = random.gauss(mean, var)
            # self._draw(expand_data[i])
        x_msr = np.array(expand_data)  # (2048, 64, 256)

        y_msr = [[i for i in range(region.shape[0])] for i in range(N)]  # [256,8]
        y_msr = np.array(y_msr).reshape(N * region.shape[0])  # (2048，)

        return x_msr, y_msr  # (2048, 64, 256)  (2048,)

    def _generate_by_pic(self, data):  # (256, 64, 256)
        # 初始化一个空的列表来存储所有GAF图像
        gaf_images = []

        channel = np.array([26, 30, 39])

        # 遍历每个样本和每个通道，转换为GAF并添加到列表中
        for sample_idx in range(data.shape[0]):
            # for channel_idx in range(data.shape[1]):
            for channel_idx in channel:  # 现在不选择64个通道了，我们只选择3个通道。
                # 获取单个通道的时间序列数据
                time_series = data[sample_idx, channel_idx]

                # 转换为GAF
                gaf_summation = self._time_series_to_gaf(time_series, type='summation')  # 或者 'difference'

                # 将GAF图像添加到列表中
                gaf_images.append(gaf_summation)

        # 将列表中的所有GAF图像重新组织成所需的形状 (256, 64, 256, 256)
        # gaf_images = np.array(gaf_images).reshape((data.shape[0], 64, 256, 256))
        gaf_images = np.array(gaf_images).reshape((data.shape[0], 3, 256, 256))
        return gaf_images  # ((256, 64, 256, 256)  (256,3,256,256)

    def _time_series_to_gaf(self, time_series, type='summation'):
        # 对时间序列进行归一化到 [-1, 1] 的范围
        normalized = (time_series - np.mean(time_series)) / (np.std(time_series) + 1e-8)

        # 将归一化的时间序列映射到余弦和正弦值（极坐标）
        phi = np.arccos(np.clip(normalized, -1.0, 1.0))
        cos_phi = np.cos(phi)
        sin_phi = np.sin(phi)

        if type == 'summation':
            gaf = np.outer(cos_phi, cos_phi) - np.outer(sin_phi, sin_phi)
        elif type == 'difference':
            gaf = np.outer(sin_phi, cos_phi) - np.outer(cos_phi, sin_phi)
        else:
            raise ValueError("Type must be either 'summation' or 'difference'.")

        return gaf

    import numpy as np
    import random

    def _generate_by_ftr(self, data, sample_rate=256):
        """
        在频率域对 EEG 数据进行高斯掩码操作。

        参数:
            data: 输入 EEG 数据，形状为 (256, 64, 256)。
            sample_rate: 采样率，默认为 256 Hz。

        返回:
            x_ftr: 掩码后的数据，形状为 (2560, 64, 256)。
            y_ftr: 对应的频率段标签，形状为 (2560,)。
        """
        N, C, T = data.shape  # (256, 64, 256)
        freq_range = 50  # 只考虑 0-50 Hz
        band_width = 5  # 每 5 Hz 作为一个频率段
        num_bands = freq_range // band_width  # 10 个频率段

        # Step 1: 对数据进行傅里叶变换，转换到频率域
        data_fft = np.fft.fft(data, axis=-1)  # (256, 64, 256)

        # Step 2: 初始化掩码后的数据和标签
        x_ftr = []
        y_ftr = []

        # Step 3: 对每个样本进行频率域掩码操作
        for i in range(N):  # 对于每个样本
            for band in range(num_bands):  # 对于每个频率段
                # 计算当前频率段的起始和结束索引
                start_freq = band * band_width
                end_freq = (band + 1) * band_width
                start_idx = int(start_freq * T / sample_rate)
                end_idx = int(end_freq * T / sample_rate)

                # 复制当前样本的频率域数据
                masked_fft = data_fft[i].copy()  # (64, 256)

                # 对当前频率段进行高斯掩码
                for j in range(C):  # 对于每个通道
                    # 提取当前频率段的频率分量
                    band_signal = masked_fft[j, start_idx:end_idx]
                    mean, var = np.mean(band_signal), np.var(band_signal)

                    # 用高斯分布随机值替换当前频率段
                    masked_fft[j, start_idx:end_idx] = np.random.normal(mean, np.sqrt(var), end_idx - start_idx)

                # 将掩码后的频率域数据转换回时域
                masked_data = np.fft.ifft(masked_fft, axis=-1).real  # (64, 256)

                # 添加到结果中
                x_ftr.append(masked_data)
                y_ftr.append(band)  # 标签为当前频率段的索引

        # Step 4: 将结果转换为 numpy 数组
        x_ftr = np.array(x_ftr)  # (2560, 64, 256)
        y_ftr = np.array(y_ftr)  # (2560,)

        return x_ftr, y_ftr


class MTCNDataGenerate(Dataset):
    def __init__(self, Dataset, x, y, sub_id, mode):  # x(256, 64, 256)  # y (256,)
        # 数据集类型，输入数据，输入标签
        self.dataset = Dataset  # 'THU'
        self.x = x
        self.y = y
        field = "train" if mode == True else "test"
        try:
            self.x_mtr = np.load(
                os.path.join(os.getcwd(), 'Dataset', self.dataset, f'S{sub_id:>02d}', f'x_mtr_{field}.npy'))
            self.y_mtr = np.load(
                os.path.join(os.getcwd(), 'Dataset', self.dataset, f'S{sub_id:>02d}', f'y_mtr_{field}.npy'))
            self.x_msr = np.load(
                os.path.join(os.getcwd(), 'Dataset', self.dataset, f'S{sub_id:>02d}', f'x_msr_{field}.npy'))
            self.y_msr = np.load(
                os.path.join(os.getcwd(), 'Dataset', self.dataset, f'S{sub_id:>02d}', f'y_msr_{field}.npy'))
            self.x_ftr = np.load(
                os.path.join(os.getcwd(), 'Dataset', self.dataset, f'S{sub_id:>02d}', f'x_ftr_{field}.npy'))
            self.y_ftr = np.load(
                os.path.join(os.getcwd(), 'Dataset', self.dataset, f'S{sub_id:>02d}', f'y_ftr_{field}.npy'))
        except:
            self.x_mtr, self.y_mtr = self._generate_by_mtr(x)
            self.x_msr, self.y_msr = self._generate_by_msr(x)
            self.x_ftr, self.y_ftr = self._generate_by_ftr(x)
            np.save(os.path.join(os.getcwd(), 'Dataset', self.dataset, f'S{sub_id:>02d}', f'x_mtr_{field}.npy'),
                    self.x_mtr)
            np.save(os.path.join(os.getcwd(), 'Dataset', self.dataset, f'S{sub_id:>02d}', f'y_mtr_{field}.npy'),
                    self.y_mtr)
            np.save(os.path.join(os.getcwd(), 'Dataset', self.dataset, f'S{sub_id:>02d}', f'x_msr_{field}.npy'),
                    self.x_msr)
            np.save(os.path.join(os.getcwd(), 'Dataset', self.dataset, f'S{sub_id:>02d}', f'y_msr_{field}.npy'),
                    self.y_msr)
            np.save(os.path.join(os.getcwd(), 'Dataset', self.dataset, f'S{sub_id:>02d}', f'x_ftr_{field}.npy'),
                    self.x_ftr)
            np.save(os.path.join(os.getcwd(), 'Dataset', self.dataset, f'S{sub_id:>02d}', f'y_ftr_{field}.npy'),
                    self.y_ftr)
            print(self.x_mtr.shape, self.y_mtr.shape, self.x_msr.shape, self.y_msr.shape, self.x_ftr.shape,
                  self.y_ftr.shape)

    def __getitem__(self, index):
        x = self.x[index, ...]
        y = self.y[index, ...]
        x_mtr = self.x_mtr[9 * index: 9 * index + 9, ...]
        y_mtr = self.y_mtr[9 * index: 9 * index + 9, ...]
        x_msr = self.x_msr[8 * index: 8 * index + 8, ...]
        y_msr = self.y_msr[8 * index: 8 * index + 8, ...]
        # x_ftr = self.x_ftr[10 * index:10 * index + 10, ...]
        # y_ftr = self.y_ftr[10 * index:10 * index + 10, ...]
        x_ftr = self.x_ftr[5 * index: 5 * index + 5, ...]
        y_ftr = self.y_ftr[5 * index: 5 * index + 5, ...]
        return x, y, x_mtr, y_mtr, x_msr, y_msr, x_ftr, y_ftr

    def __len__(self):
        return len(self.x)

    def _draw(self, data):
        # data->(C, T)
        for i in range(data.shape[0]):
            plt.plot(data[i] + i * 0.2)
        plt.show()

    def _generate_by_mtr(self, data):  # (256, 64, 256)
        # 生成 masked temporal recongnition task 数据集
        masked_temporal_martrix = np.array([[0, 28], [29, 34], [35, 40], [41, 51], [52, 64], [65, 85],
                                            [86, 115], [116, 137], [138, 255]])  # (9,2)
        [N, C, T] = data.shape  # (256, 64, 256)

        # step1：扩充
        expand_data = []
        for i in range(N):  # 对于每一个样本
            for j in range(masked_temporal_martrix.shape[0]):  # 对于每一个时间段
                expand_data.append(data[i])  # 每一个样本要连着放9份
        expand_data = np.array(expand_data)  # (2304,) 里面每一个都是(64,256)
        # step2: 修改
        for i in range(N * masked_temporal_martrix.shape[0]):  # 对于现在的2304个样本
            sequence = i % masked_temporal_martrix.shape[0]  # sequence是第几个时间段
            for j in range(C):  # 对于64个通道
                raw_signal = expand_data[i, j,
                             masked_temporal_martrix[sequence][0]: masked_temporal_martrix[sequence][1]]
                mean, var = np.mean(raw_signal), np.var(raw_signal)
                for m in range(masked_temporal_martrix[sequence][0], masked_temporal_martrix[sequence][1]):
                    expand_data[i, j, m] = random.gauss(mean, var)
        x_mtr = np.array(expand_data)  # (2304, 64, 256)

        y_mtr = [[i for i in range(masked_temporal_martrix.shape[0])] for i in
                 range(N)]  # (256, ) 每一个都是[0,1,2,3,4,5,6,7,8]
        y_mtr = np.array(y_mtr).reshape(N * masked_temporal_martrix.shape[0])  # (2304, ) 把每一行展开了

        return x_mtr, y_mtr  # (2304, 64, 256), (2304, )

    def _generate_by_msr(self, data):  # (256, 64, 256)
        # 生成 masked spatial recongnition task 数据集
        BiosemiRegion = [[0, 1, 2, 3, 4, 5, 6, 62],
                         [7, 8, 9, 10, 11, 16, 17, 18],
                         [13, 15, 20, 22, 24, 29, 31, 33],
                         [36, 59, 38, 42, 43, 45, 47, 60],
                         [37, 39, 41, 44, 46, 48, 61, 63],
                         [50, 51, 52, 53, 54, 55, 56, 57],
                         [12, 14, 19, 21, 23, 28, 30, 32],
                         [25, 26, 27, 34, 35, 42, 49, 58]]  # (8, 8)

        NeuralScanRegion = [[0, 1, 2, 3, 4, 59, 62, 63],
                            [7, 8, 9, 10, 11, 17, 18, 19],
                            [12, 13, 20, 21, 22, 29, 30, 31],
                            [32, 33, 34, 42, 43, 44, 41, 58],
                            [26, 27, 28, 35, 36, 37, 45, 53],
                            [38, 39, 40, 46, 47, 48, 49, 61],
                            [50, 51, 52, 54, 56, 57, 60, 55],
                            [5, 6, 14, 15, 16, 23, 24, 25]]  # (8, 8)
        region = NeuralScanRegion if self.dataset == "CAS" else BiosemiRegion
        region = np.array(region)  # (8,8)

        [N, C, T] = data.shape  # (256, 64, 256)

        expand_data = []
        # step1: 扩充
        for i in range(N):  # 对于256个样本
            for j in range(region.shape[0]):  # 对于8个脑区
                expand_data.append(data[i])  # 每一个样本要连着放8份
        expand_data = np.array(expand_data)  # (2048, 64, 256)

        # step2: 修改
        for i in range(N * region.shape[0]):  # 对于2048个样本
            sequence = i % region.shape[0]  # sequence是第几个脑区
            for j in region[sequence]:  # 每个脑区里有8个通道
                raw_signal = expand_data[i, j, :]
                mean, var = np.mean(raw_signal), np.var(raw_signal)
                for m in range(T):
                    expand_data[i, j, m] = random.gauss(mean, var)
            # self._draw(expand_data[i])
        x_msr = np.array(expand_data)  # (2048, 64, 256)

        y_msr = [[i for i in range(region.shape[0])] for i in range(N)]  # [256,8]
        y_msr = np.array(y_msr).reshape(N * region.shape[0])  # (2048，)

        return x_msr, y_msr  # (2048, 64, 256)  (2048,)

    def _generate_by_ftr(self, data, sample_rate=256):
        """
        在频率域对 EEG 数据进行高斯掩码操作，使用脑电标准频段划分。
        参数:
            data: 输入 EEG 数据，形状为 (256, 64, 256)。
            sample_rate: 采样率，默认为 256 Hz。

        返回:
            x_ftr: 掩码后的数据，形状为 (1280, 64, 256)。
            y_ftr: 对应的频率段标签，形状为 (1280,)。
        """
        N, C, T = data.shape  # (256, 64, 256)

        # 定义脑电标准频段 [1,2,3](@ref)
        bands = [
            ('Delta', 0.5, 4),  # δ波：0.5～4Hz
            ('Theta', 4, 8),  # θ波：4～8Hz
            ('Alpha', 8, 12),  # α波：8～13Hz
            ('Beta', 12, 30),  # β波：12～30Hz
            ('Gamma', 30, 80)  # γ波：30～80Hz
        ]
        num_bands = len(bands)  # 5个频率段

        # Step 1: 对数据进行傅里叶变换，转换到频率域
        data_fft = np.fft.fft(data, axis=-1)  # (256, 64, 256)

        # Step 2: 初始化掩码后的数据和标签
        x_ftr = []
        y_ftr = []

        # Step 3: 对每个样本进行频率域掩码操作
        for i in range(N):  # 对于每个样本
            for band_idx, (band_name, low_freq, high_freq) in enumerate(bands):  # 对于每个脑电频段
                # 计算当前频率段的起始和结束索引
                start_idx = int(low_freq * T / sample_rate)
                end_idx = int(high_freq * T / sample_rate)

                # 确保索引不超出范围
                start_idx = max(0, min(start_idx, T))
                end_idx = max(0, min(end_idx, T))

                if start_idx >= end_idx:
                    continue  # 跳过无效的频段

                # 复制当前样本的频率域数据
                masked_fft = data_fft[i].copy()  # (64, 256)

                # 对当前频率段进行高斯掩码
                for j in range(C):  # 对于每个通道
                    # 提取当前频率段的频率分量
                    band_signal = masked_fft[j, start_idx:end_idx]
                    if len(band_signal) == 0:
                        continue

                    mean, var = np.mean(band_signal), np.var(band_signal)

                    # 用高斯分布随机值替换当前频率段
                    masked_fft[j, start_idx:end_idx] = np.random.normal(
                        mean, np.sqrt(var + 1e-8), end_idx - start_idx
                    )

                # 将掩码后的频率域数据转换回时域
                masked_data = np.fft.ifft(masked_fft, axis=-1).real  # (64, 256)

                # 添加到结果中
                x_ftr.append(masked_data)
                y_ftr.append(band_idx)  # 标签为当前频率段的索引

        # Step 4: 将结果转换为 numpy 数组
        x_ftr = np.array(x_ftr)  # (1280, 64, 256) - 256个样本 * 5个频段 = 1280
        y_ftr = np.array(y_ftr)  # (1280,)

        # print(f"生成的掩码数据形状: {x_ftr.shape}")
        # print(f"频率段标签分布: {np.bincount(y_ftr)}")
        # print("频率段对应关系:")
        # for band_idx, (band_name, low_freq, high_freq) in enumerate(bands):
        #     print(f"  标签 {band_idx}: {band_name}波 ({low_freq}-{high_freq}Hz)")

        return x_ftr, y_ftr

    #
    # def _generate_by_ftr(self, data, sample_rate=256):
    #     """
    #     在频率域对 EEG 数据进行高斯掩码操作。
    #
    #     参数:
    #         data: 输入 EEG 数据，形状为 (256, 64, 256)。
    #         sample_rate: 采样率，默认为 256 Hz。
    #
    #     返回:
    #         x_ftr: 掩码后的数据，形状为 (2560, 64, 256)。
    #         y_ftr: 对应的频率段标签，形状为 (2560,)。
    #     """
    #     N, C, T = data.shape  # (256, 64, 256)
    #     freq_range = 50  # 只考虑 0-50 Hz
    #     band_width = 5  # 每 5 Hz 作为一个频率段
    #     num_bands = freq_range // band_width  # 10 个频率段
    #
    #     # Step 1: 对数据进行傅里叶变换，转换到频率域
    #     data_fft = np.fft.fft(data, axis=-1)  # (256, 64, 256)
    #
    #     # Step 2: 初始化掩码后的数据和标签
    #     x_ftr = []
    #     y_ftr = []
    #
    #     # Step 3: 对每个样本进行频率域掩码操作
    #     for i in range(N):  # 对于每个样本
    #         for band in range(num_bands):  # 对于每个频率段
    #             # 计算当前频率段的起始和结束索引
    #             start_freq = band * band_width
    #             end_freq = (band + 1) * band_width
    #             start_idx = int(start_freq * T / sample_rate)
    #             end_idx = int(end_freq * T / sample_rate)
    #
    #             # 复制当前样本的频率域数据
    #             masked_fft = data_fft[i].copy()  # (64, 256)
    #
    #             # 对当前频率段进行高斯掩码
    #             for j in range(C):  # 对于每个通道
    #                 # 提取当前频率段的频率分量
    #                 band_signal = masked_fft[j, start_idx:end_idx]
    #                 mean, var = np.mean(band_signal), np.var(band_signal)
    #
    #                 # 用高斯分布随机值替换当前频率段
    #                 masked_fft[j, start_idx:end_idx] = np.random.normal(mean, np.sqrt(var), end_idx - start_idx)
    #
    #             # 将掩码后的频率域数据转换回时域
    #             masked_data = np.fft.ifft(masked_fft, axis=-1).real  # (64, 256)
    #
    #             # 添加到结果中
    #             x_ftr.append(masked_data)
    #             y_ftr.append(band)  # 标签为当前频率段的索引
    #
    #     # Step 4: 将结果转换为 numpy 数组
    #     x_ftr = np.array(x_ftr)  # (2560, 64, 256)
    #     y_ftr = np.array(y_ftr)  # (2560,)
    #
    #     return x_ftr, y_ftr


class GeneralData(Dataset):
    # 通用模型的数据装载器
    def __init__(self, x, y):
        super(GeneralData, self).__init__()
        self.x = x
        self.y = y

    def __getitem__(self, index):
        x = self.x[index, ...]
        y = self.y[index, ...]
        return x, y

    def __len__(self):
        return len(self.x)


class MEDGNNTestData(Dataset):
    # 通用模型的数据装载器
    def __init__(self, x, y):
        super(MEDGNNTestData, self).__init__()
        self.x_total = np.transpose(x, (0, 2, 1))
        self.y_total = y


    def __getitem__(self, index):
        x = self.x_total[index, ...]
        y = self.y_total[index, ...]
        return x, y

    def __len__(self):
        return len(self.x_total)



class MTCNTestData(Dataset):
    # 通用模型的数据装载器
    def __init__(self, x, y):
        super(MTCNTestData, self).__init__()
        self.x_test, self.y = self._select_x(x, y)
        self.x_pic = self._generate_by_pic(self.x_test)

    def __getitem__(self, index):
        x_test = self.x_test[index, ...]
        y = self.y[index, ...]
        x_pic = self.x_pic[index, ...]
        return x_test, y, x_pic

    def __len__(self):
        return len(self.x_test)

    def _select_x(self, data, y):
        # 分离两类的索引
        indices_0 = np.where(y == 0)[0]
        indices_1 = np.where(y == 1)[0]

        # 确定较小类别的大小，并进行欠采样
        min_class_size = min(len(indices_0), len(indices_1))
        np.random.seed(42)  # 设置随机种子以保证结果可复现
        selected_indices_0 = np.random.choice(indices_0, size=min_class_size, replace=False)
        selected_indices_1 = np.random.choice(indices_1, size=min_class_size, replace=False)

        # 合并选择的索引并排序以保持原始顺序
        selected_indices = np.concatenate([selected_indices_0, selected_indices_1])
        selected_indices.sort()

        # 根据选择的索引获取平衡后的数据和标签
        balanced_data = data[selected_indices]
        balanced_y = y[selected_indices]

        return balanced_data, balanced_y

    def _generate_by_pic(self, data):  # (256, 64, 256)
        # 初始化一个空的列表来存储所有GAF图像
        gaf_images = []

        channel = np.array([26, 30, 39])
        # 遍历每个样本和每个通道，转换为GAF并添加到列表中
        for sample_idx in range(data.shape[0]):
            # for channel_idx in range(data.shape[1]):
            for channel_idx in channel:
                # 获取单个通道的时间序列数据
                time_series = data[sample_idx, channel_idx]

                # 转换为GAF
                gaf_summation = self._time_series_to_gaf(time_series, type='summation')  # 或者 'difference'

                # 将GAF图像添加到列表中
                gaf_images.append(gaf_summation)

        # 将列表中的所有GAF图像重新组织成所需的形状 (242, 64, 256, 256)
        gaf_images = np.array(gaf_images).reshape((data.shape[0], 3, 256, 256))
        return gaf_images  # ((242, 64, 256, 256)

    def _time_series_to_gaf(self, time_series, type='summation'):
        # 对时间序列进行归一化到 [-1, 1] 的范围
        normalized = (time_series - np.mean(time_series)) / (np.std(time_series) + 1e-8)

        # 将归一化的时间序列映射到余弦和正弦值（极坐标）
        phi = np.arccos(np.clip(normalized, -1.0, 1.0))
        cos_phi = np.cos(phi)
        sin_phi = np.sin(phi)

        if type == 'summation':
            gaf = np.outer(cos_phi, cos_phi) - np.outer(sin_phi, sin_phi)
        elif type == 'difference':
            gaf = np.outer(sin_phi, cos_phi) - np.outer(cos_phi, sin_phi)
        else:
            raise ValueError("Type must be either 'summation' or 'difference'.")

        return gaf


class DataManage:
    def __init__(self, Name, Mode, DataName, SubID, BatchSize):
        # Name (str): the name of method 
        # Mode (bool): True denotes Train mode, False denote Test mode
        # DataName (str): 数据集名
        self.method_name = Name
        self.mode = Mode
        self.dataset = DataName
        self.sub_id = SubID
        self.batch_size = BatchSize
        self.preprocesser = DataProcess()

    def getData(self):
        # 基础网络训练集和测试集数据只有是否shuffle的区别（EEGNet, DeepConvNet et al）
        # 其它有数据增强操作的网络需要自定义（DRL, MTCN et al）

        # 加载原始数据
        field = "train" if self.mode == True else "test"
        x = np.load(os.path.join(os.getcwd(), 'Dataset', self.dataset, f'S{self.sub_id:>02d}', f'x_{field}.npy'))
        y = np.load(os.path.join(os.getcwd(), 'Dataset', self.dataset, f'S{self.sub_id:>02d}', f'y_{field}.npy'))

        x, y = np.array(x, dtype='float32'), np.array(y, dtype='float32')
        # 数据预处理
        x = self.preprocesser.band_pass_filter(data=x, freq_low=0.1, freq_high=40, fs=256)
        x = self.preprocesser.scale_data(x)

        x_npy, y_npy = copy.deepcopy(x), copy.deepcopy(y)  # (256,64,256)  (256,)

        # （MTCN et al）带来的附加操作
        if self.method_name == "MTCN":
            if self.mode == True:  # 训练
                # data = MTCNPicDataGenerate(self.dataset, x, y, self.sub_id, self.mode)
                data = MTCNDataGenerate(self.dataset, x, y, self.sub_id, self.mode)
                data_loader = DataLoader(data, batch_size=self.batch_size, shuffle=(self.mode is True))
            else:  # 测试
                data = GeneralData(x, y)
                # data = MTCNTestData(x, y)
                data_loader = DataLoader(data, batch_size=self.batch_size, shuffle=(self.mode is True))
        elif self.method_name == "VisionEagle":
            if self.mode == True:
                data = VisionEagleDataGenerate(self.dataset, x, y, self.sub_id, self.mode)
                data_loader = DataLoader(data, batch_size=self.batch_size, shuffle=(self.mode is True))
            else:
                data = GeneralData(x, y)
                data_loader = DataLoader(data, batch_size=self.batch_size, shuffle=(self.mode is True))
        elif self.method_name == "MEDGNN":
            if self.mode == True:
                data = MEDGNNDataGenerate(self.dataset, x, y, self.sub_id, self.mode)
                data_loader = DataLoader(data,
                                         batch_size=self.batch_size,
                                         shuffle=(self.mode is True),
                                         collate_fn=lambda x: collate_fn(
                                                            x, max_len=256
                                                        ),
                                         )
            else:
                data = MEDGNNTestData(x, y)
                data_loader = DataLoader(data, batch_size=self.batch_size, shuffle=(self.mode is True),collate_fn=lambda x: collate_fn(
                                                            x, max_len=256
                                                        ),)
        else:
            data = GeneralData(x, y)
            data_loader = DataLoader(data, batch_size=self.batch_size, shuffle=(self.mode is True))
        del x, y, field
        gc.collect()
        # data_loader 为深度学习方法准备，后者为传统方法准备
        return data_loader, x_npy, y_npy
