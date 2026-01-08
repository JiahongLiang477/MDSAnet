import numpy as np
import torch

class Evaluator(object):
    def __init__(self, num_class):
        self.num_class = num_class
        self.confusion_matrix = np.zeros((self.num_class,) * 2)
        # matrix shape(num_class, num_class) with elements 0 in our match. it will be 4*4

    def Kappa(self):
        xsum = np.sum(self.confusion_matrix, axis=1)  # sum by row
        ysum = np.sum(self.confusion_matrix, axis=0)  # sum by column

        Pe = np.sum(ysum * xsum) * 1.0 / (self.confusion_matrix.sum() ** 2)
        P0 = np.diag(self.confusion_matrix).sum() / self.confusion_matrix.sum()  # predict right / all the data
        cohens_coefficient = (P0 - Pe) / (1 - Pe)

        return cohens_coefficient

    def ProducerA(self):
        #
        return np.diag(self.confusion_matrix) / np.sum(self.confusion_matrix, axis=1)

    def UserA(self):
        #
        return np.diag(self.confusion_matrix) / np.sum(self.confusion_matrix, axis=0)

    def Pixel_Accuracy(self):
        Acc = np.diag(self.confusion_matrix).sum() / self.confusion_matrix.sum()
        return Acc

    def Pixel_Accuracy_Class(self):
        Acc = np.diag(self.confusion_matrix) / self.confusion_matrix.sum(axis=1)
        # each pred right class is in diag. sum by row is the count of corresponding class
        mAcc = np.nanmean(Acc)  #
        return mAcc, Acc

    def Mean_Intersection_over_Union(self):
        IoU = np.diag(self.confusion_matrix) / (
                np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) -
                np.diag(self.confusion_matrix))
        MIoU = np.nanmean(IoU)
        return MIoU, IoU

    def Frequency_Weighted_Intersection_over_Union(self):
        freq = np.sum(self.confusion_matrix, axis=1) / np.sum(self.confusion_matrix)
        iu = np.diag(self.confusion_matrix) / (
                np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) -
                np.diag(self.confusion_matrix))

        FWIoU = (freq[freq > 0] * iu[freq > 0]).sum()
        return FWIoU

    def _generate_matrix(self, gt_image, pre_image):
        # gt_image = batch_size*256*256   pre_image = batch_size*256*256
        mask = (gt_image >= 0) & (gt_image < self.num_class)  # valid in mask show True, ignored in mask show False
        label = self.num_class * gt_image[mask].astype('int') + pre_image[mask]
        # gt_image[mask] : find out valid pixels. elements with 0,1,2,3 , so label range in  0-15
        count = np.bincount(label, minlength=self.num_class ** 2)
        # [0, 1, 2, 3,  confusion_matrix like this:
        #  4, 5, 6, 7,  and if the element is on the diagonal, it means predict the right class.
        #  8, 9, 10,11, row means the real label, column means pred label
        #  12,13,14,15]
        # return a array [a,b....], each letters holds the count of a class and map to class0, class1...
        confusion_matrix = count.reshape(self.num_class, self.num_class)
        return confusion_matrix

    def add_batch(self, gt_image, pre_image):
        assert gt_image.shape == pre_image.shape
        self.confusion_matrix += self._generate_matrix(gt_image, pre_image)

    def reset(self):
        self.confusion_matrix = np.zeros((self.num_class,) * 2)



def evaluate_OA(data_iter, net, loss, device, model_type_flag):  # for train_patch
    acc_sum, samples_counter = 0, 0

    with torch.no_grad():
        net.eval()
        if model_type_flag == 1:  # data for single spatial net
            for X_spa, y in data_iter:
                loss_sum = 0
                X_spa, y = X_spa.to(device), y.to(device)
                y_pred = net(X_spa)

                ls = loss(y_pred, y.long())

                acc_sum += (y_pred.argmax(dim=1) == y).sum().cpu().item()
                loss_sum += ls

                samples_counter += y.shape[0]
        elif model_type_flag == 2:  # data for single spectral net
            for X_spe, y in data_iter:
                loss_sum = 0
                X_spe, y = X_spe.to(device), y.to(device)
                y_pred = net(X_spe)

                ls = loss(y_pred, y.long())

                acc_sum += (y_pred.argmax(dim=1) == y).sum().cpu().item()
                loss_sum += ls

                samples_counter += y.shape[0]
        elif model_type_flag == 3:  # data for spectral-spatial net
            for X_spa, X_spe, y in data_iter:
                loss_sum = 0
                X_spa, X_spe, y = X_spa.to(device), X_spe.to(device), y.to(device)
                y_pred = net(X_spa, X_spe)

                ls = loss(y_pred, y.long())

                acc_sum += (y_pred.argmax(dim=1) == y).sum().cpu().item()
                loss_sum += ls

                samples_counter += y.shape[0]

    return [acc_sum / samples_counter, loss_sum]



class Evaluator(object):
    def __init__(self, num_class):
        self.num_class = num_class
        self.confusion_matrix = np.zeros((self.num_class,) * 2)
        # matrix shape(num_class, num_class) with elements 0 in our match. it will be 4*4

    def Kappa(self):
        xsum = np.sum(self.confusion_matrix, axis=1)  # sum by row
        ysum = np.sum(self.confusion_matrix, axis=0)  # sum by column

        Pe = np.sum(ysum * xsum) * 1.0 / (self.confusion_matrix.sum() ** 2)
        P0 = np.diag(self.confusion_matrix).sum() / self.confusion_matrix.sum()  # predict right / all the data
        cohens_coefficient = (P0 - Pe) / (1 - Pe)

        return cohens_coefficient

    def ProducerA(self):
        #
        return np.diag(self.confusion_matrix) / np.sum(self.confusion_matrix, axis=1)

    def UserA(self):
        #
        return np.diag(self.confusion_matrix) / np.sum(self.confusion_matrix, axis=0)

    def Pixel_Accuracy(self):
        Acc = np.diag(self.confusion_matrix).sum() / self.confusion_matrix.sum()
        return Acc

    def Pixel_Accuracy_Class(self):
        Acc = np.diag(self.confusion_matrix) / self.confusion_matrix.sum(axis=1)
        # each pred right class is in diag. sum by row is the count of corresponding class
        mAcc = np.nanmean(Acc)  #
        return mAcc, Acc

    def Mean_Intersection_over_Union(self):
        IoU = np.diag(self.confusion_matrix) / (
                np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) -
                np.diag(self.confusion_matrix))
        MIoU = np.nanmean(IoU)
        return MIoU, IoU

    def Frequency_Weighted_Intersection_over_Union(self):
        freq = np.sum(self.confusion_matrix, axis=1) / np.sum(self.confusion_matrix)
        iu = np.diag(self.confusion_matrix) / (
                np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) -
                np.diag(self.confusion_matrix))

        FWIoU = (freq[freq > 0] * iu[freq > 0]).sum()
        return FWIoU

    def _generate_matrix(self, gt_image, pre_image):
        # gt_image = batch_size*256*256   pre_image = batch_size*256*256
        mask = (gt_image >= 0) & (gt_image < self.num_class)  # valid in mask show True, ignored in mask show False
        label = self.num_class * gt_image[mask].astype('int') + pre_image[mask]
        count = np.bincount(label, minlength=self.num_class ** 2)
        confusion_matrix = count.reshape(self.num_class, self.num_class)
        return confusion_matrix

    def add_confusion_matrix(self, confusion_matrix):
        self.confusion_matrix += confusion_matrix

    def add_batch(self, gt_image, pre_image):
        assert gt_image.shape == pre_image.shape
        self.confusion_matrix += self._generate_matrix(gt_image, pre_image)

    def reset(self):
        self.confusion_matrix = np.zeros((self.num_class,) * 2)