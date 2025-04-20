import cv2
import os.path
import torch
from torchvision import transforms
import numpy as np
import glob
from torch.autograd import Variable
import datetime
import time

import json
import matplotlib.pyplot as plt
from sklearn.metrics import average_precision_score
from utils.autoMetric import *

class Validator(object):

    def __init__(self, valid_img_dir, valid_lab_dir, valid_result_dir, valid_log_dir, best_model_dir,
                 net,image_format = "jpg",lable_format = "png", datasetName=None,normalize = False):

        self.valid_img_dir = valid_img_dir  
        self.valid_lab_dir = valid_lab_dir 
        self.valid_res_dir = valid_result_dir 
        self.best_model_dir = best_model_dir
        self.valid_log_dir = valid_log_dir + "/"+datasetName+"_valid.txt" 
        self.image_format = image_format
        self.lable_format = lable_format
        self.dataname = datasetName
        
        self.res = {'ODS':0.0 ,'OIS':0.0, 'F1':0.0, 'mIoU':0.0, 'DICE':0.0, 'mAP':0.0, 'AUC':0.0, 'FPS':0.0}

        if os.path.exists(self.best_model_dir)==False:
            os.makedirs(self.best_model_dir)
        self.net = net
        self.normalize = normalize

        if self.normalize:
            self.transforms = transforms.Compose(
                [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        else:
            self.transforms = transforms.ToTensor()

    def update_metrics(self, ods, ois, f1, mIoU, DICE, mAP, AUC, FPS):
        self.res['ODS'] = max(self.res['ODS'], ods)
        self.res['OIS'] = max(self.res['OIS'], ois)
        self.res['F1'] = max(self.res['F1'], f1)
        self.res['mIoU'] = max(self.res['mIoU'], mIoU)
        self.res['DICE'] = max(self.res['DICE'], DICE)
        self.res['mAP'] = max(self.res['mAP'], mAP)
        self.res['AUC'] = max(self.res['AUC'], AUC)
        self.res['FPS'] = max(self.res['FPS'], FPS)

    def make_dir(self):
        try:
            if not os.path.exists(self.valid_res_dir):
                os.makedirs(self.valid_res_dir)
        except:
            print("创建valid_res文件失败")


    def make_dataset(self, epoch_num):
        pred_imgs, gt_imgs = [], []
        for pred_path in glob.glob(os.path.join(self.valid_res_dir + str(epoch_num) + "/", "*." + self.image_format)):

            gt_path = os.path.join(self.valid_lab_dir, os.path.basename(pred_path)[:-4] + "." + self.lable_format)
        
            gt_img = self.imread(gt_path, thresh=80)
            pred_img = self.imread(pred_path, gt_img)
            gt_imgs.append(gt_img)
            pred_imgs.append(pred_img)

        return pred_imgs, gt_imgs

    def imread(self, path, rgb2gray=None, load_size=0, load_mode=cv2.IMREAD_GRAYSCALE, convert_rgb=False, thresh=-1):
        # print(path)
        im = cv2.imread(path, load_mode)
        H,W=im.shape[0],im.shape[1]
        if H==448 and W==448:
            im=cv2.resize(im,(512,512),cv2.INTER_NEAREST)
        if H==600 and W==800:
            if im.shape==3:
                im=im[:592,::,::]
            else:
                im=im[:592,::]
        elif H==720 and W==960:
            if im.shape==3:
                im=im[:592,:800,::]
            else:
                im=im[:592,:800]  
        if convert_rgb:
            im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        if load_size > 0:
            im = cv2.resize(im, (load_size, load_size), interpolation=cv2.INTER_CUBIC)
        if thresh > 0:
            _, im = cv2.threshold(im, thresh, 255, cv2.THRESH_BINARY)
        else:
            im = ((rgb2gray == 255) + (rgb2gray == 0)) * im
        return im

    def get_statistics(self, pred, gt):
        """
        return tp, fp, fn
        """
        tp = np.sum((pred == 1) & (gt == 1))
        fp = np.sum((pred == 1) & (gt == 0))
        fn = np.sum((pred == 0) & (gt == 1))
        
        return [tp, fp, fn]


    def cal_prf_metrics(self, pred_list, gt_list, thresh_step=0.01):
        final_accuracy_all = []
        for thresh in np.arange(0.0, 1.0, thresh_step):
            statistics = []

            for pred, gt in zip(pred_list, gt_list):
            
                gt_img = (gt / 255).astype('uint8')
                pred_img = ((pred / 255) > thresh).astype('uint8')
                # calculate each image
                statistics.append(self.get_statistics(pred_img, gt_img))
                

            # get tp, fp, fn
            tp = np.sum([v[0] for v in statistics])
            fp = np.sum([v[1] for v in statistics])
            fn = np.sum([v[2] for v in statistics])
            

            # calculate precision
            p_acc = 1.0 if tp == 0 and fp == 0 else tp / (tp + fp  + 1e-4)
            # calculate recall
            r_acc = tp / (tp + fn  + 1e-4)
            # mIoU
            m_IoU = tp / (tp + fp + fn  + 1e-4)

            # calculate f-score
            final_accuracy_all.append([thresh, p_acc, r_acc, m_IoU, 2 * p_acc * r_acc / (p_acc + r_acc  + 1e-4)])
        

        return final_accuracy_all


    def cal_ois_metrics(self,pred_list, gt_list, thresh_step=0.01):
        final_acc_all = []
        for pred, gt in zip(pred_list, gt_list):
            statistics = []
            for thresh in np.arange(0.0, 1.0, thresh_step):
                gt_img = (gt / 255).astype('uint8')
                pred_img = (pred / 255 > thresh).astype('uint8')
                tp, fp, fn = self.get_statistics(pred_img, gt_img)
                p_acc = 1.0 if tp == 0 and fp == 0 else tp / (tp + fp)
                r_acc = tp / (tp + fn)

                if p_acc + r_acc == 0:
                    f1 = 0
                else:
                    f1 = 2 * p_acc * r_acc / (p_acc + r_acc)
                statistics.append([thresh, f1])
            max_f = np.amax(statistics, axis=0)
            final_acc_all.append(max_f[1])
        return np.mean(final_acc_all)

    def validate(self, epoch_num):
        print('开始验证')
        image_list = os.listdir(self.valid_img_dir)

        t_all = []

        self.net.eval() 
        with torch.no_grad():
            for image_name in image_list:
                image = os.path.join(self.valid_img_dir, image_name)

                image = cv2.imread(image)
                if image.shape[0]==448:
                    image=cv2.resize(image,(512,512),cv2.INTER_NEAREST)
                if image.shape[0]==600:
                    image=image[:592,:800,::]
                elif image.shape[1]==720:
                    image=image[:592,:800,::]
                x = Variable(self.transforms(image))
                x = x.unsqueeze(0).cuda()

                t1 = time.time()
                outs = self.net.forward(x)  
                t2 = time.time()
                t_all.append(t2 - t1)

                # y = outs[-1].squeeze(1)
                y=outs  
                output = torch.sigmoid(y)
                # print(output.shape)
                out_clone = output.clone()
                img_fused = np.squeeze(out_clone.cpu().detach().numpy(), axis=0)
                # print(img_fused.shape)
                img_fused = np.transpose(img_fused, (1, 2, 0))
                if os.path.exists(self.valid_res_dir  + str(epoch_num) )==False:
                    os.mkdir(self.valid_res_dir  + str(epoch_num) )
                cv2.imwrite(self.valid_res_dir  + str(epoch_num) + '/' + image_name, img_fused * 255.0)

        img_list, gt_list = self.make_dataset(epoch_num)
        final_results = self.cal_prf_metrics(img_list, gt_list, 0.01)  # thresh, p, r, m_IoU, ods
        final_ois = self.cal_ois_metrics(img_list, gt_list, 0.01)
        

        ap = calAP(final_results)
        fps = calfps(t_all)
        # autores = drawCurve(gt_list, [img_list], ['unet'], self.dataname) # FPR,TPR,AUC,MAP,IOU
        process = CollectData()
        process.reload(gt_list, img_list)
        (FPR, TPR, AUC), (Precision, Recall, MAP), F1, IoU, DICE = process.statistics()
        
        max_f = np.amax(final_results, axis=0)

        self.update_metrics(max_f[-1], final_ois, F1, max(max_f[-2], IoU), DICE, max(ap, MAP), AUC, 0)
        if max_f[-1] > self.res['ODS'] :
            ods_str = datetime.datetime.now().strftime("%Y-%m-%d-%H-") + str(max_f[3])[0:5]
            if max_f[3]>=0.3:
                print('save ' + ods_str)
                torch.save(self.net.state_dict(), self.best_model_dir + ods_str + ".pth")
        with open(self.valid_log_dir, 'a', encoding='utf-8') as fout:
            line = time.strftime('%m-%d %X')+" epoch:{} | ODS:{:.6f} | OIS:{:.6f} | F1:{:.6f} | mIoU:{:.6f} {:.6f} | DICE:{:.6f} | AP:{:.6f} {:.6f} | AUC:{:.6f} | FPS:{:.6f}"\
                .format(epoch_num, max_f[-1], final_ois, F1, max_f[-2], IoU, DICE, ap, MAP, AUC, fps) + '\n'
            fout.write(line)
        print("epoch={} | ODS:{:.6f} | OIS:{:.6f} | F1:{:.6f} | mIoU:{:.6f} {:.6f} | DICE:{:.6f} | AP:{:.6f} {:.6f} | AUC:{:.6f} | FPS:{:.6f}"
              .format(epoch_num, max_f[-1], final_ois, F1, max_f[-2], IoU, DICE, ap, MAP, AUC, fps))

        if epoch_num == 'debug' or epoch_num == 496:
            res_formatted = {k: float(v) for k, v in self.res.items()}
            with open(self.valid_log_dir, 'a', encoding='utf-8') as fout:
                fout.write('\n' + json.dumps(res_formatted, ensure_ascii=False) + '\n')
            torch.save(self.net.state_dict(), self.best_model_dir + str(epoch_num) + ".pth")


def calAP(final_accuracy_all):

    P = [item[1] for item in final_accuracy_all]
    R = [item[2] for item in final_accuracy_all]

    sorted_indices = np.argsort(P)[::-1]
    R_sorted = np.array(R)[sorted_indices]
    P_sorted = np.array(P)[sorted_indices]

    max_precision = np.maximum.accumulate(P_sorted)

    area = np.trapz(max_precision, R_sorted)

    AP = area / (1-R_sorted[0])
    print("Average Precision (AP):", AP)
    return AP

def calfps(t_all):
    fps = 1 / np.mean(t_all)
    return fps
