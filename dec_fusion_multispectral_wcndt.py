import copy

import numpy
import torch
import cv2
import numpy as np
import sklearn.metrics as skm
import torchvision.transforms as transforms
from model import ClassiFilerNet, ResNet50
from utils.torch_utils import select_device, load_classifier, time_synchronized
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import os
from utils import SSIM
from utils import fda
from PIL import Image
from utils import guidefilter
from utils.deep_DIM import deep_DIM_test_once as deepDIM
import torch.nn.functional as F

# change the Dec_Fusion_Multispectral


def letterbox(img, new_shape=(64, 64), color=(128, 128, 128), auto=False, scaleFill=False, scaleup=True, stride=32):
    # Resize and pad image while meeting stride-multiple constraints
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better test mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return img, ratio, (dw, dh)



class MatchnetDD(object):
    __instance = None
    __init_flag = False
    def __new__(cls, *args, **kwargs):
        if cls.__instance == None:
            cls.__instance = object.__new__(cls)
            return cls.__instance
        else:
            return cls.__instance

    def __init__(self, opt):
        if self.__init_flag ==False:
            self.opt = opt
            self.device = select_device(opt.device)
            self.model_feature = ResNet50()
            self.model = ClassiFilerNet("resnet")
            weights_cls = opt.weights_cls
            # load model
            print("=> loading checkpoint '{}'".format(weights_cls))
            checkpoint = torch.load(weights_cls, map_location = self.device)
            self.model.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(weights_cls, checkpoint['epoch']))
            self.model_feature.to(self.device).eval()
            self.model.to(self.device).eval()
            self.__init_flag = True

        else:
            pass

    def matchnetDD_similar(self,object1, object2):

        object1 = self.transform_image(object1)
        object2 = self.transform_image(object2)

        feature1 = self.model_feature(object1)
        feature1 = feature1.reshape((object1.shape[0], -1))
        feature2 = self.model_feature(object2)
        feature2 = feature2.reshape((object2.shape[0], -1))
        out = self.model((feature1, feature2))
        return out[0][1]

    def transform_image(self,img):
        img = letterbox(img, 64, stride=2)[0]
        img = torch.from_numpy((img - 128) / 160).to(torch.float32)
        img = transforms.ToPILImage()(img)
        img = img.convert("RGB")
        img = transforms.ToTensor()(img).reshape(1, 3, 64, 64).to(self.device)
        return img


def intersect(box_a, box_b):

    A = box_a.size(0)
    B = box_b.size(0)
    max_xy = torch.min(box_a[:, 2:].unsqueeze(1).expand(A, B, 2),
                       box_b[:, 2:].unsqueeze(0).expand(A, B, 2))
    min_xy = torch.max(box_a[:, :2].unsqueeze(1).expand(A, B, 2),
                       box_b[:, :2].unsqueeze(0).expand(A, B, 2))
    inter = torch.clamp((max_xy - min_xy), min=0)
    return inter[:, :, 0] * inter[:, :, 1]


def jaccard(box_a, box_b):

    inter = intersect(box_a, box_b)
    area_a = ((box_a[:, 2]-box_a[:, 0]) *
              (box_a[:, 3]-box_a[:, 1])).unsqueeze(1).expand_as(inter)  # [A,B]
    area_b = ((box_b[:, 2]-box_b[:, 0]) *
              (box_b[:, 3]-box_b[:, 1])).unsqueeze(0).expand_as(inter)  # [A,B]
    union = area_a + area_b - inter
    return inter / union  # [A,B]

def contain(box_a, box_b):

    inter = intersect(box_a, box_b)
    area_a = ((box_a[:, 2]-box_a[:, 0]) *
              (box_a[:, 3]-box_a[:, 1])).unsqueeze(1).expand_as(inter)  # [A,B]

    return inter / area_a  # inter/A

def Dec_Fusion_Object_Detection(pred,pred_ir,iou_thres):
    if len(pred[0]) == 0:
        pred_fuse = copy.deepcopy(pred_ir)
    elif len(pred_ir[0]) == 0:
        pred_fuse = copy.deepcopy(pred)
    else:
        pred_fuse = copy.deepcopy(pred)
        fuse_iou = jaccard(pred_ir[0][:, :4], pred[0][:, :4])
        m, n = fuse_iou.shape
        for i in range(m):
            temp1 = []
            add = True
            temp4 = 0
            for j in range(n):

                if(fuse_iou[i][j] > iou_thres):
                    print('Fuse_Done_Better')
                    add = False
                    if((pred_ir[0][i][4] > pred[0][j][4]) and (fuse_iou[i][j] > temp4)) :
                        temp4 = fuse_iou[i][j]
                        pred[0][j][4:6] = pred_ir[0][i][4:6]
            # if add == False:
            #     pred_ir[0]=torch.cat([pred_ir[0][0:i], pred_ir[0][i+1:]], dim = 0)

            if add == True:
                print('Fuse_Done_Add')
                # temp2 = torch.cat([pred_fuse[0], pred_ir[0]], dim=0)
                # temp1.append(temp2)
                # pred_fuse=temp1
                pred_fuse[0] = torch.cat([pred_fuse[0], torch.unsqueeze(pred_ir[0][i],0)], dim=0)
    return pred_fuse

def Dec_Fusion_Multispectral(img_ori,img_ir_ori,pred,pred_ir,iou_thres,contain_thres):
    if len(pred[0]) == 0:
        pred_fuse = copy.deepcopy(pred_ir)
    elif len(pred_ir[0]) == 0:
        pred_fuse = copy.deepcopy(pred)
    else:
        pred_fuse = copy.deepcopy(pred)
        # for i in range(len(pred_fuse[0])):
        #     pred_fuse[0][i][5] = 1

        fuse_iou = jaccard(pred_ir[0][:, :4], pred[0][:, :4])
        contain_iou_1 = contain(pred_ir[0][:, :4], pred[0][:, :4])
        contain_iou_2 = contain(pred[0][:, :4], pred_ir[0][:, :4])
        contain_iou = torch.max(contain_iou_1, contain_iou_2.t())
        m, n = fuse_iou.shape
        for i in range(m):
            add = True
            temp4 = 0
            for j in range(n):
                if fuse_iou[i][j] > iou_thres or contain_iou[i][j] > contain_thres:
                    print('Fuse_Done_Better')
                    add = False
                    if (pred_ir[0][i][4] > pred[0][j][4]) and (fuse_iou[i][j] > temp4):
                        temp4 = fuse_iou[i][j]
                        pred_fuse[0][j][4:6] = pred_ir[0][i][4:6]
                        pred_fuse[0][j][5] = 0
                    else:
                        pred_fuse[0][j][5] = 0

            # if add == False:
            #     pred_ir[0]=torch.cat([pred_ir[0][0:i], pred_ir[0][i+1:]], dim = 0)
            if add is True:
                print('Fuse_Done_Add')
                # temp2 = torch.cat([pred_fuse[0], pred_ir[0]], dim=0)
                # temp1.append(temp2)
                # pred_fuse=temp1
                temp5 = copy.deepcopy(pred_ir[0][i])
                temp5[5] = 0
                pred_fuse[0] = torch.cat([pred_fuse[0], torch.unsqueeze(temp5, 0)], dim=0)
    return pred_fuse

def bayes(p1,p2):
    return (p1*p2)/(p1*p2 + (1-p1)*(1-p2))

def Dec_Fusion_wcndt(img_ori,img_ir_ori,pred,pred_ir,iou_thres,contain_thres):
    if len(pred[0]) == 0:
        pred_fuse = copy.deepcopy(pred_ir)
    elif len(pred_ir[0]) == 0:
        pred_fuse = copy.deepcopy(pred)
    else:
        pred_fuse = copy.deepcopy(pred)

        fuse_iou = jaccard(pred_ir[0][:, :4], pred[0][:, :4])
        # contain_iou_1 = contain(pred_ir[0][:, :4], pred[0][:, :4])
        # contain_iou_2 = contain(pred[0][:, :4], pred_ir[0][:, :4])
        # contain_iou = torch.max(contain_iou_1, contain_iou_2.t())
        m, n = fuse_iou.shape
        for i in range(m):
            add = True
            temp4 = 0
            for j in range(n):
                # if fuse_iou[i][j] > iou_thres or contain_iou[i][j] > contain_thres:
                if fuse_iou[i][j] > iou_thres:
                    print('Fuse_Done_Better')
                    add = False
                    if fuse_iou[i][j] > temp4:
                        # if pred_ir[0][i][5] == pred[0][j][5]:
                        #     pred_fuse[0][j][4] = bayes(pred_ir[0][i][4], pred[0][j][4])
                        # else:
                        #     if pred_ir[0][i][4] > pred[0][j][4]:
                        #         pred_fuse[0][j][5] = pred_ir[0][i][5]
                        #         pred_fuse[0][j][4] = bayes(pred_ir[0][i][4], pred[0][j][4])
                        temp4 = fuse_iou[i][j]
                        if pred_ir[0][i][5] != pred[0][j][5]:
                            if pred_ir[0][i][4] > pred[0][j][4]:
                                pred_fuse[0][j][5] = pred_ir[0][i][5]
                            else:
                                pred_fuse[0][j][5] = pred[0][j][5]
                        pred_fuse[0][j][4] = bayes(pred_ir[0][i][4], pred[0][j][4])
                        # pred_fuse[0][j][4] = (pred_ir[0][i][4] + pred[0][j][4])/2
                        # if pred_ir[0][i][4] >= pred[0][j][4]:
                        #     pred_fuse[0][j][4] = pred_ir[0][i][4]
                        # else:
                        #     pred_fuse[0][j][4] = pred[0][j][4]




                    #
                    #
                    # if pred_ir[0][i][5] == pred[0][j][5]:
                    #     pred_fuse[0][j][4] = bayes(pred_ir[0][i][5],pred[0][j][5])
                    #
                    #
                    # if (pred_ir[0][i][4] > pred[0][j][4]) and (fuse_iou[i][j] > temp4):
                    #     temp4 = fuse_iou[i][j]
                    #     pred_fuse[0][j][4:6] = pred_ir[0][i][4:6]
                    #     pred_fuse[0][j][5] = 0
                    # else:
                    #     pred_fuse[0][j][5] = 0

            # if add == False:
            #     pred_ir[0]=torch.cat([pred_ir[0][0:i], pred_ir[0][i+1:]], dim = 0)
            if add is True:
                print('Fuse_Done_Add')
                # temp2 = torch.cat([pred_fuse[0], pred_ir[0]], dim=0)
                # temp1.append(temp2)
                # pred_fuse=temp1
                temp5 = copy.deepcopy(pred_ir[0][i])
                # temp5[5] = 0
                pred_fuse[0] = torch.cat([pred_fuse[0], torch.unsqueeze(temp5, 0)], dim=0)
    return pred_fuse

def box2windows(opt, x1, x2, y1, y2, window_h_max, window_w_max, stride):
    # 0: generate window from n times of the box
    # 1: 100 * 100 area with box center point as reference
    method = 2
    sample_name = os.path.split(opt.main_dir)[-1]

    window_h_start = 0
    window_h_end = 0
    window_w_start = 0
    window_w_end = 0
    if method == 0:
        times = 5
        window_h_start = int(max((((times + 1)/2) * y1 - ((times - 1)/2) * y2), 0))
        window_h_end = int(min((((times + 1)/2) * y2 - ((times - 1)/2) * y1), window_h_max))
        window_w_start = int(max((((times + 1)/2) * x1 - ((times - 1)/2) * x2), 0))
        window_w_end = int(min((((times + 1)/2) * x2 - ((times - 1)/2) * x1), window_w_max))

    if method == 1:
        wh = 80
        if wh % stride == 0:
            center_x = (x1 + x2)/2
            center_y = (y1 + y2)/2
            window_h_start = int(max(center_y - wh/2, 0))
            window_h_end = int(min(center_y + wh/2, window_h_max))
            window_w_start = int(max(center_x - wh/2, 0))
            window_w_end = int(min(center_x + wh / 2, window_w_max))

        else:
            print('window setup error')

    if method == 2:
        larger_range_sample_list =['07']
        if sample_name in larger_range_sample_list:
            num = 20
        else:
            num = 15        #15
        window_h_start = int(max(y1 - num * stride, 0))
        window_h_end = int(min(y2 + num * stride, window_h_max))
        window_w_start = int(max(x1 - num * stride, 0))
        window_w_end = int(min(x2 + num * stride, window_w_max))
        oriw_loc = num
        orih_loc = num

    if method == 3:
        w_shift = 5
        h_shift = 50
        if w_shift % stride == 0 and h_shift % stride == 0:
            window_h_start = int(max(y1 - h_shift, 0))
            window_h_end = int(min(y2 , window_h_max))
            window_w_start = int(max(x1 - w_shift, 0))
            window_w_end = int(min(x2 + w_shift, window_w_max))
            oriw_loc = int(((window_w_end - window_w_start - x2 + x1)/stride)/2)
            orih_loc = int(((window_h_end - window_h_start - y2 + y1)/stride)/2)
        else:
            print('window setup error')


    return window_h_start, window_h_end, window_w_start, window_w_end, oriw_loc, orih_loc

def ssim_caculation(x,y):
    x = torch.from_numpy(x).float().unsqueeze(0).unsqueeze(0)
    y = torch.from_numpy(y).float().unsqueeze(0).unsqueeze(0)
    ret = SSIM.ssim(x, y)
    return ret

def ssim_struct_caculation(x,y):
    x = torch.from_numpy(x).float().unsqueeze(0).unsqueeze(0)
    y = torch.from_numpy(y).float().unsqueeze(0).unsqueeze(0)
    ret = SSIM.ssim_struct(x, y)
    return ret

def msssim_caculation(x,y):
    x = torch.from_numpy(x).float().unsqueeze(0).unsqueeze(0)
    y = torch.from_numpy(y).float().unsqueeze(0).unsqueeze(0)
    ret = SSIM.msssim(x, y)
    return ret

def img_prepro(opt, reference_img, sensed_img, img_name):       #sensed:ir
    # 0: without prepro
    method = 3             # 3

    if method == 0:
        reference_pro = reference_img
        sensed_pro = sensed_img

    if method == 1:
        reference_pro = reference_img
        sensed_pro = 255 - sensed_img

    if method == 2:
        ref_img = copy.deepcopy(reference_img)
        sen_img = 255 - copy.deepcopy(sensed_img)
        ref_img = Image.fromarray(cv2.cvtColor(ref_img, cv2.COLOR_GRAY2RGB))
        sen_img = Image.fromarray(cv2.cvtColor(sen_img, cv2.COLOR_GRAY2RGB))

        reference_pro = reference_img
        sensed_pro = fda.FDA_source_to_target_np(sen_img, ref_img, L=0.05)
        sensed_pro = sensed_pro[:,:,0]
        sensed_pro = ((sensed_pro - sensed_pro.min()) / (sensed_pro.max() - sensed_pro.min())) * 255
        sensed_pro = sensed_pro.astype(np.uint8)


        # sensed_pro = sensed_pro.astype(np.uint8)

    if method == 3:
        ref_img = copy.deepcopy(reference_img)
        sen_img = 255 - copy.deepcopy(sensed_img)
        ref_img = Image.fromarray(cv2.cvtColor(ref_img, cv2.COLOR_GRAY2RGB))
        sen_img = Image.fromarray(cv2.cvtColor(sen_img, cv2.COLOR_GRAY2RGB))

        # sensed_pro = fda.FDA_source_to_target_np(sen_img, ref_img, L=0.05)

        reference_pro = fda.FDA_source_to_target_np(ref_img, sen_img, L=0.01)

        sensed_pro = 255 - sensed_img
        reference_pro = reference_pro[:, :, 0]

        for i in range(reference_pro.shape[0]):
            for j in range(reference_pro.shape[1]):
                if reference_pro[i][j] > 255:
                    reference_pro[i][j] = 255

        reference_pro = reference_pro.astype(np.uint8)
    if method == 7:
        ref_img = copy.deepcopy(reference_img)
        sen_img = copy.deepcopy(sensed_img)
        ref_img = Image.fromarray(cv2.cvtColor(ref_img, cv2.COLOR_GRAY2RGB))
        sen_img = Image.fromarray(cv2.cvtColor(sen_img, cv2.COLOR_GRAY2RGB))

        # sensed_pro = fda.FDA_source_to_target_np(sen_img, ref_img, L=0.05)

        reference_pro = fda.FDA_source_to_target_np(ref_img, sen_img, L=0.01)

        sensed_pro = sensed_img
        reference_pro = reference_pro[:, :, 0]

        for i in range(reference_pro.shape[0]):
            for j in range(reference_pro.shape[1]):
                if reference_pro[i][j] > 255:
                    reference_pro[i][j] = 255

        reference_pro = reference_pro.astype(np.uint8)
        # sensed_pro = cv2.cvtColor(np.asarray(sensed_pro), cv2.COLOR_RGB2GRAY)

    if method == 6:
        ref_img = copy.deepcopy(reference_img)
        sen_img = 255 - copy.deepcopy(sensed_img)

        r = 2
        esp = 0.01

        ref_img = guidefilter.guidedFilter(cv2.cvtColor(copy.deepcopy(ref_img), cv2.COLOR_GRAY2RGB), cv2.cvtColor(copy.deepcopy(ref_img), cv2.COLOR_GRAY2RGB), r, esp)[:, :, 0]
        sen_img = guidefilter.guidedFilter(cv2.cvtColor(copy.deepcopy(sen_img), cv2.COLOR_GRAY2RGB), cv2.cvtColor(copy.deepcopy(sen_img), cv2.COLOR_GRAY2RGB), r, esp)[:, :, 0]

        ref_img = Image.fromarray(cv2.cvtColor(ref_img, cv2.COLOR_GRAY2RGB))
        sen_img = Image.fromarray(cv2.cvtColor(sen_img, cv2.COLOR_GRAY2RGB))

        # sensed_pro = fda.FDA_source_to_target_np(sen_img, ref_img, L=0.05)

        reference_pro = fda.FDA_source_to_target_np(ref_img, sen_img, L=0.01)

        sensed_pro = 255 - sensed_img
        reference_pro = reference_pro[:, :, 0]

        for i in range(reference_pro.shape[0]):
            for j in range(reference_pro.shape[1]):
                if reference_pro[i][j] > 255:
                    reference_pro[i][j] = 255

        reference_pro = reference_pro.astype(np.uint8)


    if method == 4:             #test
        ref_img = copy.deepcopy(reference_img)
        sen_img = 255 - copy.deepcopy(sensed_img)
        ref_img = Image.fromarray(cv2.cvtColor(ref_img, cv2.COLOR_GRAY2RGB))
        sen_img = Image.fromarray(cv2.cvtColor(sen_img, cv2.COLOR_GRAY2RGB))

        # sensed_pro = fda.FDA_source_to_target_np(sen_img, ref_img, L=0.05)

        reference_fda = fda.FDA_source_to_target_np(ref_img, sen_img, L=0.01)


        # sensed_pro = 255 - sensed_img
        reference_pro = reference_fda[:, :, 0]
        reference_fda_image = Image.fromarray(reference_fda.astype(np.uint8))
        sensed_pro = fda.FDA_source_to_target_np(sen_img, reference_fda_image, L=0.01)
        sensed_pro = sensed_pro[:, :, 0]

        reference_pro = reference_pro.astype(np.uint8)
        sensed_pro = sensed_pro.astype(np.uint8)

    if method == 5:             #test:guide filter
        ref_img = copy.deepcopy(reference_img)
        sen_img = 255 - copy.deepcopy(sensed_img)
        ref_img = Image.fromarray(cv2.cvtColor(ref_img, cv2.COLOR_GRAY2RGB))
        sen_img = Image.fromarray(cv2.cvtColor(sen_img, cv2.COLOR_GRAY2RGB))

        # sensed_pro = fda.FDA_source_to_target_np(sen_img, ref_img, L=0.05)

        reference_fda = fda.FDA_source_to_target_np(ref_img, sen_img, L=0.01)

        sensed_fda = cv2.cvtColor(255 - copy.deepcopy(sensed_img), cv2.COLOR_GRAY2RGB)

        # reference_pro = reference_pro.astype(np.uint8)
        r = 2
        esp = 0.01

        reference_pro = guidefilter.guidedFilter(reference_fda, reference_fda, r, esp)[:, :, 0]
        sensed_pro = guidefilter.guidedFilter(sensed_fda, sensed_fda, r, esp)[:, :, 0]

        if opt.view_checkpoint:
            outdir = os.path.join(opt.view_output, img_name)
            os.makedirs(outdir, exist_ok=True)
            cv2.imwrite(os.path.join(outdir, '0000_sense_fda.jpg'), sensed_fda)
            cv2.imwrite(os.path.join(outdir, '0000_reference_fda.jpg'), reference_fda)

        # sensed_pro = cv2.cvtColor(np.asarray(sensed_pro), cv2.COLOR_RGB2GRAY)

    if opt.view_checkpoint:
        outdir = os.path.join(opt.view_output, img_name)
        os.makedirs(outdir, exist_ok=True)
        cv2.imwrite(os.path.join(outdir, '0000_sense_image.jpg'), sensed_img)
        cv2.imwrite(os.path.join(outdir, '0000_reference_image.jpg'), reference_img)
        cv2.imwrite(os.path.join(outdir, '0000_sense_image_pro.jpg'), sensed_pro)
        cv2.imwrite(os.path.join(outdir, '0000_reference_image_pro.jpg'), reference_pro)

    return reference_pro, sensed_pro




def calculate_similarity(opt, x, y):
    # 0:calculate MI
    # 1:calculate corr
    # 2:calculate
    # 5:matchnet
    method = opt.match_method

    if method == 0:
        x = np.reshape(x, -1)
        y = np.reshape(y, -1)
        value = skm.mutual_info_score(x, y)

    if method == 1:
        value = cv2.matchTemplate(x, y, method = cv2.TM_CCORR)
    if method == 2:
        x = (x - x.min())/(x.max() - x.min())*255
        x = x.astype(np.uint8)
        y = (y - y.min())/(y.max() - y.min())*255
        y = y.astype(np.uint8)
        value = cv2.matchTemplate(x, y, method=cv2.TM_CCORR_NORMED)
    if method == 3:                                                     #50000
        value = cv2.matchTemplate(x, y, method=cv2.TM_CCOEFF)
    if method == 4:
        value = cv2.matchTemplate(x, y, method=cv2.TM_CCOEFF_NORMED)
    if method == 5:
        matchnetdd = MatchnetDD(opt)
        value = matchnetdd.matchnetDD_similar(x, y)
    if method == 6:
        x = (x - x.min())/(x.max() - x.min())*255
        x = x.astype(np.uint8)
        y = (y - y.min())/(y.max() - y.min())*255
        y = y.astype(np.uint8)
        # value = ssim_caculation(x, y)
        value = ssim_struct_caculation(x, y)
    if method == 7:
        value = msssim_caculation(x, y)
    if method == 8:
        x = (x - x.min()) / (x.max() - x.min()) * 255
        x = x.astype(np.uint8)
        y = (y - y.min()) / (y.max() - y.min()) * 255
        y = y.astype(np.uint8)
        value = ssim_caculation(x, y)

    return value

def calculate_best_match(num, opt, x1, x2, y1, y2, window_h_start, window_h_end, window_w_start, window_w_end, stride, oriw_loc, orih_loc, sensed_image, reference_image, img_name, sensed_ori, reference_ori):
    # 0: calculate best match from max_MI
    # 1: calculate best match from gaussian attenuation
    # 2: calculate best match from distribution of MI

    # sensed_img is vis

    conf = 0.3  #methods4:0.32        methods3:50000    methods5:0.5    methods6:0.10

    comman_conf = 5.5

    if opt.match_method == 6:
        method = 2
        if opt.ablation_study == 'ours':
            sample_name = os.path.split(opt.main_dir)[-1]
            sample_list = sorted(['01', '02', '03', '04', '05', '06', '07'])
            logvar_conf_list = [5, 5, 6.5, 6, 4, 5, 5.5]
            logvar_conf_dict = dict(zip(sample_list, logvar_conf_list))
            if sample_name in sample_list:
                logvar_conf = logvar_conf_dict[sample_name]
            else:
                logvar_conf = comman_conf  # ours:5.5    04 MI:3.5
        else:
            logvar_conf = comman_conf
    elif opt.match_method == 8:
        logvar_conf = comman_conf
        method = 2
    else:
        logvar_conf = comman_conf
        method = 0

    box_height = y2 - y1
    box_width = x2 - x1
    h = window_h_end - window_h_start
    w = window_w_end - window_w_start

    if opt.match_method == 9:
        template = copy.deepcopy(sensed_image[y1:y2, x1:x2])
        gray_img = copy.deepcopy(reference_image[window_h_start:window_h_end, window_w_start:window_w_end])
        mean, stddev = cv2.meanStdDev(gray_img)
        normalized_img = (gray_img - mean) / stddev
        mean, stddev = cv2.meanStdDev(template)
        normalized_template = (template - mean) / stddev
        integral_img = cv2.integral(normalized_img)
        template_integral_img = cv2.integral(normalized_template)

    result = np.zeros([int((h - box_height) / stride), int((w - box_width) / stride)])
    result_rev = np.zeros([int((h - box_height) / stride), int((w - box_width) / stride)])
    x = sensed_image[y1:y2, x1:x2]
    # x_ori = sensed_ori[y1:y2, x1:x2]
    for i in range(int((h - box_height) / stride)):
        for j in range(int((w - box_width) / stride)):
            reference_h_start = window_h_start + i * stride
            reference_h_end = window_h_start + i * stride + box_height
            reference_w_start = window_w_start + j * stride
            reference_w_end = window_w_start + j * stride + box_width
            y = reference_image[reference_h_start:reference_h_end, reference_w_start:reference_w_end]
            if opt.match_method == 9:
                template_height, template_width = template.shape
                window_sum = integral_img[i + template_height, j + template_width] + integral_img[i, j] \
                             - integral_img[i + template_height, j] - integral_img[i, j + template_width]
                window_mean = window_sum / (template_height * template_width)
                template_sum = template_integral_img[template_height, template_width] + template_integral_img[0, 0] \
                               - template_integral_img[0, template_width] - template_integral_img[template_height, 0]
                template_mean = template_sum / (template_height * template_width)
                window_stddev = np.sqrt((np.sum(np.square(normalized_img[i:i + template_height, j:j + template_width])) \
                                         + np.sum(np.square(normalized_template - template_mean))) / (template_height * template_width))
                ncc = np.sum((normalized_img[i:i + template_height, j:j + template_width] - window_mean) * (
                            normalized_template - template_mean)) / (window_stddev * stddev)
                result[i, j] = abs(ncc)

            else:
                result[i, j] = calculate_similarity(opt, x, y)

                if opt.view_checkpoint:
                    outdir = os.path.join(opt.view_output, img_name)
                    os.makedirs(outdir, exist_ok=True)
                    if result[i, j] > 0:         #conf
                        name = num + '_{}_{}_{:.4f}.jpg'.format(j, i, result[i, j])
                        outpath = os.path.join(outdir, name)
                        cv2.imwrite(outpath, y)

                result_rev[i, j] = calculate_similarity(opt, 255 - x, y)
                # result_rev[i, j] = calculate_similarity(opt, x_ori, y)

                if opt.view_checkpoint:
                    outdir = os.path.join(opt.view_output, img_name)
                    os.makedirs(outdir, exist_ok=True)
                    if result_rev[i, j] > 0:         #conf
                        name = num + '_{}_{}_{:.4f}_rev.jpg'.format(j, i, result_rev[i, j])
                        outpath = os.path.join(outdir, name)
                        cv2.imwrite(outpath, y)
    if opt.ablation_study == 'ours':
        result = np.maximum(result,result_rev)

    # result_tensor = torch.tensor(result)
    # result_tensor = result_tensor.unsqueeze(0).unsqueeze(0)
    # kernel = torch.tensor([[0, 1, 0],
    #                        [1, 1, 1],
    #                        [0, 1, 0]], dtype=torch.float64)
    # output = F.conv2d(result_tensor, kernel.unsqueeze(0).unsqueeze(0), padding=1)
    # result = output.numpy().squeeze()
    # result = result/5



    # result = (result - np.min(result))/(np.max(result)-np.min(result))

    if method == 0:

        if(np.max(result) < conf):
            max_location = np.where(result == np.max(result))
            max_location[0][0] = 0
            max_location[1][0] = 0
            match_h_start = y1
            match_h_end = y2
            match_w_start = x1
            match_w_end = x2
        else:
            max_location = np.where(result == np.max(result))
            match_h_start = window_h_start + max_location[0][0] * stride
            match_h_end = window_h_start + max_location[0][0] * stride + box_height
            match_w_start = window_w_start + max_location[1][0] * stride
            match_w_end = window_w_start + max_location[1][0] * stride + box_width
        if opt.view_checkpoint:
            outdir = os.path.join(opt.view_output, img_name)
            cv2.imwrite(os.path.join(outdir, num + '_match_similarity.jpg'),
                        ((result - result.min()) / (result.max() - result.min())) * 255)

    if method == 2:
        similarity_var = result.var()
        opt.var.append(similarity_var)
        # print(opt.var)


        if(np.max(result) < conf or -np.log(similarity_var) > logvar_conf):
            max_location = np.where(result == np.max(result))
            max_location[0][0] = 0
            max_location[1][0] = 0
            match_h_start = y1
            match_h_end = y2
            match_w_start = x1
            match_w_end = x2
        else:
            max_location = np.where(result == np.max(result))
            match_h_start = window_h_start + max_location[0][0] * stride
            match_h_end = window_h_start + max_location[0][0] * stride + box_height
            match_w_start = window_w_start + max_location[1][0] * stride
            match_w_end = window_w_start + max_location[1][0] * stride + box_width
        if opt.view_checkpoint:
            print('similarity_logvar = {}'.format(-np.log(similarity_var)))
            cv2.imwrite(os.path.join(outdir, num + '_match_similarity_var{:.5f}.jpg'.format(-np.log(similarity_var))),
                        ((result - result.min()) / (result.max() - result.min())) * 255)


    if method == 1:
        kernel_size = max(int((h - box_height) / stride) ,int((w - box_width) / stride))
        sigma = 10
        kx = cv2.getGaussianKernel(kernel_size, sigma)
        ky = cv2.getGaussianKernel(kernel_size, sigma)
        gaussian_kernel = np.multiply(kx, np.transpose(ky))
        if (kernel_size == int((h - box_height) / stride)):
            weight = gaussian_kernel[:, int((h - box_height - w + box_width) / (2*stride)):int((h - box_height - w + box_width) / (2*stride)) + int((w - box_width) / stride)]
        elif (kernel_size == int((w - box_width) / stride)):
            weight = gaussian_kernel[int((w - box_width - h + box_height) / (2*stride)):int((w - box_width - h + box_height) / (2*stride)) + int((h - box_height) / stride),:]
        weight = weight / weight[int(weight.shape[0] / 2),int(weight.shape[1] / 2)]
        result = result * weight
        max_location = np.where(result == np.max(result))
        if opt.view_checkpoint:
            cv2.imwrite(os.path.join(outdir, num + '_match_similarity.jpg'),
                        ((result - result.min()) / (result.max() - result.min())) * 255)

    if opt.view_checkpoint:
        print('MI_max = {}'.format(np.max(result)))
        outdir = os.path.join(opt.view_output, img_name)
        cv2.imwrite(os.path.join(outdir, num + '_sense_image.jpg'), x)
        cv2.imwrite(os.path.join(outdir, num + '_sense_image_rev.jpg'), 255 - x)
        cv2.imwrite(os.path.join(outdir, num + '_window.jpg'), reference_image[window_h_start:window_h_end, window_w_start:window_w_end])
        cv2.imwrite(os.path.join(outdir, num + '_bestmatch_{}_{}_{:.5f}.jpg'.format(max_location[1][0], max_location[0][0], result.max())),
            reference_image[match_h_start:match_h_end, match_w_start:match_w_end])
        # print('write sucsses')


    return max_location, match_h_start, match_h_end, match_w_start, match_w_end

def MI_match(opt, sensed_pred,reference_image,sensed_image, img_name, reference_ori, sensed_ori):
    sensed_pred_match = copy.deepcopy(sensed_pred)
    for k in range(len(sensed_pred[0])):

        x1 = int(sensed_pred[0][k][0].item())
        x2 = int(sensed_pred[0][k][2].item())
        y1 = int(sensed_pred[0][k][1].item())
        y2 = int(sensed_pred[0][k][3].item())
        box_height = y2 - y1
        box_width = x2 - x1

        if opt.enlarge_box:
            if os.path.split(opt.main_dir)[-1] == '07':
                if int(img_name) >= 514 and int(img_name) <= 522:
                    if box_width < 25 and box_height < 25:
                        enlarge_time = 0.3
                    else:
                        enlarge_time = 0.0
                else:
                    enlarge_time = 0.0
            else:
                enlarge_time = 0.0
                # if box_width < 25 and box_height < 25:
                #     enlarge_time = 0.2
                # elif box_height/box_width > 2.5 or box_width/box_height > 2.5:
                #     enlarge_time = 0.2
                # else:
                #     enlarge_time = 0.0
        else:
            enlarge_time = 0.0


        box_h_max, box_w_max = reference_image.shape
        box_height_enlarge = int(box_height * enlarge_time)
        box_width_enlarge = int(box_width * enlarge_time)
        x1_change = x1 - max(0, x1 - box_width_enlarge)
        x2_change = x2 - min(x2 + box_width_enlarge, box_w_max)
        y1_change = y1 - max(0, y1 - box_height_enlarge)
        y2_change = y2 - min(y2 + box_height_enlarge, box_h_max)

        x1 = x1 - copy.deepcopy(x1_change)
        x2 = x2 - copy.deepcopy(x2_change)
        y1 = y1 - copy.deepcopy(y1_change)
        y2 = y2 - copy.deepcopy(y2_change)

        stride = 1
        window_h_max,window_w_max = reference_image.shape


        window_h_start,window_h_end,window_w_start,window_w_end, oriw_loc, orih_loc = box2windows(opt, x1,x2,y1,y2,window_h_max,window_w_max,stride)

        max_location, match_h_start, match_h_end, match_w_start, match_w_end = calculate_best_match(str(k), opt, x1,x2,y1,y2,window_h_start,window_h_end,window_w_start,window_w_end,stride,oriw_loc,orih_loc,sensed_image,reference_image, img_name, sensed_ori, reference_ori)


        sensed_pred_match[0][k][1] = match_h_start + y1_change
        sensed_pred_match[0][k][0] = match_w_start + x1_change
        sensed_pred_match[0][k][3] = match_h_end + y2_change
        sensed_pred_match[0][k][2] = match_w_end + x2_change

    return sensed_pred_match

def pred_NMS(pred1,pred2,iou_thres):        #pred1 is sensed(ir)
    pred_result = copy.deepcopy(pred2)
    fuse_iou = jaccard(pred1[0][:, :4], pred2[0][:, :4])
    m, n = fuse_iou.shape
    for i in range(m):
        add = True
        temp4 = 0
        for j in range(n):
            if (fuse_iou[i][j] > iou_thres):
                add = False
                if ((pred1[0][i][4] > pred2[0][j][4]) and (fuse_iou[i][j] > temp4)):
                    temp4 = fuse_iou[i][j]
                    pred_result[0][j][4:6] = pred1[0][i][4:6]
        if add == True:
            print('Fuse_Done_Add')
            temp5 = copy.deepcopy(pred1[0][i])
            # temp5[5] = 1
            pred_result[0] = torch.cat([pred_result[0], torch.unsqueeze(temp5, 0)], dim=0)

    return pred_result

def box_area(box):
    return (box[2]-box[0]) * (box[3]-box[1])

def pred_NMSC(pred, contain_thres, class_num):        #pred1 is sensed(ir)
    class_pred = [torch.zeros((0, 6), device=pred[0].device)] * class_num
    result = [torch.zeros((0, 6), device=pred[0].device)]
    for i in range(len(pred[0])):
        class_pred[int(pred[0][i][5])] = torch.cat((class_pred[int(pred[0][i][5])],torch.unsqueeze(pred[0][i],0)),0)

    for i in range(len(class_pred)):
        if len(class_pred[i]) >= 2:
            flag = True
            while(flag):
                last_pred = copy.deepcopy(class_pred[i])
                contain_iou = contain(class_pred[i][:, :4], class_pred[i][:, :4])
                for j in range(len(contain_iou)):
                    contain_iou[j][j] = 0
                    contain_numpy = contain_iou[j].cpu().numpy()
                    if np.max(contain_numpy) > contain_thres:
                        max_location = np.where(contain_numpy == np.max(contain_numpy))
                        # if(class_pred[i][j][4] > class_pred[i][max_location[0][0]][4]):
                        #     class_pred[i] = class_pred[i][torch.arange(class_pred[i].size(0))!=max_location[0][0]]
                        #     break
                        # else:
                        #     class_pred[i] = class_pred[i][torch.arange(class_pred[i].size(0)) != 0]
                        #     break
                        if (box_area(class_pred[i][j]) > box_area(class_pred[i][max_location[0][0]])):
                            class_pred_cls_max = torch.max(class_pred[i][:,4])
                            for k in range(len(class_pred[i])):
                                class_pred[i][k][4] = class_pred_cls_max
                            class_pred[i] = class_pred[i][torch.arange(class_pred[i].size(0))!=max_location[0][0]]
                            break
                        else:
                            class_pred[i] = class_pred[i][torch.arange(class_pred[i].size(0)) != j]
                            break
                if(last_pred.equal(class_pred[i])):
                    break
    for i in range(len(class_pred)):
        result[0] = torch.cat((result[0], class_pred[i]), 0)

    return result

def Dec_Fusion_Object_Detection_Withmatch(opt, img,img_ir,pred,pred_ir,iou_thres, contain_thres, class_num):
    if len(pred_ir[0]) == 0:
        pred_fuse = copy.deepcopy(pred)
        pred_fuse = pred_NMSC(pred_fuse, contain_thres, class_num)
    else:
        # pred_fuse = copy.deepcopy(pred)
        img = img.cpu().numpy()[0,:, :, :]*255
        img = np.transpose(img, (1, 2, 0))
        img = img.astype(np.uint8)
        r, g, b = cv2.split(img)
        img = cv2.merge([b, g, r])
        VIS_grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        img_ir = img_ir.cpu().numpy()[0,:,:,:]*255
        img_ir = np.transpose(img_ir, (1, 2, 0))
        img_ir = img_ir.astype(np.uint8)
        r, g, b = cv2.split(img_ir)
        img_ir = cv2.merge([b, g, r])
        IR_grey = cv2.cvtColor(img_ir, cv2.COLOR_BGR2GRAY)


        pred_ir_match = MI_match(opt, pred_ir,VIS_grey, IR_grey)

        # if len(pred[0]) == 0:
        #     pred_fuse = pred_ir_match
        # else:
        pred_fuse = pred_NMS(pred_ir_match, pred, iou_thres)

        pred_fuse = pred_NMSC(pred_fuse, contain_thres, class_num)

    # for i in range(len(pred_fuse[0])):
    #     if(pred_fuse[0][i][5] == 1):
    #         pred_fuse[0][i][5] = 2
    #     elif(pred_fuse[0][i][5] == 2):
    #         pred_fuse[0][i][5] = 1
        # pred_fuse = copy.deepcopy(pred_ir_match)

    return pred_fuse

def Dec_Fusion_Multispectral_Withmatch(opt, img,img_ir,pred,pred_ir,iou_thres,contain_thres, img_name):
    print('/n')
    if len(pred[0]) == 0:
        pred_fuse = copy.deepcopy(pred_ir)
        # for i in range(len(pred_ir[0])):
        #     pred_fuse[0][i][5] = 1

    else:
        img = img.cpu().numpy()[0,:, :, :]*255
        img = np.transpose(img, (1, 2, 0))
        img = img.astype(np.uint8)
        r, g, b = cv2.split(img)
        img = cv2.merge([b, g, r])
        VIS_grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        img_ir = img_ir.cpu().numpy()[0,:,:,:]*255
        img_ir = np.transpose(img_ir, (1, 2, 0))
        img_ir = img_ir.astype(np.uint8)
        r, g, b = cv2.split(img_ir)
        img_ir = cv2.merge([b, g, r])
        IR_grey = cv2.cvtColor(img_ir, cv2.COLOR_BGR2GRAY)
        #
        if opt.match_method == 20:
            VIS_pro, IR_pro = img_prepro(opt, VIS_grey, IR_grey, img_name)

            VIS_dim = np.stack((VIS_pro,) * 3, axis=-1)
            IR_dim = np.stack((IR_pro,) * 3, axis=-1)

            pred_match = deepDIM.deepDIM_match(opt, pred, IR_dim,VIS_dim, img_name, IR_grey, VIS_grey)


            # IR_pro, VIS_pr o = img_prepro(opt, IR_grey, VIS_grey, img_name)
            #
            # pred_match = MI_match(opt, pred, VIS_pro, IR_pro, img_name)

            pred_fuse = Dec_Fusion_Multispectral(img, img_ir, pred_ir, pred_match, iou_thres, contain_thres)

        elif opt.ablation_study == 'ours':
            VIS_pro, IR_pro = img_prepro(opt, VIS_grey, IR_grey, img_name)

            pred_match = MI_match(opt, pred, IR_pro, VIS_pro, img_name, IR_grey, VIS_grey)


            # IR_pro, VIS_pro = img_prepro(opt, IR_grey, VIS_grey, img_name)
            #
            # pred_match = MI_match(opt, pred, VIS_pro, IR_pro, img_name)

            pred_fuse = Dec_Fusion_Multispectral(img, img_ir, pred_ir, pred_match, iou_thres, contain_thres)
        elif opt.ablation_study == 'CR_BR':
            pred_match = MI_match(opt, pred, IR_grey, VIS_grey, img_name, IR_grey, VIS_grey)
            pred_fuse = Dec_Fusion_Multispectral(img, img_ir, pred_ir, pred_match, iou_thres, contain_thres)
        elif opt.ablation_study == 'CR':
            pred_fuse = Dec_Fusion_Multispectral(img, img_ir, pred_ir, pred, iou_thres, contain_thres)

    return pred_fuse