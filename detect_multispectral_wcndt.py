# -*- coding:utf-8 -*-
"""
Author: LJC
Email: jiacheng_li@std.uestc.edu.cn
name: detect_dual.py
Data: 2021.07.26
Note: Improved by Yolov5_detect
"""

import argparse
import copy
import time
from pathlib import Path

import cv2
import torch
from torch.nn import functional as F
import os
import torch.backends.cudnn as cudnn
from numpy import random

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized
import dec_fusion_multispectral_wcndt as Fu
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
import shutil

def del_file(path):
    ls = os.listdir(path)
    for i in ls:
        c_path = os.path.join(path, i)
        if os.path.isdir(c_path):
            del_file(c_path)
        else:
            os.remove(c_path)

def image_shift(input,sx,sy):   #input is torch and float   outputshow:plt.imshow(test.numpy().astype(np.uint8).transpose(1,2,0))    sx(- means down(right),+ means up(left))
    theta = torch.tensor([
        [1, 0, sx],
        [0, 1, sy]
    ], dtype=torch.float)
    grid = F.affine_grid(theta.unsqueeze(0), input.unsqueeze(0).size())
    output = F.grid_sample(input.unsqueeze(0), grid)
    return output[0]

def detect(save_img=False):
    save_dir_txt, source_vis, source_ir, weights_vis, weights_ir, view_img, save_txt, imgsz, sx, sy = opt.txt_dir,opt.source_vis,opt.source_ir, opt.weights_vis,opt.weights_ir, opt.view_img, opt.save_txt, opt.img_size, opt.shiftx, opt.shifty
    save_img = not opt.nosave and not source_vis.endswith('.txt')  # save inference images

    # Directories


    if opt.match_method == 6:
        if opt.ablation_study == 'ours':
            save_dir_fusion_txt = os.path.join(save_dir_txt, 'detection_fusion_ours_txt')
            opt.project = os.path.join(opt.project, 'detect_fusion_ours')
        elif opt.ablation_study == 'CR':
            save_dir_fusion_txt = os.path.join(save_dir_txt, 'detection_fusion_CR_txt')
            opt.project = os.path.join(opt.project, 'detect_fusion_CR')
        elif opt.ablation_study == 'CR_BR':
            save_dir_fusion_txt = os.path.join(save_dir_txt, 'detection_fusion_CR_BR_txt')
            opt.project = os.path.join(opt.project, 'detect_fusion_CR_BR')
    elif opt.match_method == 5:
        save_dir_fusion_txt = os.path.join(save_dir_txt, 'detection_fusion_Matchnet_txt')
        opt.project = os.path.join(opt.project, 'detect_fusion_Matchnet')
    elif opt.match_method == 0:
        save_dir_fusion_txt = os.path.join(save_dir_txt, 'detection_fusion_MI_txt')
        opt.project = os.path.join(opt.project, 'detect_fusion_MI')
    elif opt.match_method == 8:
        save_dir_fusion_txt = os.path.join(save_dir_txt, 'detection_fusion_SSIM_txt')
        opt.project = os.path.join(opt.project, 'detect_fusion_SSIM')
    elif opt.match_method == 9:
        save_dir_fusion_txt = os.path.join(save_dir_txt, 'detection_fusion_inteNCC_txt')
        opt.project = os.path.join(opt.project, 'detect_fusion_inteNCC')
    elif opt.match_method == 20:
        save_dir_fusion_txt = os.path.join(save_dir_txt, 'detection_fusion_deepDIM_txt')
        opt.project = os.path.join(opt.project, 'detect_fusion_deepDIM')
    else:
        print('match_method error')

    save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir


    save_dir_ir_txt = os.path.join(os.path.abspath(os.path.join(save_dir_fusion_txt,"..")),'detection_ir_txt')
    save_dir_vis_txt = os.path.join(os.path.abspath(os.path.join(save_dir_fusion_txt,"..")),'detection_vis_txt')
    save_dir_fusion_txt = Path(save_dir_fusion_txt)
    save_dir_ir_txt = Path(save_dir_ir_txt)
    save_dir_vis_txt = Path(save_dir_vis_txt)

    (save_dir_fusion_txt if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir
    del_file(save_dir_fusion_txt)
    (save_dir_ir_txt if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir
    del_file(save_dir_ir_txt)
    (save_dir_vis_txt if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir
    del_file(save_dir_vis_txt)


    if opt.view_checkpoint:
        # view_output_dir = Path(opt.view_output)
        # view_output_dir.mkdir(parents=True, exist_ok=True)  # make dir
        # del_file(view_output_dir)
        if os.path.exists(opt.view_output):
            shutil.rmtree(opt.view_output)
        os.makedirs(opt.view_output)

    # Initialize
    set_logging()
    device = select_device(opt.device)
    half = device.type != 'cpu'  # half precision only supported on CUDA


    # Load model
    model = attempt_load(weights_vis, map_location=device)  # load FP32 model        model_vis
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size
    if half:
        model.half()  # to FP16


    model_ir= attempt_load(weights_ir, map_location=device)  # load FP32 model        model_ir
    stride_ir = int(model_ir.stride.max())  # model stride
    imgsz_ir = check_img_size(imgsz, s=stride_ir)  # check img_size
    if half:
        model_ir.half()  # to FP16

    # Set Dataloader
    vid_path, vid_writer = None, None

    dataset = LoadImages(source_vis, img_size=imgsz, stride=stride)
    dataset_ir = LoadImages(source_ir, img_size=imgsz_ir, stride=stride_ir)

    # Get names and colors
    names = model_ir.module.names if hasattr(model_ir, 'module') else model_ir.names
    # names.append('sub_defect')



    # names.append('sub_defect')
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once

    if device.type != 'cpu':
        model_ir(torch.zeros(1, 3, imgsz_ir, imgsz_ir).to(device).type_as(next(model_ir.parameters())))  # run once
    t0 = time.time()
    fusion_time = []

    # for data, data_ir in zip(dataset, dataset_ir):

    for (path, img, im0s, vid_cap) , (path_ir, img_ir, im0s_ir, vid_cap_ir) in zip(dataset, dataset_ir):

        # img_ori = copy.deepcopy(img)
        # img_ir_ori = copy.deepcopy(img_ir)
        img_name = path.split('/')[-1].split('.')[0]

        img = torch.from_numpy(img).to(device)
        img = image_shift(img.to('cpu').float(), sx, sy)
        img = img.type(torch.uint8).to(device)

        im0st = image_shift(torch.from_numpy(im0s.transpose(2,0,1)).float(),sx,sy)
        im0s = im0st.numpy().astype(np.uint8).transpose(1,2,0)

        # plt.imshow(img.to('cpu').numpy().transpose(1, 2, 0))

        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        img_ir = torch.from_numpy(img_ir).to(device)
        img_ir = img_ir.half() if half else img_ir.float()  # uint8 to fp16/32
        img_ir /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img_ir.ndimension() == 3:
            img_ir = img_ir.unsqueeze(0)

        img_fuse = copy.deepcopy(img)
        im0s_fuse = copy.deepcopy(im0s)


        # Inference

        pred = model(img, augment=opt.augment)[0]

        # Apply NMS
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)


        for i in range(len(pred[0])):

            if pred[0][i][0] < 0:
                pred[0][i][0] = 0
            if pred[0][i][1] < 0:
                pred[0][i][1] = 0
            if pred[0][i][2] > img.shape[3]:
                pred[0][i][2] = img.shape[3]
            if pred[0][i][3] > img.shape[2]:
                pred[0][i][3] = img.shape[2]

        pred_ir = model_ir(img_ir, augment=opt.augment)[0]

        # Apply NMS
        pred_ir = non_max_suppression(pred_ir, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)

        for i in range(len(pred_ir[0])):

            if pred_ir[0][i][0] < 0:
                pred_ir[0][i][0] = 0
            if pred_ir[0][i][1] < 0:
                pred_ir[0][i][1] = 0
            if pred_ir[0][i][2] > img_ir.shape[3]:
                pred_ir[0][i][2] = img_ir.shape[3]
            if pred_ir[0][i][3] > img_ir.shape[2]:
                pred_ir[0][i][3] = img_ir.shape[2]
        t1 = time_synchronized()
        #Fuse
        # pred_fuse = Fu.Dec_Fusion_Multispectral(img, img_ir, pred, pred_ir, opt.iou_thres, opt.contain_thres)
        # pred_fuse = Fu.Dec_Fusion_Object_Detection(pred,pred_ir,opt.iou_thres)
        # pred_fuse = Fu.Dec_Fusion_Multispectral_Withmatch(opt, img, img_ir, pred, pred_ir, opt.fuseiou_thres, opt.contain_thres, img_name)
        # pred_fuse[0] = torch.cat([pred[0],pred_fuse[0]])
        # if img_name == '0000575':
        #     a=1
        pred_fuse = Fu.Dec_Fusion_wcndt(img, img_ir, pred, pred_ir, opt.fuseiou_thres, opt.contain_thres)
        # pred_fuse = []
        #
        # if (len(pred[0] != 0)):
        #     for i in range(len(pred)):
        #         pred_fuse.append(pred[i])
        # if (len(pred_ir[0] != 0)):
        #     for i in range(len(pred_ir)):
        #         pred_fuse.append(pred_ir[i])
        # pred_fuse = non_max_suppression(pred_fuse, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
        # pred_fuse = non_max_suppression(pred_fuse, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
        t2 = time_synchronized()

        image_name = img_name
        fusion_time_str = image_name + ' {:.4f}'.format(t2 - t1)
        fusion_time.append(fusion_time_str)
        # if img_name =='0000077':
        #     a=1
        # Process detections
        for (i, det) ,(i_ir, det_ir) ,(i_fuse, det_fuse) in zip(enumerate(pred), enumerate(pred_ir), enumerate(pred_fuse)):  # detections per image

            p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            p_vis_temp = p.name.split('.')
            name1 = p_vis_temp[0]+'_vis.'+p_vis_temp[1]
            save_path = str(save_dir / name1)  # img.jpg
            s += '%gx%g ' % img.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh

            p_ir, s_ir, im0_ir, frame_ir = path_ir, '', im0s_ir, getattr(dataset_ir, 'frame_ir', 0)

            p_ir = Path(p_ir)  # to Path
            p_ir_temp = p_ir.name.split('.')
            name2 = p_ir_temp[0]+'_ir.'+p_ir_temp[1]
            save_path_ir = str(save_dir / name2)  # img.jpg
            s_ir += '%gx%g ' % img_ir.shape[2:]  # print string
            gn_ir = torch.tensor(im0_ir.shape)[[1, 0, 1, 0]]  # normalization gain whwh

            p_fuse, s_fuse, im0_fuse, frame_fuse = path, '', im0s_fuse, getattr(dataset, 'frame_fuse', 0)

            p_fuse = Path(p_fuse)  # to Path

            # temp3 = p_fuse.name.split('_')    #kaist fuse result name
            # name = temp3[0]+'_'+temp3[1]+'_fuse_'+temp3[3]

            temp3 = p_fuse.name.split('.')
            name3 = temp3[0]+'_fuse.'+temp3[1]

            save_path_fuse = str(save_dir / name3)  # img.jpg
            txt_path_ir = str(save_dir_ir_txt / p_ir.stem) + (
                '' if dataset.mode == 'image' else f'_{frame_ir}')  # img.txt
            txt_path_vis = str(save_dir_vis_txt / p.stem) + (
                '' if dataset.mode == 'image' else f'_{frame}')  # img.txt
            txt_path_fuse = str(save_dir_fusion_txt / p_fuse.stem) + ('' if dataset.mode == 'image' else f'_{frame_fuse}')  # img.txt
            s_fuse += '%gx%g ' % img_fuse.shape[2:]  # print string
            gn_fuse = torch.tensor(im0_fuse.shape)[[1, 0, 1, 0]]  # normalization gain whwh

            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    if save_txt:  # Write to file
                        # xywh_fuse = (xyxy2xywh(torch.tensor(xyxy_fuse).view(1, 4)) / gn_fuse).view(-1).tolist()  # normalized xywh
                        line = (names[int(cls)], conf, *xyxy) if opt.save_conf else (cls, *xyxy)  # label format
                        with open(txt_path_vis + '.txt', 'a') as f:
                            f.write(('%s ' + '%g ' * (len(line)-1)).rstrip() % line + '\n')
                    if save_img or view_img:  # Add bbox to image
                        im0 = im0.copy()
                        label = f'{names[int(cls)]} {conf:.2f}'
                        plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=3)
            elif save_txt:
                f = open(txt_path_vis + '.txt', 'a')
                f.close()

            if len(det_ir):
                # Rescale boxes from img_size to im0 size
                det_ir[:, :4] = scale_coords(img_ir.shape[2:], det_ir[:, :4], im0_ir.shape).round()

                # Print results
                for c_ir in det_ir[:, -1].unique():
                    n_ir = (det_ir[:, -1] == c_ir).sum()  # detections per class
                    s_ir += f"{n_ir} {names[int(c_ir)]}{'s_ir' * (n_ir > 1)}, "  # add to string

                # Write results
                for *xyxy_ir, conf_ir, cls_ir in reversed(det_ir):
                    if save_txt:  # Write to file
                        # xywh_fuse = (xyxy2xywh(torch.tensor(xyxy_fuse).view(1, 4)) / gn_fuse).view(-1).tolist()  # normalized xywh
                        line_ir = (names[int(cls_ir)], conf_ir, *xyxy_ir) if opt.save_conf else (cls_ir, *xyxy_ir)  # label format
                        with open(txt_path_ir + '.txt', 'a') as f_ir:
                            f_ir.write(('%s ' + '%g ' * (len(line_ir)-1)).rstrip() % line_ir + '\n')
                    if save_img or view_img:  # Add bbox to image
                        im0_ir = im0_ir.copy()
                        label_ir = f'{names[int(cls_ir)]} {conf_ir:.2f}'
                        plot_one_box(xyxy_ir, im0_ir, label=label_ir, color=colors[int(cls_ir)], line_thickness=3)
            elif save_txt:
                f_ir = open(txt_path_ir + '.txt', 'a')
                f_ir.close()

            if len(det_fuse):
                # Rescale boxes from img_size to im0 size
                det_fuse[:, :4] = scale_coords(img_fuse.shape[2:], det_fuse[:, :4], im0_fuse.shape).round()

                # Print results
                for c_fuse in det_fuse[:, -1].unique():
                    n_fuse = (det_fuse[:, -1] == c_fuse).sum()  # detections per class
                    s_fuse += f"{n_fuse} {names[int(c_fuse)]}{'s_fuse' * (n_fuse > 1)}, "  # add to string
                # Write results
                for *xyxy_fuse, conf_fuse, cls_fuse in reversed(det_fuse):
                    if save_txt:  # Write to file
                        # xywh_fuse = (xyxy2xywh(torch.tensor(xyxy_fuse).view(1, 4)) / gn_fuse).view(-1).tolist()  # normalized xywh
                        line_fuse = (names[int(cls_fuse)], conf_fuse, *xyxy_fuse) if opt.save_conf else (cls_fuse, *xyxy_fuse)  # label format
                        with open(txt_path_fuse + '.txt', 'a') as f_fuse:
                            f_fuse.write(('%s ' + '%g ' * (len(line_fuse)-1)).rstrip() % line_fuse + '\n')
                    if save_img or view_img:  # Add bbox to image
                        im0_fuse = im0_fuse.copy()
                        label_fuse = f'{names[int(cls_fuse)]} {conf_fuse:.2f}'
                        plot_one_box(xyxy_fuse, im0_fuse, label=label_fuse, color=colors[int(cls_fuse)], line_thickness=3)
            elif save_txt:
                f_fuse = open(txt_path_fuse + '.txt', 'a')
                f_fuse.close()

            # Print time (inference + NMS)
            print(f'{s}Done. ({t2 - t1:.3f}s)')

            # Stream results
            if view_img:
                cv2.imshow(str(p), im0)
                cv2.imshow(str(p_ir), im0_ir)
                cv2.imshow(str(p_fuse), im0_fuse)

            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)
                    cv2.imwrite(save_path_ir, im0_ir)
                    cv2.imwrite(save_path_fuse, im0_fuse)
                # else:  # 'video' or 'stream'
                #     if vid_path != save_path:  # new video
                #         vid_path = save_path
                #         if isinstance(vid_writer, cv2.VideoWriter):
                #             vid_writer.release()  # release previous video writer
                #         if vid_cap:  # video
                #             fps = vid_cap.get(cv2.CAP_PROP_FPS)
                #             w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                #             h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                #         else:  # stream
                #             fps, w, h = 30, im0.shape[1], im0.shape[0]
                #             save_path += '.mp4'
                #         vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                #     vid_writer.write(im0)

            # if save_txt or save_img:
            #     s_fuse = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
            #     print(f"Results saved to {save_dir}{s_fuse}")

    print(f'Done. ({time.time() - t0:.3f}s)')

    with open(os.path.join(opt.main_dir, 'fusion_time_{}.txt'.format(opt.match_method)), 'w') as file:
        for item in fusion_time:
            file.write(item + '\n')

    if opt.view_checkpoint:
        if opt.var:
            var = copy.deepcopy(opt.var)
            print('var_min = {}'.format(min(var)))
            print('var_max = {}'.format(max(var)))
            var = -np.log(var)
            var_str = 0.5
            result = [0 for i in range(int(max(var) // var_str) + 1)]
            for i in range(len(var)):
                result[int(var[i] // var_str)] = result[int(var[i] // var_str)] + 1
            x = [i*var_str for i in range(int(max(var) // var_str) + 1)]
            plt.plot(x, result)
            plt.ylabel('num')
            plt.xlabel('-log(var)')
            plt.title('Statistics histogram of log(var) of similarity matrix')
            plt.savefig(opt.view_output + 'var_statistics_histogram.png')
            f = open(opt.view_output + 'var_statistics_histogram.txt', 'w')
            for i in var:
                f.write(str(i) + '\n')
            f.close()
            plt.show()



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

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # parser.add_argument('--weights-vis', nargs='+', type=str,  default='./runs/train/VIS/weights/best.pt', help='model.pt path(s)')
    # parser.add_argument('--weights-ir', nargs='+', type=str, default='./runs/train/IR3/weights/best.pt', help='model.pt path(s)')
    # parser.add_argument('--weights-vis', nargs='+', type=str, default='1-221228-vis-laser-03.pt', help='model.pt path(s)')
    # parser.add_argument('--weights-ir', nargs='+', type=str, default='1-221228-ir-laser-03.pt', help='model.pt path(s)')
    parser.add_argument('--weights-vis', nargs='+', type=str, default='/home/ljc/vm/ljc_model/yolov5/yolov5-master/runs/train/exp41/weights/best.pt', help='model.pt path(s)')
    parser.add_argument('--weights-ir', nargs='+', type=str, default='/home/ljc/vm/ljc_model/yolov5/yolov5-master/runs/train/exp42/weights/best.pt', help='model.pt path(s)')
    parser.add_argument('--weights-cls', nargs='+', type=str, default='/home/ljc/vm/ljc_model/Image-registration/matchNetPytorch-main/model_best.pth.tar', help='model.pth.tar path(s)')
    parser.add_argument('--weights-deepDIM', nargs='+', type=str, default='/home/ljc/vm/ljc_model/yolov5/yolov5-master/utils/deep_DIM/model/model_D.pth.tar', help='model.pth.tar path(s)')
    # parser.add_argument('--weights-vis', nar gs='+', type=str, default='7-1227weights-vis-best.pt', help='model.pt path(s)')
    # parser.add_argument('--weights-ir', nargs='+', type=str, default='3-1206weights-ir-best.pt', help='model.pt path(s)')

    # parser.add_argument('--weights-vis', nargs='+', type=str, default='vis_kaist.pt', help='model.pt path(s)')
    # parser.add_argument('--weights-ir', nargs='+', type=str, default='ir_kaist.pt', help='model.pt path(s)')

    # parser.add_argument('--weights-vis', nargs='+', type=str, default='ir_model.pt', help='model.pt path(s)')
    # parser.add_argument('--weights-ir', nargs='+', type= str, default='vis_model.pt', help='model.pt path(s)')

    parser.add_argument('--match-method', type=int, default=6, help='choose match-method')  #0:MI  5:Matchnet 6:ours 8:SSIM 20:deepDIM 9:integral_NCC
    parser.add_argument('--ablation-study', default='CR', help='CR, CR_BR, ours = CR+BR+DA')  #CR,CR_BR,ours
    parser.add_argument('--main-dir', type=str, default= '/home/ljc/vm/ljc_model/yolov5/yolov5-master/data/wcndt/forward_test/',help='main_dir')  # file/folder, 0 for webcam   183

    parser.add_argument('--source-vis', type=str, default= 'VIS_test', help='source')  # file/folder, 0 for webcam   183
    parser.add_argument('--source-ir', type=str, default= 'IR_test', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--view-output', default= 'checkpoint', help='save results to project/n8 ame')
    parser.add_argument('--txt-dir', type=str, default= 'detection_txt',help='source')  # file/folder, 0 for webcam

    parser.add_argument('--project', default= 'detect', help='save results to project/name')

    parser.add_argument('--shiftx', type=int, default=0, help='shift along the x-axis')         #0.1 means 32 pixel     # -0.05
    parser.add_argument('--shifty', type=int, default=0, help='shift along the y-axis')
    parser.add_argument('--var', type=list, default=[], help='shift along the y-axis')
    parser.add_argument('--enlarge-box', type=bool, default=True, help='shift along the y-axis')

    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.4, help='object confidence threshold')   #0.4
    parser.add_argument('--iou-thres', type=float, default=0.6, help='IOU threshold for NMS')          #0.6
    parser.add_argument('--fuseiou-thres', type=float, default=0.3, help='IOU threshold for NMS')  #0.6
    parser.add_argument('--contain-thres', type=float, default=0.5, help='A contain intersect')     #0.3
    parser.add_argument('--device', default='0,1', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')

    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--view-checkpoint', action='store_true', help='display results')

    opt = parser.parse_args()

    opt.source_vis = os.path.join(opt.main_dir, opt.source_vis)
    opt.source_ir = os.path.join(opt.main_dir, opt.source_ir)
    opt.view_output = os.path.join(opt.main_dir, opt.view_output)
    opt.txt_dir = os.path.join(opt.main_dir, opt.txt_dir)
    opt.project = os.path.join(opt.main_dir, opt.project)

    print(opt)
    check_requirements(exclude=('pycocotools', 'thop'))

    with torch.no_grad():
        if opt.update:  # update all models (to fix SourceChangeWarning)
            for opt.weights in ['yolov5s.pt', 'yolov5m.pt', 'yolov5l.pt', 'yolov5x.pt']:
                detect()

                strip_optimizer(opt.weights)
        else:
            detect()


