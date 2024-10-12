import numpy as np
import os
import data_io

from prf_metrics import cal_prf_metrics
def cal_global_acc(pred, gt):
    """
    acc = (TP+TN)/all_pixels
    """
    h,w = gt.shape
    return [np.sum(pred==gt), float(h*w)]

def get_statistics_seg(pred, gt, num_cls=2):
    """
    return tp, fp, fn
    """
    h,w = gt.shape
    statistics = []
    for i in range(num_cls):
        tp = np.sum((pred==i)&(gt==i))
        fp = np.sum((pred==i)&(gt!=i))
        fn = np.sum((pred!=i)&(gt==i))
        statistics.append([tp, fp, fn])
    return statistics

def get_statistics_prf(pred, gt):
    """
    return tp, fp, fn
    """
    tp = np.sum((pred==1)&(gt==1))
    fp = np.sum((pred==1)&(gt==0))
    fn = np.sum((pred==0)&(gt==1))
    return [tp, fp, fn]

def segment_metrics(pred_list, gt_list, num_cls = 2):
    global_accuracy_cur = []
    statistics = []

    for pred, gt in zip(pred_list, gt_list):
        gt_img = (gt / 255).astype('uint8')
        pred_img = (pred / 255).astype('uint8')
        # calculate each image
        # global_accuracy_cur里面是全部图片global acc的值
        # statistics 里面是全部图片 tp, fp, fn的值
        # statistics_AIU 里面是全部图片 IOU 的值
        global_accuracy_cur.append(cal_global_acc(pred_img, gt_img))
        statistics.append(get_statistics_seg(pred_img, gt_img, num_cls))


    # get global accuracy with corresponding threshold: (TP+TN)/all_pixels
    global_acc = np.sum([v[0] for v in global_accuracy_cur]) / np.sum([v[1] for v in global_accuracy_cur])

    # get tp, fp, fn
    counts = []
    for i in range(num_cls):
        tp = np.sum([v[i][0] for v in statistics])
        fp = np.sum([v[i][1] for v in statistics])
        fn = np.sum([v[i][2] for v in statistics])

        counts.append([tp, fp, fn])

    # calculate mean accuracy
    mean_acc = np.sum([v[0] / (v[0] + v[2]) for v in counts]) / num_cls
    # calculate mean iou
    mean_iou_acc = np.sum([v[0] / (np.sum(v)) for v in counts]) / num_cls


    return global_acc, mean_acc, mean_iou_acc

def prf_metrics(pred_list, gt_list):
    statistics = []

    for pred, gt in zip(pred_list, gt_list):
        gt_img = (gt / 255).astype('uint8')
        # pred_img = (pred / 255 ).astype('uint8')
        # print(pred_img)

        #gt_img = (gt / np.max(gt))
        pred_img = (((pred / np.max(pred))>0.5)).astype('uint8')

        # calculate each image
        statistics.append(get_statistics_prf(pred_img, gt_img))

    # get tp, fp, fn
    tp = np.sum([v[0] for v in statistics])
    fp = np.sum([v[1] for v in statistics])
    fn = np.sum([v[2] for v in statistics])
    print("tp:{}, fp:{}, fn:{}".format(tp,fp,fn))
    # calculate precision
    p_acc = 1.0 if tp == 0 and fp == 0 else tp / (tp + fp)
    # calculate recall
    r_acc = tp / (tp + fn)
    # calculate f-score
    f_acc = 2 * p_acc * r_acc / (p_acc + r_acc)
    return p_acc,r_acc,f_acc

def thred_half(src_img_list, tgt_img_list):
    Precision, Recall, F_score = prf_metrics(src_img_list, tgt_img_list)
    Global_Accuracy, Class_Average_Accuracy, Mean_IOU = segment_metrics(src_img_list, tgt_img_list)
    print("Global Accuracy:{}, Class Average Accuracy:{}, Mean IOU:{}, Precision:{}, Recall:{}, F score:{}".format(
        Global_Accuracy, Class_Average_Accuracy, Mean_IOU, Precision, Recall, F_score))

def get_statistics(pred, gt):
    """
    return tp, fp, fn
    """
    tp = np.sum((pred==1)&(gt==1))
    fp = np.sum((pred==1)&(gt==0))
    fn = np.sum((pred==0)&(gt==1))
    return [tp, fp, fn]

# def cal_OIS_metrics(pred_list, gt_list, thresh_step=0.01,issave=False):
#     save_data = {
#         "p_acc":[],
#         "r_acc": [],
#         "F1": [],
#     }
#     final_F1_list = []
#     for pred, gt in zip(pred_list, gt_list):
#         p_acc_list = []
#         r_acc_list = []
#         F1_list = []
#         for thresh in np.arange(0.0, 1.0, thresh_step):
#             gt_img = (gt / 255).astype('uint8')
#             pred_img = (pred / 255 > thresh).astype('uint8')
#             tp, fp, fn = get_statistics(pred_img, gt_img)
#             # calculate precision
#             p_acc = 1.0 if tp == 0 and fp == 0 else tp / (tp + fp)
#             # calculate recall
#             #print("nan racc", tp, fn)
#             if tp + fn == 0:
#                 r_acc=0
#             else:
#                 r_acc = tp / (tp + fn)
#             # F1
#             #print("nan f1:",  thresh, p_acc,r_acc)
#             if p_acc + r_acc==0:
#                 F1 = 0
#             else:
#                 F1 = 2 * p_acc * r_acc / (p_acc + r_acc)
#             #
#
#             #
#             p_acc_list.append(p_acc)
#             r_acc_list.append(r_acc)
#             #print("F1:",F1)
#             F1_list.append(F1)
#         #
#         if save_data:
#             save_data["p_acc"].append(p_acc_list)
#             save_data["r_acc"].append(r_acc_list)
#             save_data["F1"].append(F1_list)
#
#         assert len(p_acc_list)==100, "p_acc_list is not 100"
#         assert len(r_acc_list)==100, "r_acc_list is not 100"
#         assert len(F1_list)==100, "F1_list is not 100"
#         #
#         max_F1 = np.max(np.array(F1_list))
#         #print("max_F1:", max_F1)
#         #if np.isnan(max_F1):
#         #    raise "---"
#         final_F1_list.append(max_F1)
#
#     #print("-", np.sum(np.array(final_F1_list)), len(final_F1_list))
#     final_F1 = np.sum(np.array(final_F1_list))/len(final_F1_list)
#     if save_data:
#         save_OIS_txt(data=save_data)
#     return final_F1

def cal_OIS_metrics(pred_list, gt_list, thresh_step=0.01,issave=False):
    save_data = {
        "p_acc":[],
        "r_acc": [],
        "F1": [],
    }
    final_F1_list = []
    for pred, gt in zip(pred_list, gt_list):
        p_acc_list = []
        r_acc_list = []
        F1_list = []
        for thresh in np.arange(0.0, 1.0, thresh_step):
            gt_img = (gt / 255).astype('uint8')
            pred_img = (pred / 255 > thresh).astype('uint8')
            tp, fp, fn = get_statistics(pred_img, gt_img)
            # calculate precision
            p_acc = 1.0 if tp == 0 and fp == 0 else tp / (tp + fp)
            # calculate recall
            #print("nan racc", tp, fn)
            if tp + fn == 0:
                r_acc=0
            else:
                r_acc = tp / (tp + fn)
            # F1
            #print("nan f1:",  thresh, p_acc,r_acc)
            if p_acc + r_acc==0:
                F1 = 0
            else:
                F1 = 2 * p_acc * r_acc / (p_acc + r_acc)
            #

            #
            p_acc_list.append(p_acc)
            r_acc_list.append(r_acc)
            #print("F1:",F1)
            F1_list.append(F1)
        #
        if issave:
            save_data["p_acc"].append(p_acc_list)
            save_data["r_acc"].append(r_acc_list)
            save_data["F1"].append(F1_list)

        assert len(p_acc_list)==100, "p_acc_list is not 100"
        assert len(r_acc_list)==100, "r_acc_list is not 100"
        assert len(F1_list)==100, "F1_list is not 100"
        max_F1 = np.max(np.array(F1_list))
        final_F1_list.append(max_F1)

    final_F1 = np.sum(np.array(final_F1_list))/len(final_F1_list)
    if save_data:
        save_OIS_txt(data=save_data)
    return final_F1

def save_OIS_txt(data,thresh_step = 0.01,root_path = "save_OIS_data"):
    p_acc = data["p_acc"]
    r_acc = data["r_acc"]
    F1 = data["F1"]

    len_image = len(p_acc)
    assert len(p_acc[0])==100
    for i, thresh in enumerate(np.arange(0.0, 1.0, thresh_step)):
        path = os.path.join(root_path,str(thresh))
        if not os.path.isdir(path):
            os.makedirs(path)

        for j in range(len_image):
            pacc = p_acc[j][i]
            racc = r_acc[j][i]
            f1 = F1[j][i]
            f = open(os.path.join(path,'pacc.txt'), 'a')
            f.write("{}:{}\n".format(j,(str(pacc))))
            f.close()
            f = open(os.path.join(path,'racc.txt'), 'a')
            f.write("{}:{}\n".format(j,(str(racc))))
            f.close()
            f = open(os.path.join(path,'f1.txt'), 'a')
            f.write("{}:{}\n".format(j,(str(f1))))
            f.close()

def cal_ODS_metrics(pred_list, gt_list, thresh_step=0.01,issave=False):
    save_data = {
        "ODS": [],
    }
    final_ODS = []
    for thresh in np.arange(0.0, 1.0, thresh_step):
        ODS_list = []
        for pred, gt in zip(pred_list, gt_list):
            gt_img = (gt / 255).astype('uint8')
            pred_img = (pred / 255 > thresh).astype('uint8')
            tp, fp, fn = get_statistics(pred_img, gt_img)
            # calculate precision
            p_acc = 1.0 if tp == 0 and fp == 0 else tp / (tp + fp)
            if tp + fn == 0:
                r_acc=0
            else:
                r_acc = tp / (tp + fn)
            if p_acc + r_acc==0:
                F1 = 0
            else:
                F1 = 2 * p_acc * r_acc / (p_acc + r_acc)
            ODS_list.append(F1)

        if issave:
            save_data["ODS"].append(ODS_list)
        ave_F1 = np.mean(np.array(ODS_list))
        final_ODS.append(ave_F1)
    ODS = np.max(np.array(final_ODS))
    return ODS

def save_ODS_txt(data,thresh_step = 0.01,root_path = "save_ODS_data"):
    ODS_list = data["ODS"]

    assert len(ODS_list) == 100
    len_image = len(ODS_list[0])

    for i, thresh in enumerate(np.arange(0.0, 1.0, thresh_step)):
        path = os.path.join(root_path,str(thresh))
        if not os.path.isdir(path):
            os.makedirs(path)

        for j in range(len_image):
            ODS = ODS_list[i][j]

            f = open(os.path.join(path,'ODS.txt'), 'a')
            f.write("{}:{}\n".format(j,(str(ODS))))
            f.close()

def cal_mIoU_metrics(pred_list, gt_list, thresh_step=0.01,issave=False,pred_imgs_names=None, gt_imgs_names=None):
    final_iou = []
    for thresh in np.arange(0.0, 1.0, thresh_step):
        iou_list = []
        # ĳһ  ֵ £     ͼƬ  miou
        for i, (pred, gt) in enumerate(zip(pred_list, gt_list)):
            gt_img = (gt / 255).astype('uint8')
            pred_img = (pred / 255 > thresh).astype('uint8')
            TP = np.sum((pred_img == 1) & (gt_img == 1)) #TP
            TN = np.sum((pred_img == 0) & (gt_img == 0))  # TN
            FP = np.sum((pred_img == 1) & (gt_img == 0))  # FP
            FN = np.sum((pred_img == 0) & (gt_img == 1))  # FN
            if (FN + FP + TP) <= 0:
                iou = 0
            else:
                iou_1 = TP / (FN + FP + TP)
                iou_0 = TN / (FN + FP + TN)
                iou = (iou_1 + iou_0)/2
            iou_list.append(iou)

        ave_iou = np.mean(np.array(iou_list))
        final_iou.append(ave_iou)
    mIoU = np.max(np.array(final_iou))
    return mIoU


results_dir = "../results"
model_name = "SFIAN"
suffix_gt = "label_viz"
suffix_pred = "fused"

results_dir = os.path.join(results_dir, model_name, 'test_latest', 'images')
src_img_list, tgt_img_list, pred_imgs_names, gt_imgs_names = data_io.get_image_pairs(results_dir, suffix_gt, suffix_pred)
#
assert len(src_img_list) == len(tgt_img_list)

final_accuracy_all = cal_prf_metrics(src_img_list, tgt_img_list)
final_accuracy_all = np.array(final_accuracy_all)

# Precision, Recall, F_score = final_accuracy_all[:,1], final_accuracy_all[:,2], final_accuracy_all[:,3]
Precision_list = []
Recall_list = []
F_score_list = []
Precision_list, Recall_list, F_score_list = final_accuracy_all[:,1], final_accuracy_all[:,2], final_accuracy_all[:,3]
final_f1 = np.max(np.array(F_score_list))


# print(len(F_score_list))
for i in range(len(F_score_list)):
    if F_score_list[i] == final_f1:
        max_i = i
        # print("index -> ", i)

# print('Precision -> ')
# print(Precision_list)
# print('Recall -> ')
# print(Recall_list)
# print('F_score -> ')
# print(F_score_list)
# #


mIoU = cal_mIoU_metrics(src_img_list,tgt_img_list,issave=True,pred_imgs_names=pred_imgs_names, gt_imgs_names=gt_imgs_names)
print("mIoU:", mIoU)

ODS = cal_ODS_metrics(src_img_list,tgt_img_list,issave=True)
print("ODS:", ODS)

OIS = cal_OIS_metrics(src_img_list,tgt_img_list,issave=True)
print("OIS:",OIS)

F1 = final_f1
print("F1: ", F1)

print("Reacll: ", Recall_list[max_i])

print("Precision: ", Precision_list[max_i])


