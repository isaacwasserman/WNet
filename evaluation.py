# From https://github.com/kanezaki/pytorch-unsupervised-segmentation-tip/issues/2
from glob import glob
import numpy as np
import tqdm
import os
import cv2

def calculate_mIOU(gt_path, pred_path):
    user_input_flg = False
    resize_flg = False
    no_background_flg = False

    args_input = pred_path
    args_gt = gt_path
    args_bsd500 = True
    args_mode = 1

    input_list = sorted(glob(args_input + '/*'))

    miou_list = []
    max_miou_list = []
    categorical_miou_array = np.zeros( (21, 2) )

    for input_csv in tqdm.tqdm(input_list):
        if True:
            raw_input_array = np.loadtxt(input_csv, delimiter=',')
            if user_input_flg:
                input_csv = input_csv[:-3] + "csv"
                gt_arrays = np.loadtxt(args_gt + input_csv[-16:], delimiter=',')
            elif args_bsd500:
                no_background_flg = False
                gt_arrays = []
                for i in range(100):
                    fname = args_gt + "/" + input_csv.split("/")[-1][:-4] + "-" + str(i) + ".csv"
                    if not os.path.exists(fname):
                        break
                    gt_arrays.append( np.loadtxt(fname, delimiter=',') )
                if args_mode == 2:
                    gt_arrays = gt_arrays[ np.argmax( np.array([ len(np.unique(g)) for g in gt_arrays ]) ) ]
                elif args_mode == 3:
                    gt_arrays = gt_arrays[ np.argmin( np.array([ len(np.unique(g)) for g in gt_arrays ]) ) ]
                gt_arrays = np.array(gt_arrays)
            else:
                gt_arrays = cv2.imread(args_gt + input_csv[-16:-3]+"png", -1)
            if resize_flg:
                input_array = cv2.resize( raw_input_array, (gt_arrays.shape[1],gt_arrays.shape[0]) , interpolation = cv2.INTER_NEAREST  )
            else:
                input_array = raw_input_array

        if len(gt_arrays.shape) == 2:
            gt_arrays = [gt_arrays]
        
        miou_per_gt_segmentation = []
        for gt_array in gt_arrays:
            miou_for_each_class = []
            label_list = np.unique(gt_array)

            # gt_mask is 0 where gt label is 0 (background) and 1 where gt label is not 0 (foreground)
            gg = np.zeros(gt_array.shape)
            gt_mask = np.where(gt_array > 0, 1, gg)

            # determinant array is input array but with background discarded (set to 0)
            determinant_array = gt_mask * input_array
            # label_list is the list of labels in the input array (range(k))
            label_list = np.unique(gt_array)


            gt_array_1d = gt_array.reshape((gt_array.shape[0]*gt_array.shape[1])) # 1d array of gt labels
            input_array_1d = input_array.reshape((input_array.shape[0]*input_array.shape[1])) # 1d array of input labels

            # For each class in range k
            for l in label_list:
                inds = np.where( gt_array_1d == l )[0] # indices of gt where label is l
                pred_labels = input_array_1d[ inds ] # predictions at those indices
                u_pred_labels = np.unique(pred_labels) # unique predictions at those indices
                hists = [ np.sum(pred_labels == u) for u in u_pred_labels ] # frequency of each unique prediction at those indices
                fractions = [ len(inds) + np.sum(input_array_1d == u) - np.sum(pred_labels == u) for u in u_pred_labels ] # (total number of pixels in gt with label l) + (total number of pixels in input with label u) - (total number of pixels in input with label u and gt label l)
                mious = hists / np.array(fractions,dtype='float')
                miou_list.append( np.max(mious) )
                miou_for_each_class.append( np.max(mious) )
            miou_per_gt_segmentation.append( np.mean(miou_for_each_class) )
        max_miou_list.append( np.max(miou_per_gt_segmentation) )
        

    average_mIOU = sum(miou_list) / float(len(miou_list))
    max_mIOU = sum(max_miou_list) / float(len(max_miou_list))
    return average_mIOU, max_mIOU