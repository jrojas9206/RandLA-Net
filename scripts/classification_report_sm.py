import os 
import sys 
import glob 
import json 
import argparse 
import numpy as np 
#import seaborn as sn
#import pandas as pd
# Metrics 
import sklearn 
import sklearn.metrics
# from sklearn.metrics import accuracy_score
# from sklearn.metrics import precision_score
# from sklearn.metrics import recall_score
# from sklearn.metrics import f1_score
# from sklearn.metrics import jaccard_score
# from sklearn.metrics import balanced_accuracy_score
# from sklearn.metrics import matthews_corrcoef



def get_report(lst_files, path2labels, ann_column=3):
    dic2ret = {}
    #print("List of files: %i" %(len(lst_files)))
    if(len(lst_files)==0):
        sys.exit()
    for a_idx, a_file in enumerate(lst_files):
        print("-> Processing[%i/%i]" %(a_idx, len(lst_files)))
        file_name = os.path.split(a_file)[-1].split(".")[0]
        label_path = os.path.join(path2labels, "%s.%s" %(file_name, "txt"))
        if(not os.path.isfile(label_path)):
            continue
        try: 
            original_pc = np.loadtxt(a_file)
        except ValueError as err:
            original_pc = np.loadtxt(a_file, delimiter=',')
        #print(np.unique(original_pc[:,ann_column]))
        
        number_of_fruits = np.max(original_pc[:,ann_column])

        indx2update = np.where(original_pc[:,ann_column] > 0)

        original_pc[indx2update, ann_column] = 1

        # Predictions 
        predicted_labels = np.loadtxt(label_path)[:,3]
        Y = original_pc[:,ann_column]
        Y_Pred = predicted_labels
        # Metrics 
        # balacc = balanced_accuracy_score(original_pc[:,ann_column], predicted_labels, adjusted=True)

        # precision_pc = precision_score(original_pc[:,ann_column], predicted_labels, average="macro", pos_label=[1]) # Apple

        # recall_pc = recall_score(original_pc[:,ann_column], predicted_labels)

        # f1_pc = f1_score(original_pc[:,ann_column], predicted_labels, pos_label=1) # Apple

        # acc_point_cloud = accuracy_score(original_pc[:,ann_column], predicted_labels)
        
        # mcc = matthews_corrcoef(original_pc[:,ann_column], predicted_labels)
        
        # jaccard_pc_apple = jaccard_score(original_pc[:,ann_column], predicted_labels, labels=[1], average='macro')
        # jaccard_pc_other = jaccard_score(original_pc[:,ann_column], predicted_labels, labels=[0], average='macro')
        # jacc_macro = (jaccard_pc_other+jaccard_pc_apple)/2.0

        global_balanced_acc = sklearn.metrics.balanced_accuracy_score(
            Y, Y_Pred)

        precision, recall, f1_score_, support = sklearn.metrics.precision_recall_fscore_support(
            Y, Y_Pred)

        macro_recall = sum(recall) / 2.0
        macro_precision = sum(precision) / 2.0

        macro_f1_score = (2 * macro_precision * macro_recall) / \
            (macro_precision + macro_recall)

        confusion_matrix = sklearn.metrics.multilabel_confusion_matrix(
            Y, Y_Pred)

        mcc = sklearn.metrics.matthews_corrcoef(Y, Y_Pred)

        iou = sklearn.metrics.jaccard_score(Y, Y_Pred, average=None)

        miou = (iou[0] + iou[1]) / 2

        # Values to keep 
        if("names" not in dic2ret.keys()):
            dic2ret["names"] = [file_name]
            dic2ret["number_of_fruits"] = [number_of_fruits]
            #dic2ret["acc"] = [acc_point_cloud]
            dic2ret["precision_nonapple"] = [precision[0]]
            dic2ret["precision_apple"] = [precision[1]]
            dic2ret["recall_nonapple"] = [recall[0]]
            dic2ret["recall_apple"] = [recall[1]]
            dic2ret["f1_nonapple"] = [f1_score_[0] ]
            dic2ret["f1_apple"] = [f1_score_[1]]
            dic2ret["jaccard_nonapple"] = [iou[0]]
            dic2ret["jaccard_apple"] = [iou[1]]
            #dic2ret["jaccard_other"] = [jaccard_pc_other]
            dic2ret["jaccard_macro"] = [miou]
            dic2ret["balancedACC"] = [global_balanced_acc]
            dic2ret["mcc"] = [mcc]
            dic2ret["macroF1"]  = [macro_f1_score]
        else:
            dic2ret["names"].append(file_name)
            dic2ret["number_of_fruits"].append(number_of_fruits)
            #dic2ret["acc"].append(acc_point_cloud)
            dic2ret["precision_nonapple"].append(precision[0])
            dic2ret["precision_apple"].append(precision[1])
            dic2ret["recall_nonapple"].append(recall[0])
            dic2ret["recall_apple"].append(recall[1])
            dic2ret["f1_nonapple"].append(f1_score_[0])
            dic2ret["f1_apple"].append(f1_score_[1])
            dic2ret["jaccard_nonapple"].append(iou[0])
            dic2ret["jaccard_apple"].append(iou[1])
            #dic2ret["jaccard_other"].append(jaccard_pc_other)
            dic2ret["balancedACC"].append(global_balanced_acc)
            dic2ret["mcc"].append(mcc)
            dic2ret["jaccard_macro"].append(miou)
            dic2ret["macroF1"].append(macro_f1_score)

    # print("acc: %f" %( np.mean(np.array(dic2ret["acc"]))) )
    #print("precision: %f" %(np.mean(np.array(dic2ret["precision_apple"]))))
    #print("recall: %f" %(np.mean(np.array(dic2ret["recall_apple"]))))
    #print("F1: %f " %(np.mean(np.array(dic2ret["f1_apple"]))))
    #print("jaccard_apple: %f" %(np.mean(np.array(dic2ret["jaccard_apple"]))))
    #print("jaccard macro: %f" %(jacc_macro))
    #print("Balanced acc: %f"  %(np.mean(np.array(dic2ret["balancedACC"]))))
    print("mcc: %f" %(np.mean(np.array(dic2ret["mcc"]))))
    print("F1-Macro: %f" %(np.mean(np.array(dic2ret["macroF1"]))))
    print("F1-Apple: %f" %(np.mean(np.array(dic2ret["f1_apple"]))))
    print("balancedACC: %f" %(np.mean(np.array(dic2ret["balancedACC"]))))
    print("jaccard_macro: %f" %(np.mean(np.array(dic2ret["jaccard_macro"]))))

    
    return dic2ret



def main():
    parser = argparse.ArgumentParser(" ")
    parser.add_argument("originalpc", type=str, help="Path to the original point clouds")
    parser.add_argument("predicted_label", type=str, help="Path to the predicted labels")
    parser.add_argument("exp", type=str, help="Name of the experiment")
    parser.add_argument("--annColumnGT", type=int, help="Index of the annotation of the ground truth", default=6)
    args = parser.parse_args()
    dic2save = {}

    lst_files = glob.glob(os.path.join(args.originalpc, "*.txt"))

    print("Found files: %s" %(len(lst_files)))

    dicreport = get_report(lst_files, args.predicted_label, ann_column=args.annColumnGT)

    dic2save[args.exp] = dicreport


    with open("%s_%s.json" %(args.exp, "report"), 'w') as f:
        json.dump(dic2save, f, ensure_ascii=False)

    return 0

if __name__=="__main__":
    sys.exit(main())