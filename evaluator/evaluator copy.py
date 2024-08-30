import os
import sys
import csv
import glob
import re
from operator import itemgetter

import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns

# Constant values
CHK_MACHINE_TYPE_LINE = 2
FILENAME_COL = 0
MACHINE_TYPE_COL = 0
Y_TRUE_COL = 2
EXTRACTION_ID_COL = 0
SCORE_COL = 1

# Parameters
MAX_FPR = 0.1
EVAL_DATA_LIST_PATH = "./eval_data_list.csv"
TEAMS_ROOT_DIR = "./teams/"
RESULT_DIR = "./teams/"

def save_csv(save_file_path, save_data):
    with open(save_file_path, "w", newline="") as f:
        writer = csv.writer(f, lineterminator='\n')
        writer.writerows(save_data)

def output_result(team_dir, machine_types):
    dir_name = os.path.basename(team_dir)
    result_name = "result_" + dir_name + ".csv"
    result_file = "{result_dir}/{result_name}".format(result_dir=RESULT_DIR, result_name=result_name)
    csv_lines = []

    averaged_result = []
    nums = 0
    machine_types = ['fan', 'pump', 'slider', 'valve']
    for machine_type in machine_types:
        anomaly_score_path_list = sorted(
            glob.glob("{dir}/anomaly_score_{machine_type}_id*".format(dir=team_dir, machine_type=machine_type)))

        csv_lines.append([machine_type])
        csv_lines.append(["id", "AUC", "pAUC", "Optimal Threshold", "F1 Score at Optimal Threshold", "Confusion Matrix"])

        performance = []
        print("=============================================")
        print("MACHINE TYPE IS [{}]".format(machine_type))
        print("---------------------------------------------")

        for anomaly_score_path in anomaly_score_path_list:

            with open(anomaly_score_path) as fp:
                anomaly_score_list = list(csv.reader(fp))
                anomaly_score_list_sort = sorted(anomaly_score_list, key=itemgetter(0))

            machine_id = re.findall('id_[0-9][0-9]', anomaly_score_path)[EXTRACTION_ID_COL]
            print(machine_id)

            y_true = []

            for eval_data in eval_data_list:
                if len(eval_data) < CHK_MACHINE_TYPE_LINE:
                    flag = True if eval_data[MACHINE_TYPE_COL] == machine_type else False
                else:
                    if flag and machine_id in str(eval_data[FILENAME_COL]):
                        y_true.append(float(eval_data[Y_TRUE_COL]))

            y_pred = [float(anomaly_score[SCORE_COL]) for anomaly_score in anomaly_score_list_sort]

            if len(y_true) != len(y_pred):
                print("Err:anomaly_score may be missing")
                sys.exit(1)

            # AUC and pAUC calculation
            auc = metrics.roc_auc_score(y_true, y_pred)
            p_auc = metrics.roc_auc_score(y_true, y_pred, max_fpr=MAX_FPR)
            
            # ROC Curve and Youden's J statistic
            fpr, tpr, thresholds = metrics.roc_curve(y_true, y_pred)
            J = tpr - fpr
            optimal_idx = np.argmax(J)
            optimal_threshold = thresholds[optimal_idx]
            max_f1 = metrics.f1_score(y_true, (np.array(y_pred) > optimal_threshold).astype(int))

            # Convert y_pred to binary predictions based on the optimal threshold
            y_pred_binary = (np.array(y_pred) > optimal_threshold).astype(int)

            # Confusion Matrix using the optimal threshold
            cm = metrics.confusion_matrix(y_true, y_pred_binary, normalize='true')

            # Save confusion matrix as image
            cm_image_file = f"{RESULT_DIR}/{machine_type}_{machine_id}_confusion_matrix.png"
            save_confusion_matrix(cm, cm_image_file, machine_type, machine_id)

            # Save ROC curve with optimal threshold marked
            roc_image_file = f"{RESULT_DIR}/{machine_type}_{machine_id}_roc_curve.png"
            
            csv_lines.append([machine_id.split("_", 1)[1], auc, p_auc, optimal_threshold, max_f1, cm.tolist()])
            performance.append([auc, p_auc])
            print("AUC :", auc)
            print("pAUC :", p_auc)
            print(f"Optimal Threshold: {optimal_threshold}, F1 Score: {max_f1}")
            print("Confusion Matrix:\n", cm)
            print(f"Confusion matrix image saved at {cm_image_file}")
            print(f"ROC curve image saved at {roc_image_file}")

        averaged_performance = np.mean(np.array(performance, dtype=float), axis=0)
        print("\nAUC Average :", averaged_performance[0])
        print("pAUC Average :", averaged_performance[1])
        csv_lines.append(["Average"] + list(averaged_performance))
        csv_lines.append([])
        save_roc_curve(fpr, tpr, thresholds, optimal_threshold, roc_image_file, averaged_performance[0])


        if nums == 0:
            averaged_result = averaged_performance
        else:
            averaged_result += averaged_performance
        nums += 1

    averaged_result /= nums
    print("\nAUC Average :", averaged_result[0])
    print("pAUC Average :", averaged_result[1])
    csv_lines.append(["Average"] + list(averaged_result))
    csv_lines.append([])
    print("=============================================")
    print("AUC and pAUC results -> {}".format(result_file))
    save_csv(save_file_path=result_file, save_data=csv_lines)

def save_confusion_matrix(cm, filename, machine_type, machine_id):
    plt.figure(figsize=(10, 7))
    sns.set(font_scale=1.4)
    sns.heatmap(cm, annot=True, fmt='.3f', cmap='Blues', xticklabels=['normal', 'anomaly'], 
                yticklabels=['normal', 'anomaly'], annot_kws={"size":20})
    plt.title(f'Confusion Matrix for {machine_type} {machine_id}')
    plt.xlabel('Predicted Label', fontsize=15)
    plt.ylabel('True Label', fontsize=15)
    plt.savefig(filename)
    plt.close()

def save_roc_curve(fpr, tpr, thresholds, optimal_threshold, filename, auc):
    plt.figure(figsize=(10, 7))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    
    # Mark the optimal threshold
    optimal_idx = np.where(thresholds == optimal_threshold)[0][0]
    plt.plot(fpr[optimal_idx], tpr[optimal_idx], marker='o', color='red', 
             label=f'Optimal Threshold: {optimal_threshold:.2f}')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=15)
    plt.ylabel('True Positive Rate', fontsize=15)
    plt.title('Receiver Operating Characteristic', fontsize=20)
    plt.legend(loc="lower right")
    plt.savefig(filename)
    plt.close()

if __name__ == "__main__":
    teams_dirs = glob.glob("{root_dir}/*".format(root_dir=TEAMS_ROOT_DIR))

    if os.path.exists(EVAL_DATA_LIST_PATH):
        with open(EVAL_DATA_LIST_PATH) as fp:
            eval_data_list = list(csv.reader(fp))
    else:
        print("Err:eval_data_list.csv not found")
        sys.exit(1)

    machine_types = []

    for idx in eval_data_list:
        if len(idx) < CHK_MACHINE_TYPE_LINE:
            machine_types.append(idx[MACHINE_TYPE_COL])

    for team_dir in teams_dirs:
        if os.path.isdir(team_dir):
            print(team_dir)
            output_result(team_dir, machine_types)
        else:
            print("{} is not directory.".format(team_dir))
