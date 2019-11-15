import os
import sys
import ast
import pandas as pd
from sklearn.metrics import roc_auc_score

this_dir = os.path.abspath(os.path.dirname(__file__))
projects_dir = os.path.dirname(this_dir)
if projects_dir not in sys.path: sys.path.append(projects_dir)

from common_utils.general_utils import csv_to_df, df_to_csv

file_dir = '/Users/recep/Documents/NumericoTech/multi_agent_incidents_prediction/preds_labels_connected_epoch31/'
all_preds_labels = sorted([f for f in os.listdir(file_dir)])
confusion = '/Users/recep/Documents/NumericoTech/multi_agent_incidents_prediction/confusion_connected_epoch31/ConfusionMatrix_2019-11-14 05:33:21.553976_epoch31_batchsize32_lr0.005_sr0.02_connected1.csv'
output = '/Users/recep/Documents/NumericoTech/multi_agent_incidents_prediction/results/auroc_inc_connected.csv'

# get amount of incidents per segment using confusion matrix
c_df = csv_to_df(confusion, sep='|')
inc_df = c_df[c_df['type'] == 'fn']
inc_df = inc_df.drop('type', 1)
inc_df['auroc'] = None

df_list = list()
epoch_counter = 1
for pred_label_epoch in all_preds_labels:
    epoch_df = inc_df.copy()
    epoch_df['epoch'] = epoch_counter
    input = os.path.join(file_dir, pred_label_epoch)
    # df for preds & labels of respective epoch
    pl_df = csv_to_df(input, sep=',')
    segments = list(pl_df)
    for segment_id in segments:
        seg_df = pl_df[segment_id]
        segment_list = seg_df.tolist()
        predictions = list()
        labels = list()
        for segment in segment_list:
            prediction, label = ast.literal_eval(segment)
            predictions.append(prediction)
            labels.append(label)
        predictions = sum(predictions, [])
        labels = sum(labels, [])
        if sum(labels) == 0:
            auroc_segment = 0
        else:
            auroc_segment = roc_auc_score(labels, predictions)
        epoch_df.loc[
            epoch_df['segment_id'] == segment_id, 'auroc'] = auroc_segment
    epoch_counter += 1
    df_list.append(epoch_df)
df_to_csv(pd.concat(df_list), output)
