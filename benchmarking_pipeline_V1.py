import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
import sys

known_interactions_file = sys.argv[1]
inference_data_file = sys.argv[2]
benchmark_type = sys.argv[3]

def known_interactions_TO_df(known_interactions):
    """Converts a known network file to a dict containing all possible interactions and whether they do "1" or do not "0" interact. Accepted formats: 'TF target value' with value being either '1' of interaction or '0' of no interaction, 'TF target' which assumes that all provided TFs regulate the following target"""
    
    TF_list = []
    target_list = []
    interaction_value_list = []
    interaction_dict = {}
    error1_list = []
    
    infile = open(known_interactions)
    for line in infile:
        line = line.strip()
        if 'y_true' in line:
            pass
        else:
            if '_' in line:
                line = line.split()
                interaction_dict[line[0]] = line[1]
            else:
                line = line.split()
                TF_list.append(line[0])
                target_list.append(line[1])
                if len(line) == 3:
                    interaction_value_list.append(line[2])
                else:
                    interaction_value_list.append('1')

    if not error1_list:
        if '0' in interaction_value_list:
            for i in range (len(TF_list)):
                TF = TF_list[i]
                target = target_list[i]
                interaction_value = interaction_value_list[i]
                interaction_dict[TF + '_' + target] = interaction_value 
        elif '0' not in interaction_value_list:
            for i in range (len(TF_list)):
                TF = TF_list[i]
                target = target_list[i]
                interaction_value = interaction_value_list[i]
                interaction_dict[TF + '_' + target] = 1
                for target_2 in target_list:
                    if TF + '_' + target_2 not in interaction_dict:
                        interaction_dict[TF + '_' + target_2] = 0

        return interaction_dict


def compute_benchmark(known_interactions, inference_data, benchmark):
    """Allows for the computation of two benchmarks the ROC and the precision recall with as input a known network dict and the Gene Regulatory Network Inference data in a 'TF target score' format"""
    
    interaction_dict_known = known_interactions_TO_df(known_interactions)
    interaction_dict_data = {}
    df_index_list = []

    infile = open(inference_data)
    for line in infile:
        line = line.strip()
        line = line.split()
        interaction = line[0] + '_' + line[1]
        if interaction in interaction_dict_known:
            interaction_dict_data[interaction] = float(line[2])
            df_index_list.append(interaction)

    df = pd.DataFrame(index= df_index_list)
    y_true_list = []
    score_list = []

    for interaction in interaction_dict_data:
        y_true_list.append(interaction_dict_known[interaction])
        score_list.append(interaction_dict_data[interaction])

    df['y_true'] = y_true_list
    df['score'] = score_list   

    if benchmark == 'ROC':
        print(roc_auc_score(df["y_true"], df["score"]))
    elif benchmark == 'precision_recall':
        print(average_precision_score(df["y_true"], df["score"]))
    else:
        print('Choose either the "ROC" or "precision_recall" benchmark as third argument')

compute_benchmark(known_interactions_file, inference_data_file, benchmark_type)
