# FINAL VERSION 3, INCLUDING BENCHMARKING

import pandas as pd     # Import required modules. 
import numpy as np
import sys
from sklearn.metrics import roc_auc_score, average_precision_score

known_interactions_file = sys.argv[1]
inference_data_file = sys.argv[2]
benchmark_type = sys.argv[3]

def known_interactions_TO_df(known_interactions):
    """Converts a known network file to a df containing all possible interactions and whether they do "y_true=1" or do not "y_true=0" interact. Accepted formats: 'TF target value' with value being either '1' of interaction or '0' of no interaction, 'TF target' which assumes that all provided TFs regulate the following target"""
    
    TF_list = []                 # Calling used variables. 
    target_list = []
    interaction_value_list = []
    interaction_dict = {}
    
    infile = open(known_interactions) # Opening known network file and check if it already also contains the non interacting 'TF target'. Generate interaction dict for df. 
    for line in infile:
        line = line.strip()
        if 'y_true' in line:
            pass
        else:
            if '_' in line:
                line = line.split()
                interaction_dict[line[0]] = line[1]
            else:
                line = line.split('\t')
                TF_list.append(line[0])
                target_list.append(line[1])
                if len(line) == 3:
                    interaction_value_list.append(line[2])
                else:
                    interaction_value_list.append('1')

      
    if '0' in interaction_value_list:   # Converting network with only positive interactions to complete network that also includes the 'TF target's that do not interact. Returns a dict for df.
        for i in range (len(TF_list)):
            TF = TF_list[i]
            target = target_list[i]
            interaction_value = interaction_value_list[i]
            interaction_dict[TF + '_' + target] = interaction_value 
    elif '0' not in interaction_value_list:
        for i in range (len(TF_list)):
            TF = TF_list[i]
            target = target_list[i]
            interaction_dict[TF + '_' + target] = 1
            for target_2 in target_list:
                if TF + '_' + target_2 not in interaction_dict:
                    interaction_dict[TF + '_' + target_2] = 0

    df = pd.DataFrame({'y_true': interaction_dict}) 

    return df

def compute_benchmark(known_interactions, inference_data, benchmark):
    """Allows the computation of two benchmarks the ROC and the precision recall with as input a known network df and the Gene Regulatory Network Inference data in a 'TF target score' format"""  
    
    df = known_interactions_TO_df(known_interactions)

    interaction_dict = {}

    infile = open(inference_data)
    for line in infile:
        line = line.strip()
        line = line.split()
        interaction = line[0] + '_' + line[1]
        if interaction in df.index.values:
            interaction_dict[interaction] = float(line[2])

    df_2 = pd.DataFrame({'score': interaction_dict})      
    df = pd.concat([df,df_2], axis=1)
    df = df[np.isfinite(df['score'])]
    
    if benchmark == 'ROC':
        print(roc_auc_score(df["y_true"], df["score"]))
    elif benchmark == 'precision_recall':
        print(average_precision_score(df["y_true"], df["score"]))
    else:
        print('choose either the "ROC" or "precision_recall" benchmark as third argument')
        
compute_benchmark(known_interactions_file, inference_data_file, benchmark_type)  
