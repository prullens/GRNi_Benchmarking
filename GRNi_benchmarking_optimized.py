import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve
import sys

known_network_file = sys.argv[1]
if len(sys.argv) == 3:
    inference_data_file = sys.argv[2]
elif len(sys.argv) > 3:
    inference_data_file = sys.argv[2:]

def known_network_TO_dict(known_network):
    """"Input a known network file with a 'TF target' format. Output a dict with all possible regulatory interactions. Positive regulatory interactions are 1, negative are 0."""

    # Open known network file in 'TF target' format
    infile = open(known_network)

    known_network_dict = {}
    TF_list = []
    target_list = []


    # First loop to retrieve all positive regulatory interactions from known newtwork
    for line in infile:
        line = line.strip().split()
        TF = line[0]
        TF_list.append(TF)
        target = line[1]
        target_list.append(target)
        regulatory_interaction = TF + '_' + target
        # Fill known_network_dict with all positive regulatory interactions with value 1
        if regulatory_interaction not in known_network_dict:
            known_network_dict[regulatory_interaction] = 1
    infile.close()

    # Second loop to retrieve all possible regulatory interaction from kown network
    for i in range(len(TF_list)):
        TF = TF_list[i]
        target = target_list[i]
        # Get all possible targets for each TF
        for target_2 in target_list:
            regulatory_interaction = TF + '_' + target_2
            if regulatory_interaction not in known_network_dict:
                known_network_dict[regulatory_interaction] = 0
        
    return known_network_dict

def compute_benchmark_scores(known_network, inference_data):
    known_network_dict = known_network_TO_dict(known_network)

    regulatory_interaction_list = []
    y_true_list = []
    score_list = []

    # Open GRN data of infereence network in 'TF target score' format
    infile = open(inference_data)
    for line in infile:
        line = line.strip().split()
        regulatory_interaction = line[0] + '_' + line[1]
        score = line[2]
        # If discovered regulatory interaction found in known network 
        if regulatory_interaction in known_network_dict:
            regulatory_interaction_list.append(regulatory_interaction)
            y_true_list.append(float(known_network_dict[regulatory_interaction]))
            score_list.append(float(score))

    # Derive name of inference file
    inference_ID = inference_data.split('.')        
         
    # Create df with index= regulatory_interactions and y_true and inference score
    df = pd.DataFrame(index= regulatory_interaction_list)
    df['y_true'] = y_true_list  
    df['score'] = score_list

    benchmark_dict = {}

    # Compute different benchmarking scores
    # ROC AUC
    benchmark_dict['ROC'] = [roc_auc_score(df["y_true"], df["score"])]
    # Precision Recall
    benchmark_dict['Prcsn_Rcll'] = [average_precision_score(df["y_true"], df["score"])]

    # Recall at 10 and 50% false discovery rate
    precision, recall, _ = precision_recall_curve(df["y_true"], df["score"])
    fdr = 1 - precision
    cutoff_index_10 = next(i for i, x in enumerate(fdr) if x <= 0.1)
    cutoff_index_50 = next(i for i, x in enumerate(fdr) if x <= 0.5)
    benchmark_dict['Rcll, 10% FDR'] = [recall[cutoff_index_10]]
    benchmark_dict['Rcll, 50% FDR'] = [recall[cutoff_index_50]]

    # Create df including all bechmark scores
    df_benchmark = pd.DataFrame(data= benchmark_dict)
    df_benchmark['data'] = inference_ID[0]
    df_benchmark = df_benchmark.set_index('data')

    return df_benchmark

# if one inference file or more provided?
if type(inference_data_file) == str:
    print(compute_benchmark_scores(known_network_file, inference_data_file))
elif type(inference_data_file) == list:
    column_list = []
    df_local_dict = {}
    for file in inference_data_file:
        df_local = compute_benchmark_scores(known_network_file, file)
        for column in df_local:
            if column not in column_list:
                column_list.append(column)
        for index, row in df_local.iterrows():
            df_local_dict[index] = []
            for value in row:
                df_local_dict[index].append(value)
    df = pd.DataFrame(data=df_local_dict, index=column_list).T
    print(df.sort_values('ROC', ascending=False))  
