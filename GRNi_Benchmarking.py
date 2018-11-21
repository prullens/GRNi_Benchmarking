# Import python modules.
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve
import sys

# Define arguments that can be called from the shell and check if the four required arguments are provided.

known_interactions_file = sys.argv[1]
input_TFs_file = sys.argv[2]
if len(sys.argv) == 4:
    inference_data_file = sys.argv[3]
elif len(sys.argv) > 4:
    inference_data_file = sys.argv[3:]

# Define a function that converts known validated interactions to all possible interactions.    
def known_interactions_TO_df(known_interactions, input_TFs):
    """Converts a known network file to a dict containing all possible interactions and whether they do "1" or do not "0" interact. Accepted formats: 'TF target value' with value being either '1' of interaction or '0' of no interaction, 'TF target' which assumes that all provided TFs regulate the following target. Additioanlly, one may choose to select a .txt file containing specific TFs that need to be assessed. The latter is particularly usefull in case of really large datasets as it then only computes the provided list of TFs and hence increases the speed of the computational pipeline substantially."""
    
    # Defining variables used in this function.
    TF_list = []
    target_list = []
    interaction_value_list = []
    interaction_dict = {}
    input_TFs_list = []
    
    # Optionally, a file that contains a list with TFs that need to be checked can be provided, which is opened here.
    if input_TFs.endswith('.txt'):
        infile = open(input_TFs)
        for line in infile:
            line = line.strip()
            input_TFs_list.append(line)
    else:
        pass
    
    # Known interactions file is opened and checked for whether it is already converted to a complete network or yet only contains the validated TF to target interactions. 
    infile = open(known_interactions)
    for line in infile:
        line = line.strip()
        if 'y_true' in line:
            pass
        else:
            if '_' in line:  
                line = line.split()
                if line[0] in input_TFs_list:
                    interaction_dict[line[0]] = line[1]
                elif len(input_TFs_list) == 0:
                    interaction_dict[line[0]] = line[1]  
            else:
                line = line.split()
                if line[0] in input_TFs_list:
                    TF_list.append(line[0])
                    target_list.append(line[1])
                    if len(line) == 3:
                        interaction_value_list.append(line[2])
                    else:
                        interaction_value_list.append('1')
                elif len(input_TFs_list) == 0:
                    TF_list.append(line[0])
                    target_list.append(line[1])
                    if len(line) == 3:
                        interaction_value_list.append(line[2])
                    else:
                        interaction_value_list.append('1')

   # Checks if known network file is already converted to a complete network by screening for 0's, which implies that the file also contains the not interacting "TF targtes" and thus is converted already. Retrieves a dictionary with 'TF_target':1 or 0. 
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


# Define a function that allows for computing the ROC or precision_recall score.
def compute_benchmark(known_interactions, input_TFs, inference_data):
    """Allows for the computation of two benchmarks the ROC and the precision recall with as input a known network dict and the Gene Regulatory Network Inference data in a 'TF target score' format"""
    
    # Define used variables for this function
    interaction_dict_known = known_interactions_TO_df(known_interactions, input_TFs)
    interaction_dict_data = {}
    df_index_list = []
    y_true_list = []
    score_list = []
    
    # Open a .txt file that contains the data derived from the used inference method in a "TF target score" format and checks whether the inference derived interaction can be found back in the complete known interactions file.
    infile = open(inference_data)
    for line in infile:
        line = line.strip()
        line = line.split()
        interaction = line[0] + '_' + line[1]
        if interaction in interaction_dict_known:
            interaction_dict_data[interaction] = float(line[2])
            df_index_list.append(interaction)
    
    # Create a dataframe with all the derived interacitons that are also present in the complete known network as the index.
    df = pd.DataFrame(index= df_index_list)
    
    # Append two lists, one with the y_true value that corresponds to the interaction and the other with the probability score that was derived from the inference method also corresponding to the interactions.
    for interaction in interaction_dict_data:
        y_true_list.append(interaction_dict_known[interaction])
        score_list.append(interaction_dict_data[interaction])
    
    # Add new columns to the dataframe with the y_true value for the interaction from the complete known network and a probability score for the corresponding interaction that was derived from the inference method.  
    df['y_true'] = y_true_list
    df['score'] = score_list  

    # Allowing to compute two different benchmarking methods which can be defined in the shell. The ROC and precision_recall score.
    inference_ID = inference_data.split('.')
    benchmark_dict = {}

    # Allowing to compute two different benchmarking methods which can be defined in the shell. The ROC and precision_recall score.
    benchmark_dict['ROC'] = [roc_auc_score(df["y_true"], df["score"])]
    benchmark_dict['Prcsn_Rcll'] = [average_precision_score(df["y_true"], df["score"])]
    
    precision, recall, _ = precision_recall_curve(df["y_true"], df["score"])
    fdr = 1 - precision
    cutoff_index_10 = next(i for i, x in enumerate(fdr) if x <= 0.1)
    cutoff_index_50 = next(i for i, x in enumerate(fdr) if x <= 0.5)
    benchmark_dict['Rcll, 10% FDR'] = [recall[cutoff_index_10]]
    benchmark_dict['Rcll, 50% FDR'] = [recall[cutoff_index_50]]
    
    df_benchmark = pd.DataFrame(data= benchmark_dict)
    df_benchmark['data'] = inference_ID[0]
    df_benchmark = df_benchmark.set_index('data')
    return df_benchmark
    
# One inferecence_data file provided or more?
if type(inference_data_file) == str:
    print(compute_benchmark(known_interactions_file, input_TFs_file, inference_data_file))
elif type(inference_data_file) == list:
    column_list = []
    df_local_dict = {}
    for file in inference_data_file:
        df_local = compute_benchmark(known_interactions_file, input_TFs_file, file)
        for column in df_local:
            if column not in column_list:
                column_list.append(column)
        for index, row in df_local.iterrows():
            df_local_dict[index] = []
            for value in row:
                df_local_dict[index].append(value)
    df = pd.DataFrame(data=df_local_dict, index=column_list).T
    print(df.sort_values('ROC', ascending=False))    
