import pandas as pd
from sklearn import linear_model
import sys
import csv

sc_expression_data = sys.argv[1]
human_tfs = sys.argv[2]

test_data = pd.read_table(sc_expression_data, index_col=0)
tfs = pd.read_table(human_tfs, index_col=0, names='TF')
clf = linear_model.BayesianRidge()

X = test_data.loc[test_data.index.intersection(tfs.index)].T
coefs = pd.DataFrame(index=X.columns)

for target in test_data.index:
    # Get the expression values of target
    y = test_data.loc[target]
    # Expression of TFs
    X_local = X.loc[:,X.columns != target]
    # Run regression
    clf.fit(X_local, y)
    # Update the coefficients (regulatory relationships)
    coefs[target] = 0
    coefs.loc[X_local.columns,target] = clf.coef_

network = coefs.reset_index().melt(id_vars="index")

print(network.to_csv(sep="\t", quoting=csv.QUOTE_NONE))
