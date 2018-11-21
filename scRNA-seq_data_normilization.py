import numpy as np
import pandas as pd
import scanpy.api as sc
import csv

sc.settings.verbosity = 3  # verbosity: errors (0), warnings (1), info (2), hints (3)
sc.logging.print_versions()
results_file = './write/pbmc3k.h5ad'

sc.settings.set_figure_params(dpi=80)

adata = sc.read_10x_mtx('filtered_gene_bc_matrices/hg19/', var_names='gene_symbols', cache=True)

adata.var_names_make_unique()

sc.pp.filter_cells(adata, min_genes=200)
sc.pp.filter_genes(adata, min_cells=3)

adata.var["mito"] = adata.var_names.str.startswith("MT-")
sc.pp.calculate_qc_metrics(adata, feature_controls=["mito"], inplace=True)

adata = adata[adata.obs['total_features_by_counts'] < 2500, :]
adata = adata[adata.obs['pct_counts_mito'] < 5, :]

adata.raw = sc.pp.log1p(adata, copy=True)

sc.pp.normalize_per_cell(adata, counts_per_cell_after=1e4)

filter_result = sc.pp.filter_genes_dispersion(adata.X, min_mean=0.0125, max_mean=3, min_disp=0.5)

adata = adata[:, filter_result.gene_subset]

df = pd.DataFrame(adata.X.toarray(), columns=adata.var_names).T

print(df.to_csv(sep="\t", quoting=csv.QUOTE_NONE))
