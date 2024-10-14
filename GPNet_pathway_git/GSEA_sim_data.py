#%%
# let try GSEA for this data
from pydeseq2.dds import DeseqDataSet
from pydeseq2.ds import DeseqStats

import pandas as pd
import gseapy as gp
from utls.generate_sim_data import generate_sim_data

parameters = {
    'gene_num': 10000,
    'num_sample_one_label': 500,
    'select_num': 5,
    'gene_list_num': 50,
    'n_mean': 1,
    'n_std': 1,
    'selected_gene_num': 10
}

n_mean_list = [1,0.5,0.1,0.05,0.01,0.005,0.001]
type_I_error_list_n_mean = []
type_II_error_list_n_mean = []
accuracy_list_n_mean = []
for n_mean in n_mean_list:
    parameters['n_mean'] = n_mean
    sample_size_list = [3, 5, 10, 50, 100,150,200,250,300,350,400,450, 500]
    type_I_error_list_sample_size = []
    type_II_error_list_sample_size = []
    accuracy_list_sample_size = []
    for sample_size in sample_size_list:
        parameters['num_sample_one_label'] = sample_size
        type_I_error_list = []
        type_II_error_list = []
        accuracy_list = []

        for loop_idx in range(10):
            _, _, data_count, gene_list, labels, gene_list_select = generate_sim_data(parameters)

            labels = labels.astype(int)
            labels_str = labels.astype(str)
            labels_str = list(labels_str)

            gene_set = gene_list.T.to_dict(orient='list')
            gene_set = {str(k): [str(i) for i in v] for k, v in gene_set.items()}
            gs_res = gp.gsea(data=data_count, # or data='./P53_resampling_data.txt'
                            gene_sets=gene_set, # or enrichr library names
                            cls= labels_str, # cls=class_vector
                            # set permutation_type to phenotype if samples >=15
                            permutation_type='phenotype',
                            permutation_num=1000, # reduce number to speed up test
                            outdir=None,  # do not write output to disk
                            method='signal_to_noise',
                            threads=4, seed= 7)


            out = []

            for term in list(gs_res.results):
                out.append([term,
                        gs_res.results[term]['fdr'],
                        gs_res.results[term]['es'],
                        gs_res.results[term]['nes']])

            out_df = pd.DataFrame(out, columns = ['Term','fdr', 'es', 'nes']).sort_values('fdr').reset_index(drop = True)


            gene_list_select = [str(i) for i in gene_list_select]
            ground_truth_set = set(gene_list_select) 
            out_df['is_significant'] = out_df.Term.isin(ground_truth_set)
            out_df['detected_significant'] = out_df.fdr < 0.05

            # calculate TP, FP, TN, FN
            TP = out_df[(out_df.is_significant) & (out_df.detected_significant)].shape[0]
            FP = out_df[(~out_df.is_significant) & (out_df.detected_significant)].shape[0]
            TN = out_df[(~out_df.is_significant) & (~out_df.detected_significant)].shape[0]
            FN = out_df[(out_df.is_significant) & (~out_df.detected_significant)].shape[0]

            # Calculate Type I Error (False Positive Rate), Type II Error (False Negative Rate), and Accuracy
            type_I_error = FP / (FP + TN) if (FP + TN) > 0 else 0
            type_II_error = FN / (TP + FN) if (TP + FN) > 0 else 0
            accuracy = (TP + TN) / (TP + FP + TN + FN) if (TP + FP + TN + FN) > 0 else 0
            print("Type I Error (False Positive Rate):", type_I_error)
            print("Type II Error (False Negative Rate):", type_II_error)
            print("Accuracy:", accuracy)
            type_I_error_list.append(type_I_error)
            type_II_error_list.append(type_II_error)
            accuracy_list.append(accuracy)

        type_I_error_avg = sum(type_I_error_list) / len(type_I_error_list)
        type_II_error_avg = sum(type_II_error_list) / len(type_II_error_list)
        accuracy_avg = sum(accuracy_list) / len(accuracy_list)
        print("Average Type I Error (False Positive Rate):", type_I_error_avg)
        print("Average Type II Error (False Negative Rate):", type_II_error_avg)
        print("Average Accuracy:", accuracy_avg)
        type_I_error_list_sample_size.append(type_I_error_avg)
        type_II_error_list_sample_size.append(type_II_error_avg)
        accuracy_list_sample_size.append(accuracy_avg)

    print(type_I_error_list_sample_size)
    print(type_II_error_list_sample_size)
    print(accuracy_list_sample_size)

    type_I_error_list_n_mean.append(type_I_error_list_sample_size.copy())
    type_II_error_list_n_mean.append(type_II_error_list_sample_size.copy())
    accuracy_list_n_mean.append(accuracy_list_sample_size.copy())
# %%
import pickle
with open('./results/type_I_error_list_n_mean_gsea.pkl', 'wb') as f:
    pickle.dump(type_I_error_list_n_mean, f)
with open('./results/type_II_error_list_n_mean_gsea.pkl', 'wb') as f:
    pickle.dump(type_II_error_list_n_mean, f)
with open('./results/accuracy_list_n_mean_gsea.pkl', 'wb') as f:
    pickle.dump(accuracy_list_n_mean, f)