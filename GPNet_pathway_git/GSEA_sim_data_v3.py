#%%
# let try GSEA for this data
from pydeseq2.dds import DeseqDataSet
from pydeseq2.ds import DeseqStats

import pandas as pd
import gseapy as gp
from utls.generate_sim_data import generate_sim_data
import numpy as np
import pickle

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
type_I_error_avg_list_snr = []
power_avg_list_snr = []
for n_mean in n_mean_list:
    parameters['n_mean'] = n_mean
    sample_size_list = [3, 5, 10, 50, 100,150,200,250]
    type_I_error_list_sample_size = []
    type_II_error_list_sample_size = []
    accuracy_list_sample_size = []
    type_I_error_avg_list_sample_size = []
    power_avg_list_sample_size = []
    for sample_size in sample_size_list:
        parameters['num_sample_one_label'] = sample_size
        type_I_error_list = []
        type_II_error_list = []
        accuracy_list = []
        type_I_error_sum = 0
        power_sum = 0
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
                out.append([int(term),
                        gs_res.results[term]['fdr'],
                        gs_res.results[term]['es'],
                        gs_res.results[term]['nes']])

            out_df = pd.DataFrame(out, columns = ['Term','fdr', 'es', 'nes']).sort_values('Term').reset_index(drop = True)
            fdr = out_df.loc[:, 'fdr']
            gene_set_label = np.array([i in gene_list_select for i in range(len(gene_list))])
            threshold_list = np.array(sorted(fdr))

            for threshold in threshold_list:
                predictions = np.array(fdr) <= threshold
                # calculate TP, TN, FP, FN
                TP = np.sum((predictions == True) & (gene_set_label == True))
                TN = np.sum((predictions == False) & (gene_set_label == False))
                FP = np.sum((predictions == True) & (gene_set_label == False))
                FN = np.sum((predictions == False) & (gene_set_label == True))

                # Calculate Type II Error (False Negative Rate), and Accuracy
                # type_I_error = FP / (FP + TN) if (FP + TN) > 0 else 0
                type_II_error = FN / (TP + FN) if (TP + FN) > 0 else 0

                if type_II_error > 0.1:
                    continue
                else:
                    type_I_error = FP / (FP + TN) if (FP + TN) > 0 else 0
                    break
            print('threshold:', threshold)
            print('type_I_error:', type_I_error)
            type_I_error_sum += type_I_error
            power = 0
            for threshold in threshold_list[::-1]:
                predictions = np.array(fdr) <= threshold

                #calculate TP, TN, FP, FN
                TP = np.sum((predictions == True) & (gene_set_label == True))
                TN = np.sum((predictions == False) & (gene_set_label == False))
                FP = np.sum((predictions == True) & (gene_set_label == False))
                FN = np.sum((predictions == False) & (gene_set_label == True))

                # calculate Type I Error
                type_I_error = FP / (FP + TN) if (FP + TN) > 0 else 0
                if type_I_error > 0.1:
                    continue
                else:
                    power = TP / (TP+FN) if (TP+FN) > 0 else 0
                    break
            print("Power (True Positive Rate):", power)
            print("threshold:", threshold)
            power_sum += power
        power_avg = power_sum / 10
        type_I_error_avg = type_I_error_sum / 10
        type_I_error_avg_list_sample_size.append(type_I_error_avg)
        power_avg_list_sample_size.append(power_avg)
    type_I_error_avg_list_snr.append(type_I_error_avg_list_sample_size)
    power_avg_list_snr.append(power_avg_list_sample_size)
# %%
# import pickle
# with open('./results/type_I_error_list_n_mean_gsea.pkl', 'wb') as f:
#     pickle.dump(type_I_error_list_n_mean, f)
# with open('./results/type_II_error_list_n_mean_gsea.pkl', 'wb') as f:
#     pickle.dump(type_II_error_list_n_mean, f)
# with open('./results/accuracy_list_n_mean_gsea.pkl', 'wb') as f:
#     pickle.dump(accuracy_list_n_mean, f)
with open('./results/type_I_error_avg_list_snr_sim_gsea.pkl', 'wb') as f:
    pickle.dump(type_I_error_avg_list_snr, f)
with open('./results/power_avg_list_snr_sim_gsea.pkl', 'wb') as f:
    pickle.dump(power_avg_list_snr, f)
# %%
