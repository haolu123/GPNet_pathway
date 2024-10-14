#%%
import pickle
import matplotlib.pyplot as plt
import math
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
#%%

SNR_list = [1,0.5,0.1,0.05,0.01] #,0.005,0.001]
sample_size_list = [3, 5, 10, 50, 100, 150, 200, 250]
colors_line = ['black','blue','green','coral','yellow','purple',  'red']
colors_area = ['black','blue','green','coral','yellow','purple',  'red'] #['gray', 'blue', 'lightgreen', 'lightcoral', 'yellow']
line_styles = ['-', '--', '-.', ':', 'solid','dotted']
markers = ['o', 'x', '+', 's', 'd','*']
methods = ['GPNet', 'LR', 'SVM', 'GSEA', 'ENRICHR', 'DeepHisCom', 'PASNet']

# Function to create a gradient fill
def gradient_fill_between(ax, x, y1, y2, color, num_steps=100, alpha=0.1):
    cmap = LinearSegmentedColormap.from_list('custom_gradient', [color, 'white'], N=num_steps)
    gradient_colors = [cmap(i / (num_steps*2)) for i in range(num_steps)]
    for i in range(num_steps):
        ax.fill_between(x, np.minimum(y1, y2) + i * (np.maximum(y1, y2) - np.minimum(y1, y2)) / num_steps,
                        np.minimum(y1, y2) + (i + 1) * (np.maximum(y1, y2) - np.minimum(y1, y2)) / num_steps,
                        color=gradient_colors[i], alpha=alpha)
# %%
with open('./results/type_I_error_avg_list_snr_cls_real_v2.pkl', 'rb') as f:
    type_I_error_avg_list_snr = pickle.load(f)
with open('./results/type_I_error_avg_list_snr_cls_lr_real_v1.pkl', 'rb') as f:
    type_I_error_avg_list_snr_lr = pickle.load(f)
with open('./results/type_I_error_avg_list_snr_cls_svm_real_v1.pkl', 'rb') as f:
    type_I_error_avg_list_snr_svm = pickle.load(f)
with open("./results/type_I_error_avg_list_snr_sim_cls_HistCom_real.pkl", 'rb') as f:
    type_I_error_avg_list_snr_HistCom = pickle.load(f)
with open('./results/type_I_error_avg_list_snr_PASNet_real.pkl', 'rb') as f:
    type_I_error_avg_list_snr_pasnet = pickle.load(f)

for i in range(len(type_I_error_avg_list_snr_HistCom)):
    type_I_error_avg_list_snr_HistCom[i] = sorted(type_I_error_avg_list_snr_HistCom[i], reverse=True)
    for j in range(len(type_I_error_avg_list_snr_HistCom[i])):
        if type_I_error_avg_list_snr_HistCom[i][j] <= 0.12244897959183673:
            type_I_error_avg_list_snr_HistCom[i][j] = 0.12244897959183673
name_list = ['1.0','0.5','0.1','0.05','0.01','0.005','0.001']
type_I_error_avg_list_snr_gsea_list = []
type_I_error_avg_list_snr_enrichr_list = []
mm_our =  [[10]*len(sample_size_list),[-1]*len(sample_size_list)]
mm_lr = [[10]*len(sample_size_list),[-1]*len(sample_size_list)]
mm_svm = [[10]*len(sample_size_list),[-1]*len(sample_size_list)]
mm_gsea = [[10]*len(sample_size_list),[-1]*len(sample_size_list)]
mm_enrichr = [[10]*len(sample_size_list),[-1]*len(sample_size_list)]
mm_Histcom = [[10]*len(sample_size_list),[-1]*len(sample_size_list)]
mm_PASNet = [[10]*len(sample_size_list),[-1]*len(sample_size_list)]

mm_pairs = [mm_our, mm_lr, mm_svm, mm_gsea, mm_enrichr, mm_Histcom, mm_PASNet]
fig, ax = plt.subplots()
type_I_error_avg_all = []
for i in range(len(SNR_list)):
    snr = SNR_list[i]
    snr_name = name_list[i]
    # print(snr)
    with open(f'./results/type_I_error_avg_list_snr_gsea_v3_[{snr_name}].pkl','rb') as f:
        type_I_error_avg_list_snr_gsea = pickle.load(f)
    with open(f'./results/type_I_error_avg_list_snr_enrichr_v1_[{snr_name}].pkl','rb') as f:
        type_I_error_avg_list_snr_enrichr = pickle.load(f)
    type_I_error_avg_list_snr_gsea_list.append(type_I_error_avg_list_snr_gsea[0])
    type_I_error_avg_list_snr_enrichr_list.append(type_I_error_avg_list_snr_enrichr[0])
    ax.plot(sample_size_list, type_I_error_avg_list_snr[i][:len(sample_size_list)], marker=markers[i], linestyle=line_styles[i], color=colors_line[0])
    ax.plot(sample_size_list, type_I_error_avg_list_snr_lr[i][:len(sample_size_list)], marker=markers[i], linestyle=line_styles[i], color=colors_line[1])
    ax.plot(sample_size_list, type_I_error_avg_list_snr_svm[i][:len(sample_size_list)], marker=markers[i], linestyle=line_styles[i], color=colors_line[2])
    ax.plot(sample_size_list[:len(type_I_error_avg_list_snr_gsea[0])], type_I_error_avg_list_snr_gsea[0], marker=markers[i], linestyle=line_styles[i], color=colors_line[3])
    ax.plot(sample_size_list[:len(type_I_error_avg_list_snr_enrichr[0])], type_I_error_avg_list_snr_enrichr[0], marker=markers[i], linestyle=line_styles[i], color=colors_line[4])
    ax.plot(sample_size_list, type_I_error_avg_list_snr_HistCom[i][:len(sample_size_list)], marker=markers[i], linestyle=line_styles[i], color=colors_line[5])
    ax.plot(sample_size_list, type_I_error_avg_list_snr_pasnet[i][:len(sample_size_list)], marker=markers[i], linestyle=line_styles[i], color=colors_line[6])
    # plt.plot(sample_size_list, type_I_error_avg_list_snr_gsea[i], label=f'GSEA ', marker='x', linestyle='--')
type_I_error_avg_all = [type_I_error_avg_list_snr, type_I_error_avg_list_snr_lr, 
                    type_I_error_avg_list_snr_svm, 
                    type_I_error_avg_list_snr_gsea_list, 
                    type_I_error_avg_list_snr_enrichr_list,
                    type_I_error_avg_list_snr_HistCom,
                    type_I_error_avg_list_snr_pasnet]
for j in range(len(sample_size_list)):
    for k in range(len(mm_pairs)):
        for m in range(len(SNR_list)):
            mm_pairs[k][0][j] = min(mm_pairs[k][0][j], type_I_error_avg_all[k][m][j])
            mm_pairs[k][1][j] = max(mm_pairs[k][1][j], type_I_error_avg_all[k][m][j])
            if k==0 and j==0:
                print(mm_pairs[k][0][j])

for idx, ((y1,y2), color) in enumerate(zip(mm_pairs, colors_area)):
    if idx == 0:
        gradient_fill_between(ax, sample_size_list, y1, y2, color, alpha=1)
    else:    
        gradient_fill_between(ax, sample_size_list, y1, y2, color)

method_lines = [plt.Line2D([0], [0], color=color, linestyle='-', markersize=8)
                for color in colors_line]
method_labels = methods

# Custom legend for SNR
snr_lines = [plt.Line2D([0], [0], color='black', linestyle=style, marker=marker, markersize=8)
             for style, marker in zip(line_styles, markers)]
snr_labels = [f'SNR={snr}' for snr in SNR_list]

# Add legends to the plot
first_legend = plt.legend(method_lines, method_labels, title='Methods', loc='upper left')
ax = plt.gca().add_artist(first_legend)
second_legend = plt.legend(snr_lines, snr_labels, title='SNR', loc='lower left')
ax = plt.gca().add_artist(second_legend)
plt.xlabel('Sample Size')
plt.ylabel('Type I Error Rate')
# plt.title(f'Comparison of Type I Error Rates Across Different sample sizes, \n SNR = {snr}, type II error is 0.1')
# plt.legend(loc='best')  # You can choose a different location for the legend if required
plt.grid(True)  # Optional: Adds a grid
# plt.savefig(f'./results/type_I_error_avg_snr_{snr}.png')
plt.show()
# %%
with open('./results/power_avg_list_snr_cls_real_v2.pkl', 'rb') as f:
    power_avg_list_snr = pickle.load(f)
with open('./results/power_avg_list_snr_cls_lr_real_v1.pkl', 'rb') as f:
    power_avg_list_snr_lr = pickle.load(f)
with open('./results/power_avg_list_snr_cls_svm_real_v1.pkl', 'rb') as f:
    power_avg_list_snr_svm = pickle.load(f)
with open('./results/power_avg_list_snr_sim_cls_HistCom_real.pkl', 'rb') as f:
    power_avg_list_snr_HistCom = pickle.load(f)
with open('./results/power_avg_list_snr_PASNet_real.pkl', 'rb') as f:
    power_avg_list_snr_PASNet = pickle.load(f)

name_list = ['1.0','0.5','0.1','0.05','0.01','0.005','0.001']
power_avg_list_snr_gsea_list = []
power_avg_list_snr_enrichr_list = []
mm_our =  [[10]*len(sample_size_list),[-1]*len(sample_size_list)]
mm_lr = [[10]*len(sample_size_list),[-1]*len(sample_size_list)]
mm_svm = [[10]*len(sample_size_list),[-1]*len(sample_size_list)]
mm_gsea = [[10]*len(sample_size_list),[-1]*len(sample_size_list)]
mm_enrichr = [[10]*len(sample_size_list),[-1]*len(sample_size_list)]
mm_histcom = [[10]*len(sample_size_list),[-1]*len(sample_size_list)]
mm_pasnet = [[10]*len(sample_size_list),[-1]*len(sample_size_list)]

mm_pairs = [mm_our, mm_lr, mm_svm, mm_gsea, mm_enrichr, mm_histcom, mm_pasnet]
fig, ax = plt.subplots()
for i in range(len(SNR_list)):
    snr = SNR_list[i]
    snr_name = name_list[i]
    print(snr)
    with open(f'./results/power_avg_list_snr_gsea_v3_[{snr_name}].pkl','rb') as f:
        power_avg_list_snr_gsea = pickle.load(f)
    with open(f'./results/power_avg_list_snr_enrichr_v1_[{snr_name}].pkl','rb') as f:
        power_avg_list_snr_enrichr = pickle.load(f)
    power_avg_list_snr_gsea_list.append(power_avg_list_snr_gsea[0])
    power_avg_list_snr_enrichr_list.append(power_avg_list_snr_enrichr[0])
    ax.plot(sample_size_list, power_avg_list_snr[i][:len(sample_size_list)], marker=markers[i], linestyle=line_styles[i], color=colors_line[0])
    ax.plot(sample_size_list, power_avg_list_snr_lr[i][:len(sample_size_list)], marker=markers[i], linestyle=line_styles[i], color=colors_line[1])
    ax.plot(sample_size_list, power_avg_list_snr_svm[i][:len(sample_size_list)], marker=markers[i], linestyle=line_styles[i], color=colors_line[2])
    ax.plot(sample_size_list[:len(power_avg_list_snr_gsea[0])], power_avg_list_snr_gsea[0], marker=markers[i], linestyle=line_styles[i], color=colors_line[3])
    ax.plot(sample_size_list[:len(power_avg_list_snr_enrichr[0])], power_avg_list_snr_enrichr[0], marker=markers[i], linestyle=line_styles[i], color=colors_line[4])
    ax.plot(sample_size_list, power_avg_list_snr_HistCom[i][:len(sample_size_list)], marker=markers[i], linestyle=line_styles[i], color=colors_line[5])
    ax.plot(sample_size_list, power_avg_list_snr_PASNet[i][:len(sample_size_list)], marker=markers[i], linestyle=line_styles[i], color=colors_line[6])
    # plt.plot(sample_size_list, type_I_error_avg_list_snr_gsea[i], label=f'GSEA ', marker='x', linestyle='--')

power_avg_all = [power_avg_list_snr, power_avg_list_snr_lr, 
                    power_avg_list_snr_svm, 
                    power_avg_list_snr_gsea_list, 
                    power_avg_list_snr_enrichr_list,
                    power_avg_list_snr_HistCom,
                    power_avg_list_snr_PASNet]
for j in range(len(sample_size_list)):
    for k in range(len(mm_pairs)):
        for m in range(len(SNR_list)):
            mm_pairs[k][0][j] = min(mm_pairs[k][0][j], power_avg_all[k][m][j])
            mm_pairs[k][1][j] = max(mm_pairs[k][1][j], power_avg_all[k][m][j])
            if k==0 and j==0:
                print(mm_pairs[k][0][j])

for idx, ((y1,y2), color) in enumerate(zip(mm_pairs, colors_area)):
    if idx == 0:
        gradient_fill_between(ax, sample_size_list, y1, y2, color, alpha=1)
    else:    
        gradient_fill_between(ax, sample_size_list, y1, y2, color)

# for i in range(len(SNR_list)-1):
#     snr = SNR_list[i]
#     snr_name = name_list[i]
#     print(snr)
#     plt.fill_between(sample_size_list, power_avg_list_snr_lr[i][:len(sample_size_list)],power_avg_list_snr_lr[i+1][:len(sample_size_list)], alpha=0.2, color = colors_area[1])
#     plt.fill_between(sample_size_list, power_avg_list_snr_svm[i][:len(sample_size_list)],power_avg_list_snr_svm[i+1][:len(sample_size_list)], alpha=0.4, color = colors_area[2])
#     plt.fill_between(sample_size_list, power_avg_list_snr_gsea_list[i][0],power_avg_list_snr_gsea_list[i+1][0], alpha=0.4, color = colors_area[3])
#     plt.fill_between(sample_size_list, power_avg_list_snr_enrichr_list[i][0],power_avg_list_snr_enrichr_list[i+1][0], alpha=0.2, color = colors_area[4])
#     plt.fill_between(sample_size_list, power_avg_list_snr[i][:len(sample_size_list)],power_avg_list_snr[i+1][:len(sample_size_list)], alpha=0.9, color = colors_area[0])
method_lines = [plt.Line2D([0], [0], color=color, linestyle='-', markersize=8)
                for color in colors_line]
method_labels = methods

# Custom legend for SNR
snr_lines = [plt.Line2D([0], [0], color='black', linestyle=style, marker=marker, markersize=8)
             for style, marker in zip(line_styles, markers)]
snr_labels = [f'SNR={snr}' for snr in SNR_list]

# Add legends to the plot
first_legend = plt.legend(method_lines, method_labels, title='Methods', loc='upper left')
ax = plt.gca().add_artist(first_legend)
second_legend = plt.legend(snr_lines, snr_labels, title='SNR', loc='lower left')
ax = plt.gca().add_artist(second_legend)
plt.xlabel('Sample Size')
plt.ylabel('Type I Error Rate')
# plt.title(f'Comparison of Type I Error Rates Across Different sample sizes, \n SNR = {snr}, type II error is 0.1')
# plt.legend(loc='best')  # You can choose a different location for the legend if required
plt.grid(True)  # Optional: Adds a grid
# plt.savefig(f'./results/type_I_error_avg_snr_{snr}.png')
plt.show()
# %%
with open('./results/type_I_error_avg_list_snr_sim_cls.pkl', 'rb') as f:
    type_I_error_avg_list_snr = pickle.load(f)
with open('./results/type_I_error_avg_list_snr_sim_cls_lr.pkl', 'rb') as f:
    type_I_error_avg_list_snr_lr = pickle.load(f)
with open('./results/type_I_error_avg_list_snr_sim_cls_svm.pkl', 'rb') as f:
    type_I_error_avg_list_snr_svm = pickle.load(f)
with open(f'./results/type_I_error_avg_list_snr_sim_gsea.pkl','rb') as f:
    type_I_error_avg_list_snr_gsea = pickle.load(f)
with open(f'./results/type_I_error_avg_list_snr_sim_enrichr.pkl','rb') as f:
    type_I_error_avg_list_snr_enrichr = pickle.load(f)
with open('./results/type_I_error_avg_list_snr_sim_cls_HistCom.pkl', 'rb') as f:
    type_I_error_avg_list_snr_HistCom = pickle.load(f)
with open("./results/type_I_error_avg_list_snr_sim_PASNet.pkl", 'rb') as f:
    type_I_error_avg_list_snr_sim_PASNet = pickle.load(f)

name_list = ['1.0','0.5','0.1','0.05','0.01','0.005','0.001']
mm_our =  [[10]*len(sample_size_list),[-1]*len(sample_size_list)]
mm_lr = [[10]*len(sample_size_list),[-1]*len(sample_size_list)]
mm_svm = [[10]*len(sample_size_list),[-1]*len(sample_size_list)]
mm_gsea = [[10]*len(sample_size_list),[-1]*len(sample_size_list)]
mm_enrichr = [[10]*len(sample_size_list),[-1]*len(sample_size_list)]
mm_HistCom = [[10]*len(sample_size_list),[-1]*len(sample_size_list)]
mm_PASNet = [[10]*len(sample_size_list),[-1]*len(sample_size_list)]
mm_pairs = [mm_our, mm_lr, mm_svm, mm_gsea, mm_enrichr, mm_HistCom, mm_PASNet]
fig, ax = plt.subplots()
for i in range(len(SNR_list)):
    snr = SNR_list[i]
    snr_name = name_list[i]
    print(snr)
    ax.plot(sample_size_list, type_I_error_avg_list_snr[i][:len(sample_size_list)], marker=markers[i], linestyle=line_styles[i], color=colors_line[0])
    ax.plot(sample_size_list, type_I_error_avg_list_snr_lr[i][:len(sample_size_list)], marker=markers[i], linestyle=line_styles[i], color=colors_line[1])
    ax.plot(sample_size_list, type_I_error_avg_list_snr_svm[i][:len(sample_size_list)], marker=markers[i], linestyle=line_styles[i], color=colors_line[2])
    ax.plot(sample_size_list, type_I_error_avg_list_snr_gsea[i][:len(sample_size_list)], marker=markers[i], linestyle=line_styles[i], color=colors_line[3])
    ax.plot(sample_size_list, type_I_error_avg_list_snr_enrichr[i][:len(sample_size_list)], marker=markers[i], linestyle=line_styles[i], color=colors_line[4])
    ax.plot(sample_size_list, type_I_error_avg_list_snr_HistCom[i][:len(sample_size_list)], marker=markers[i], linestyle=line_styles[i], color=colors_line[5])
    ax.plot(sample_size_list, type_I_error_avg_list_snr_sim_PASNet[i][:len(sample_size_list)], marker=markers[i], linestyle=line_styles[i], color=colors_line[6])
type_I_error_avg_all = [type_I_error_avg_list_snr, type_I_error_avg_list_snr_lr, 
                    type_I_error_avg_list_snr_svm, 
                    type_I_error_avg_list_snr_gsea, 
                    type_I_error_avg_list_snr_enrichr,
                    type_I_error_avg_list_snr_HistCom,
                    type_I_error_avg_list_snr_sim_PASNet]
for j in range(len(sample_size_list)):
    for k in range(len(mm_pairs)):
        for m in range(len(SNR_list)):
            mm_pairs[k][0][j] = min(mm_pairs[k][0][j], type_I_error_avg_all[k][m][j])
            mm_pairs[k][1][j] = max(mm_pairs[k][1][j], type_I_error_avg_all[k][m][j])

for idx, ((y1,y2), color) in enumerate(zip(mm_pairs, colors_area)):
    if idx == 0:
        gradient_fill_between(ax, sample_size_list, y1, y2, color, alpha=1)
    else:    
        gradient_fill_between(ax, sample_size_list, y1, y2, color)
method_lines = [plt.Line2D([0], [0], color=color, linestyle='-', markersize=8)
                for color in colors_line]
method_labels = methods

# Custom legend for SNR
snr_lines = [plt.Line2D([0], [0], color='black', linestyle=style, marker=marker, markersize=8)
             for style, marker in zip(line_styles, markers)]
snr_labels = [f'SNR={snr}' for snr in SNR_list]

# Add legends to the plot
first_legend = plt.legend(method_lines, method_labels, title='Methods', loc='upper left')
ax = plt.gca().add_artist(first_legend)
second_legend = plt.legend(snr_lines, snr_labels, title='SNR', loc='lower left')
ax = plt.gca().add_artist(second_legend)
plt.xlabel('Sample Size')
plt.ylabel('Type I Error Rate')
# plt.title(f'Comparison of Type I Error Rates Across Different sample sizes, \n SNR = {snr}, type II error is 0.1')
# plt.legend(loc='best')  # You can choose a different location for the legend if required
plt.grid(True)  # Optional: Adds a grid
# plt.savefig(f'./results/type_I_error_avg_snr_{snr}.png')
plt.show()
# %%
with open('./results/power_avg_list_snr_sim_cls.pkl', 'rb') as f:
    power_avg_list_snr = pickle.load(f)
with open('./results/power_avg_list_snr_sim_cls_lr.pkl', 'rb') as f:
    power_avg_list_snr_lr = pickle.load(f)
with open('./results/power_avg_list_snr_sim_cls_svm.pkl', 'rb') as f:
    power_avg_list_snr_svm = pickle.load(f)
with open(f'./results/power_avg_list_snr_sim_gsea.pkl','rb') as f:
    power_avg_list_snr_gsea = pickle.load(f)
with open(f'./results/power_avg_list_snr_sim_enrichr.pkl','rb') as f:
    power_avg_list_snr_enrichr = pickle.load(f)
with open('./results/power_avg_list_snr_sim_cls_HistCom.pkl', 'rb') as f:
    power_avg_list_snr_HistCom = pickle.load(f)
with open('./results/power_avg_list_snr_sim_PASNet.pkl', 'rb') as f:
    power_avg_list_snr_sim_PASNet = pickle.load(f)

name_list = ['1.0','0.5','0.1','0.05','0.01','0.005','0.001']
mm_our =  [[10]*len(sample_size_list),[-1]*len(sample_size_list)]
mm_lr = [[10]*len(sample_size_list),[-1]*len(sample_size_list)]
mm_svm = [[10]*len(sample_size_list),[-1]*len(sample_size_list)]
mm_gsea = [[10]*len(sample_size_list),[-1]*len(sample_size_list)]
mm_enrichr = [[10]*len(sample_size_list),[-1]*len(sample_size_list)]
mm_histcom = [[10]*len(sample_size_list),[-1]*len(sample_size_list)]
mm_pasnet = [[10]*len(sample_size_list),[-1]*len(sample_size_list)]
mm_pairs = [mm_our, mm_lr, mm_svm, mm_gsea, mm_enrichr, mm_histcom, mm_pasnet]
fig, ax = plt.subplots()
for i in range(len(SNR_list)):
    snr = SNR_list[i]
    snr_name = name_list[i]
    print(snr)
    ax.plot(sample_size_list, power_avg_list_snr[i][:len(sample_size_list)], marker=markers[i], linestyle=line_styles[i], color=colors_line[0])
    ax.plot(sample_size_list, power_avg_list_snr_lr[i][:len(sample_size_list)], marker=markers[i], linestyle=line_styles[i], color=colors_line[1])
    ax.plot(sample_size_list, power_avg_list_snr_svm[i][:len(sample_size_list)], marker=markers[i], linestyle=line_styles[i], color=colors_line[2])
    ax.plot(sample_size_list, power_avg_list_snr_gsea[i][:len(sample_size_list)], marker=markers[i], linestyle=line_styles[i], color=colors_line[3])
    ax.plot(sample_size_list, power_avg_list_snr_enrichr[i][:len(sample_size_list)], marker=markers[i], linestyle=line_styles[i], color=colors_line[4])
    ax.plot(sample_size_list, power_avg_list_snr_HistCom[i][:len(sample_size_list)], marker=markers[i], linestyle=line_styles[i], color=colors_line[5])
    ax.plot(sample_size_list, power_avg_list_snr_sim_PASNet[i][:len(sample_size_list)], marker=markers[i], linestyle=line_styles[i], color=colors_line[6])
    
power_avg_all = [power_avg_list_snr, power_avg_list_snr_lr, 
                    power_avg_list_snr_svm, 
                    power_avg_list_snr_gsea, 
                    power_avg_list_snr_enrichr,
                    power_avg_list_snr_HistCom,
                    power_avg_list_snr_sim_PASNet]
for j in range(len(sample_size_list)):
    for k in range(len(mm_pairs)):
        for m in range(len(SNR_list)):
            mm_pairs[k][0][j] = min(mm_pairs[k][0][j], power_avg_all[k][m][j])
            mm_pairs[k][1][j] = max(mm_pairs[k][1][j], power_avg_all[k][m][j])
            if k==0 and j==0:
                print(mm_pairs[k][0][j])

for idx, ((y1,y2), color) in enumerate(zip(mm_pairs, colors_area)):
    if idx == 0:
        gradient_fill_between(ax, sample_size_list, y1, y2, color, alpha=1)
    else:    
        gradient_fill_between(ax, sample_size_list, y1, y2, color)

# for i in range(len(SNR_list)-1):
#     snr = SNR_list[i]
#     snr_name = name_list[i]
#     print(snr)
#     plt.fill_between(sample_size_list, power_avg_list_snr_lr[i][:len(sample_size_list)],power_avg_list_snr_lr[i+1][:len(sample_size_list)], alpha=0.2, color = colors_area[1])
#     plt.fill_between(sample_size_list, power_avg_list_snr_svm[i][:len(sample_size_list)],power_avg_list_snr_svm[i+1][:len(sample_size_list)], alpha=0.4, color = colors_area[2])
#     plt.fill_between(sample_size_list, power_avg_list_snr_gsea_list[i][0],power_avg_list_snr_gsea_list[i+1][0], alpha=0.4, color = colors_area[3])
#     plt.fill_between(sample_size_list, power_avg_list_snr_enrichr_list[i][0],power_avg_list_snr_enrichr_list[i+1][0], alpha=0.2, color = colors_area[4])
#     plt.fill_between(sample_size_list, power_avg_list_snr[i][:len(sample_size_list)],power_avg_list_snr[i+1][:len(sample_size_list)], alpha=0.9, color = colors_area[0])
method_lines = [plt.Line2D([0], [0], color=color, linestyle='-', markersize=8)
                for color in colors_line]
method_labels = methods

# Custom legend for SNR
snr_lines = [plt.Line2D([0], [0], color='black', linestyle=style, marker=marker, markersize=8)
             for style, marker in zip(line_styles, markers)]
snr_labels = [f'SNR={snr}' for snr in SNR_list]

# Add legends to the plot
first_legend = plt.legend(method_lines, method_labels, title='Methods', loc='upper left')
ax = plt.gca().add_artist(first_legend)
second_legend = plt.legend(snr_lines, snr_labels, title='SNR', loc='lower left')
ax = plt.gca().add_artist(second_legend)
plt.xlabel('Sample Size')
plt.ylabel('Type I Error Rate')
# plt.title(f'Comparison of Type I Error Rates Across Different sample sizes, \n SNR = {snr}, type II error is 0.1')
# plt.legend(loc='best')  # You can choose a different location for the legend if required
plt.grid(True)  # Optional: Adds a grid
# plt.savefig(f'./results/type_I_error_avg_snr_{snr}.png')
plt.show()
# %%
