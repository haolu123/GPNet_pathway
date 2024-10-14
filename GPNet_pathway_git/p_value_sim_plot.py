#%%
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
import pickle
import pandas as pd
from utls.generate_sim_data import generate_sim_data
import numpy as np
from statsmodels.stats.multitest import multipletests
#%%
class DeepFNN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DeepFNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 200)  # First hidden layer
        self.fc2 = nn.Linear(200, 64)                # Second hidden layer
        self.fc3 = nn.Linear(64, output_dim)   # Output layer
        self.relu = nn.ReLU()                        # Activation function

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x
#%%    
# with open('./utls/combine_data.pkl', 'rb') as f:
#     combine_data = pickle.load(f)

# gene_list = pd.read_csv('./utls/gene_list.csv', index_col=0)
# with open('./utls/labels.pkl', 'rb') as f:
#     labels = pickle.load(f)

# with open('./utls/gene_list_select.pkl', 'rb') as f:
#     gene_list_select = pickle.load(f)
parameters = {
    'gene_num': 10000,
    'num_sample_one_label': 500,
    'select_num': 5,
    'gene_list_num': 50,
    'n_mean': 0.05,
    'n_std': 1,
    'selected_gene_num': 10
}
n_mean_list = [1,0.5,0.1,0.05,0.01,0.005,0.001]
p_value_all = {}
for n_mean in n_mean_list:
    parameters['n_mean'] = n_mean
    sample_size_list = [3, 5, 10, 50, 100,150,200,250]
    p_value_all[n_mean] = {}
    for sample_size in sample_size_list:
        parameters['num_sample_one_label'] = sample_size
        p_value_all[n_mean][sample_size] = {'diff': [], 'same': []}
        for loop_idx in range(10):
            combine_data, _, _, gene_list, labels, gene_list_select = generate_sim_data(parameters)

            # get device
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            p_values = []
            fdr = []
            for idx_gene_list in range(len(gene_list)):
                array_list2 = gene_list.to_numpy()
                try_data = combine_data[array_list2[idx_gene_list],:]

                # Assuming 'try_data' is a NumPy array and 'labels' is also a NumPy array
                X_train, X_val, y_train, y_val = train_test_split(try_data.T, labels, test_size=0.2, random_state=None, stratify=labels)

                # Standardize features by removing the mean and scaling to unit variance
                scaler = StandardScaler()
                X_train = scaler.fit_transform(X_train)
                X_val = scaler.transform(X_val)

                # Convert arrays to PyTorch tensors
                X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
                X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
                y_train_tensor = torch.tensor(y_train, dtype=torch.long)
                y_val_tensor = torch.tensor(y_val, dtype=torch.long)

                # Create datasets
                train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
                val_dataset = TensorDataset(X_val_tensor, y_val_tensor)

                # Create data loaders
                train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
                val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

                # set model
                input_dim = X_train.shape[1]
                output_dim = 2
                model = DeepFNN(input_dim, output_dim)
                model.to(device)
                criterion = nn.CrossEntropyLoss()  # Suitable for classification tasks
                optimizer = optim.Adam(model.parameters(), lr=0.001)

                # Define the number of epochs
                epochs = 50
                model.train()
                for epoch in range(epochs):
                    for data, target in train_loader:
                        data, target = data.to(device), target.to(device)
                        optimizer.zero_grad()           # Clear gradients for next train
                        output = model(data)            # Forward pass
                        loss = criterion(output, target)# Calculate loss
                        loss.backward()                 # Backward pass
                        optimizer.step()                # Update weights
                correct = 0
                total = 0

                model.eval()
                with torch.no_grad():  # For evaluation, we don't need gradients
                    confusion_matrix = torch.zeros(output_dim, output_dim)
                    for data, target in val_loader:
                        data, target = data.to(device), target.to(device)
                        output = model(data)
                        _, predicted = torch.max(output.data, 1)
                        total += target.size(0)
                        correct += (predicted == target).sum().item()
                        for t, p in zip(target.view(-1), predicted.view(-1)):
                            confusion_matrix[t.long(), p.long()] += 1
                print(f'Accuracy of the network on the validation data: {100 * correct // total}%')
                print(f'Confusion matrix:\n{confusion_matrix}')

                from utls.train import compute_p_value_for_acc
                cdf_value_0, cdf_value_1, p_value = compute_p_value_for_acc(confusion_matrix)

                p_values.append({'cdf_value_0': cdf_value_0, 'cdf_value_1': cdf_value_1, 'p_value': p_value})
                fdr.append(p_value)
                if idx_gene_list in gene_list_select:
                    p_value_all[n_mean][sample_size]['diff'].append(p_value)
                else:
                    p_value_all[n_mean][sample_size]['same'].append(p_value)
# %%
with open('./results/p_value_diff_vs_same.pkl', 'wb') as f:
    pickle.dump(p_value_all, f)


    #         p_adj = multipletests(fdr, alpha=0.05, method='fdr_bh')
    #         rejected, p_values_corrected, sidak, bonferroni = p_adj
    #         threshold_list = np.array(sorted(p_values_corrected))

    #         for threshold in threshold_list:
    #             predictions = np.array(p_values_corrected) <= threshold

    #             #calculate TP, TN, FP, FN
    #             TP = np.sum((predictions == True) & (gene_set_label == True))
    #             TN = np.sum((predictions == False) & (gene_set_label == False))
    #             FP = np.sum((predictions == True) & (gene_set_label == False))
    #             FN = np.sum((predictions == False) & (gene_set_label == True))

    #             # calculate Type II Error
    #             type_II_error = FN / (TP + FN) if (TP + FN) > 0 else 0
    #             if type_II_error > 0.1:
    #                 continue
    #             else:
    #                 type_I_error = FP / (FP + TN) if (FP + TN) > 0 else 0
    #                 break
    #         print("Type I Error (False Positive Rate):", type_I_error)
    #         print("threshold:", threshold)
    #         type_I_error_sum += type_I_error
    #         power = 0
    #         for threshold in threshold_list[::-1]:
    #             predictions = np.array(p_values_corrected) <= threshold

    #             #calculate TP, TN, FP, FN
    #             TP = np.sum((predictions == True) & (gene_set_label == True))
    #             TN = np.sum((predictions == False) & (gene_set_label == False))
    #             FP = np.sum((predictions == True) & (gene_set_label == False))
    #             FN = np.sum((predictions == False) & (gene_set_label == True))

    #             # calculate Type I Error
    #             type_I_error = FP / (FP + TN) if (FP + TN) > 0 else 0
    #             if type_I_error > 0.1:
    #                 continue
    #             else:
    #                 power = TP / (TP+FN) if (TP+FN) > 0 else 0
    #                 break
    #         print("Power (True Positive Rate):", power)
    #         print("threshold:", threshold)
    #         power_sum += power
    #     power_avg = power_sum / 10
    #     type_I_error_avg = type_I_error_sum / 10
    #     power_avg_list_sample_size.append(power_avg)
    #     type_I_error_avg_list_sample_size.append(type_I_error_avg)
    # power_avg_list_snr.append(power_avg_list_sample_size)
    # type_I_error_avg_list_snr.append(type_I_error_avg_list_sample_size)

# %%
# with open('./results/type_I_error_list_n_mean.pkl', 'wb') as f:
#     pickle.dump(type_I_error_list_n_mean, f)
# with open('./results/type_II_error_list_n_mean.pkl', 'wb') as f:
#     pickle.dump(type_II_error_list_n_mean, f)
# with open('./results/accuracy_list_n_mean.pkl', 'wb') as f:
#     pickle.dump(accuracy_list_n_mean, f)
# %%
# with open('./results/type_I_error_avg_list_snr_sim_cls.pkl', 'wb') as f:
#     pickle.dump(type_I_error_avg_list_snr, f)
# with open('./results/power_avg_list_snr_sim_cls.pkl', 'wb') as f:
#     pickle.dump(power_avg_list_snr, f)

# %%
sample_size_list = [150,200]
for sample_size in sample_size_list:
    list1 = p_value_all[0.05][sample_size]['diff']
    list2 = p_value_all[0.05][sample_size]['same']
    frequency1, bins1 = np.histogram(list1, bins=20, range=(0, 1), density=True)
    frequency2, bins2 = np.histogram(list2, bins=20, range=(0, 1), density=True)
    frequency1 = frequency1 / np.sum(frequency1)
    frequency2 = frequency2 / np.sum(frequency2)
    import matplotlib.pyplot as plt
    # Bin widths (assuming uniform bin width)
    bin_width1 = bins1[1] - bins1[0]
    bin_width2 = bins2[1] - bins2[0]

    # Plotting
    plt.bar(bins1[:-1], frequency1, width=bin_width1, alpha=0.7, label='different expression', align='edge')
    plt.bar(bins2[:-1], frequency2, width=bin_width2, alpha=0.7, label='same expression', align='edge')
    # Adding a vertical dashed line at x=0.05
    plt.axvline(x=0.05, color='gray', linestyle='--', label='Threshold (x=0.05)')

    plt.xlabel('p-value')
    plt.ylabel('Frequency')
    plt.legend()
    plt.show()
# %%
