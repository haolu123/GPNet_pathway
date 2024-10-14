#%%
import torch
import torch.nn as nn
import torch.optim as optim
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
import pickle
import pandas as pd
# from utls.generate_sim_data import generate_sim_data
from sim_data_generation.sim_data_gen_func import sim_data_gen
import numpy as np
from statsmodels.stats.multitest import multipletests
#%%
def custom_train_test_split(X, y, test_size=0.25, shuffle=True, random_state=None):
    """
    Split arrays or matrices into random train and test subsets.

    Parameters:
    - X: Features dataset (numpy array or list)
    - y: Labels dataset (numpy array or list)
    - test_size: Proportion of the dataset to include in the test split (float, between 0.0 and 1.0)
    - shuffle: Whether or not to shuffle the data before splitting (default=True)
    - random_state: Seed for random number generator (for reproducibility)

    Returns:
    - X_train: Training features
    - X_test: Test features
    - y_train: Training labels
    - y_test: Test labels
    """
    
    # Convert input to numpy arrays if they aren't already
    X = np.array(X)
    y = np.array(y)
    
    # Ensure both X and y have the same length
    assert len(X) == len(y), "The number of samples in X and y must be equal."
    
    # Set the random seed for reproducibility, if provided
    if random_state is not None:
        np.random.seed(random_state)
    
    # Get the number of samples
    num_samples = len(X)
    
    # Shuffle the data
    if shuffle:
        indices = np.arange(num_samples)
        np.random.shuffle(indices)
        X = X[indices]
        y = y[indices]
    
    # Compute the split index
    split_index = int(num_samples * (1 - test_size))
    
    # Split the data into training and test sets
    X_train, X_test = X[:split_index], X[split_index:]
    y_train, y_test = y[:split_index], y[split_index:]
    
    return X_train, X_test, y_train, y_test

class CustomStandardScaler:
    def __init__(self):
        self.mean_ = None
        self.scale_ = None

    def fit(self, X):
        """
        Compute the mean and standard deviation for each feature in the dataset X.
        
        Parameters:
        - X: Input dataset (numpy array or list)

        Returns:
        - self: The fitted scaler
        """
        X = np.array(X)
        self.mean_ = np.mean(X, axis=0)
        self.scale_ = np.std(X, axis=0)
        return self

    def transform(self, X):
        """
        Standardize the dataset by subtracting the mean and dividing by the standard deviation.
        
        Parameters:
        - X: Input dataset (numpy array or list)

        Returns:
        - X_scaled: The standardized dataset
        """
        X = np.array(X)
        if self.mean_ is None or self.scale_ is None:
            raise Exception("The scaler has not been fitted yet. Call 'fit' first.")
        # print(np.where(np.isnan(self.mean_)))
        # print(np.where(np.isnan(self.scale_)))
        return (X - self.mean_) / (self.scale_+0.00001)

    def fit_transform(self, X):
        """
        Fit to data, then transform it.
        
        Parameters:
        - X: Input dataset (numpy array or list)

        Returns:
        - X_scaled: The standardized dataset
        """
        return self.fit(X).transform(X)
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

class PASNet(nn.Module):
	def __init__(self, In_Nodes, Out_Nodes, Pathway_Nodes=200, Hidden_Nodes=64):
		super(PASNet, self).__init__()
		self.sigmoid = nn.Sigmoid()
		self.softmax = nn.Softmax(dim = 1)

		###gene layer --> pathway layer
		self.sc1 = nn.Linear(In_Nodes, Pathway_Nodes)
		###pathway layer --> hidden layer
		self.sc2 = nn.Linear(Pathway_Nodes, Hidden_Nodes)
		###hidden layer --> Output layer
		self.sc3 = nn.Linear(Hidden_Nodes, Out_Nodes)
		###randomly select a small sub-network
		self.do_m1 = torch.ones(Pathway_Nodes)
		self.do_m2 = torch.ones(Hidden_Nodes)
		###if gpu is being used
		if torch.cuda.is_available():
			self.do_m1 = self.do_m1.cuda()
			self.do_m2 = self.do_m2.cuda()
		###

	def forward(self, x):
		###force the connections between gene layer and pathway layer w.r.t. 'pathway_mask'
		self.sc1.weight.data = self.sc1.weight.data
		x = self.sigmoid(self.sc1(x))
		if self.training == True: ###construct a small sub-network for training only
			x = x.mul(self.do_m1)
		x = self.sigmoid(self.sc2(x))
		if self.training == True: ###construct a small sub-network for training only
			x = x.mul(self.do_m2)
		x = self.softmax(self.sc3(x)) # all rows add up to 1

		return x    
#%%
with open('./sim_data_generation/G_list.pkl', 'rb') as f:
    G_list = pickle.load(f)
with open('./sim_data_generation/relations_list.pkl', 'rb') as f:
    relations_list = pickle.load(f)
with open('./sim_data_generation/graphics_dict_list.pkl', 'rb') as f:
    graphics_dict_list = pickle.load(f)

count_df_in = pd.read_csv('./sim_data_generation/df_raw_unstranded.csv')

# Zero the mean of each row of count_df
count_df_in = count_df_in.set_index('gene_name')
# delete the gene_id column
count_df_in = count_df_in.drop('gene_id', axis=1)
if count_df_in.index.duplicated().any():
        print("Duplicate index labels found. Aggregating duplicates.")
        count_df_in = count_df_in.groupby(count_df_in.index).max()

array = count_df_in.values

def shuffle_array(arr, axis='rows'):
    """
    Shuffle array in-place along the specified axis.
    
    Parameters:
    - arr: NumPy array to be shuffled.
    - axis: 'rows' to shuffle rows, 'columns' to shuffle columns, 'both' to shuffle both.
    """
    if axis in ['both', 'rows']:
        # Shuffle each row
        for row in arr:
            np.random.shuffle(row)

    if axis in ['both', 'columns']:
        # Shuffle each column, operate on the transposed array
        arr_T = arr.T
        for row in arr_T:
            np.random.shuffle(row)
        arr[:] = arr_T.T
shuffle_array(array)
count_df_in = pd.DataFrame(array, index=count_df_in.index, columns=count_df_in.columns)
# Apply the shuffle function to each row
# count_df = count_df.apply(shuffle_row, axis=1)
#%%
SNR=0.1
SNR_list = [1,0.5,0.1,0.05,0.01,0.005,0.001]
sample_size_list = [3, 5, 10, 50, 100, 150, 200, 250]

type_I_error_avg_list_snr = []
power_avg_list_snr = []
for SNR in SNR_list:
    SNR = SNR * 3
    type_I_error_avg_list_sample_size = []
    power_avg_list_sample_size = []
    for sample_size in sample_size_list:
        # random select '2*sample_size' smaples from count_df_in 
        count_df_in_temp = count_df_in.sample(2*sample_size, axis=1)



        type_I_error_sum = 0
        power_sum = 0
        for loop_i in range(20):
            count_df, count_data, labels, pathway_list = sim_data_gen(count_df_in_temp, G_list, graphics_dict_list, SNR=SNR, sigma=1, pathway_num=1)

            gene_list = graphics_dict_list

            # get device
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            # device = torch.device('cpu')
            p_values = []
            fdr = []
            for idx_gene_list in range(len(gene_list)):
                gene_list_i = gene_list[idx_gene_list]
                gene_list_i = list(gene_list_i.values())
                filtered_gene_list = [gene for gene in gene_list_i if gene in count_df.index]
                try_data = count_df.loc[filtered_gene_list]
                try_data = try_data.to_numpy()
                
                X_train, X_val, y_train, y_val = custom_train_test_split(try_data.T, labels, test_size=0.2, random_state=None)
                # Standardize features by removing the mean and scaling to unit variance
                scaler = CustomStandardScaler()
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
                model = PASNet(input_dim, output_dim)
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
            # # %%
            # p_value_df = pd.DataFrame(p_values)
            # p_value_df.sort_values(by='p_value', ascending=True, inplace=True)
            # p_value_df

            gene_set_label = np.array([i in pathway_list for i in range(len(gene_list))])
            p_adj = multipletests(fdr, alpha=0.05, method='fdr_bh')
            rejected, p_values_corrected, sidak, bonferroni = p_adj
            threshold_list = np.array(sorted(p_values_corrected))
            threshold_list = np.concatenate(([-1], threshold_list, [1.1]))
            for threshold in threshold_list:
                predictions = np.array(p_values_corrected) <= threshold
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

            for threshold in threshold_list[::-1]:
                predictions = np.array(p_values_corrected) <= threshold

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
        power_avg = power_sum / 20
        type_I_error_avg = type_I_error_sum / 20
        power_avg_list_sample_size.append(power_avg)
        type_I_error_avg_list_sample_size.append(type_I_error_avg)
    power_avg_list_snr.append(power_avg_list_sample_size)
    type_I_error_avg_list_snr.append(type_I_error_avg_list_sample_size)
# %%
print(f'PASNet_real: type_I_error_avg_list_snr_PASNet_real: {type_I_error_avg_list_snr}')
print(f'PASNet_real: power_avg_list_snr: {power_avg_list_snr}')
with open('./results/type_I_error_avg_list_snr_PASNet_real.pkl', 'wb') as f:
    pickle.dump(type_I_error_avg_list_snr, f)
with open('./results/power_avg_list_snr_PASNet_real.pkl', 'wb') as f:
    pickle.dump(power_avg_list_snr, f)
# %%
