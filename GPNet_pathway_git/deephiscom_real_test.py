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
import numpy as np
# from statsmodels.stats.multitest import multipletests
from sim_data_generation.sim_data_gen_func import sim_data_gen
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

act_fn = nn.LeakyReLU(0.2)
dropout_fn = nn.Dropout(0.5)
def init_weights(m):
    if isinstance(m, nn.Linear):
        m.weight.data.fill_(0.01)
class pathwayblock(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_num):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.layer_num = layer_num
        if hidden_dim == 0:
            self.block = nn.Sequential(
                nn.Linear(input_dim, 1, bias = False),
                act_fn,
                dropout_fn
            )
        else: #layer_num = btw pathway and biological factor number. if layer_num ==0, then hidden num should be zero for setting. we do not allow =0
            modules = []
            modules.append(nn.Linear(input_dim, hidden_dim, bias = False))
            modules.append(act_fn)
            modules.append(dropout_fn)
            for i in range(layer_num-1):
                modules.append(nn.Linear(hidden_dim, hidden_dim, bias = False))
                modules.append(act_fn)
                modules.append(dropout_fn)
            modules.append(nn.Linear(hidden_dim, 1, bias = False))
            modules.append(act_fn)
            modules.append(dropout_fn)
            self.block = nn.Sequential(*modules)
        self.block.apply(init_weights)

    def forward(self, x):
        return self.block(x)



class DeepHisCoM(nn.Module):
    def __init__(self,  nvar, width, layer, covariate, device):
        super(DeepHisCoM, self).__init__()
        self.nvar = nvar
        self.width = width
        self.layer = layer 
        self.pathway_nn = nn.ModuleList([pathwayblock(nvar[i], width[i], layer[i]) for i in range(len(self.nvar))])
        self.bn_path = nn.BatchNorm1d(len(nvar))
        self.dropout_path = dropout_fn
        self.covariate = covariate
        self.fc_path_disease=nn.Linear(len(nvar) +covariate ,1)
        self.fc_path_disease.weight.data.fill_(0)
        self.fc_path_disease.bias.data.fill_(0.001)
        self.device = device

    def forward(self, x):
        kk=0
        nvarlist = list()
        nvarlist.append(kk)
        for i in range(len(self.nvar)):
            k=kk
            kk=kk+self.nvar[i]
            nvarlist.append(kk)
        nvarlist.append(kk + self.covariate)
        pathway_layer = torch.cat([self.pathway_nn[i](x[:,nvarlist[i]:nvarlist[i+1]]) for i in range(len(self.nvar))],1)
        pathway_layer = self.bn_path(pathway_layer)
        pathway_layer = pathway_layer/(torch.norm(pathway_layer,2))
        x = torch.cat([pathway_layer, x[:, nvarlist[len(self.nvar)]:nvarlist[len(self.nvar) + 1]]], 1)
        x = self.dropout_path(x)
        x = self.fc_path_disease(x)
        x = torch.sigmoid(x)
        return(x)
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
#%%
SNR=0.1
SNR_list = [1,0.5,0.1,0.05,0.01,0.005,0.001]
sample_size_list = [3, 5, 10, 50, 100, 150, 200, 250]
p_value_all = {}
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
for SNR in SNR_list:
    p_value_all[SNR] = {}
    for sample_size in sample_size_list:
        count_df_in_temp = count_df_in.sample(2*sample_size, axis=1)
        p_value_all[SNR][sample_size] = {'diff': [], 'same': []}
        param_save_list = []
        for permutation in range(0, 1000):
            count_df, count_data, labels, pathway_list = sim_data_gen(count_df_in_temp, G_list, graphics_dict_list, SNR=SNR, sigma=1, pathway_num=1)
            gene_list = graphics_dict_list
            X_train, X_val, y_train, y_val = custom_train_test_split(count_data.T, labels, test_size=0.2, random_state=None)
            X_train = np.nan_to_num(X_train)
            X_val = np.nan_to_num(X_val)
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

            nvar = [200]*50
            node_num = [50]*50
            layer_num = [1]*50
            model_DeepHisCoM = DeepHisCoM(nvar, node_num, layer_num, 0, device)
            model_DeepHisCoM.to(device)
            criterion = nn.CrossEntropyLoss()  # Suitable for classification tasks
            optimizer = optim.Adam(model_DeepHisCoM.parameters(), lr=0.001)

            epochs = 50
            model_DeepHisCoM.train()
            for epoch in range(epochs):
                for data, target in train_loader:
                    data, target = data.to(device), target.to(device)
                    optimizer.zero_grad()           # Clear gradients for next train
                    output = model_DeepHisCoM(data)            # Forward pass
                    loss = criterion(output.view(-1), torch.randint(0,2,output.shape).view(-1).float().cuda())# Calculate loss
                    loss.backward()                 # Backward pass
                    optimizer.step()                # Update weights
            param_save = np.array(list(model_DeepHisCoM.fc_path_disease.parameters())[0].tolist()[0])
            param_save_list.append(param_save)

        count_df, count_data, labels, pathway_list = sim_data_gen(count_df_in_temp, G_list, graphics_dict_list, SNR=SNR, sigma=1, pathway_num=1)
        X_train, X_val, y_train, y_val = custom_train_test_split(count_data.T, labels, test_size=0.2, random_state=None)

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

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



        nvar = [50]*20
        node_num = [50]*20
        layer_num = [1]*20
        model_DeepHisCoM = DeepHisCoM(nvar, node_num, layer_num, 0, device)
        model_DeepHisCoM.to(device)
        criterion = nn.CrossEntropyLoss()  # Suitable for classification tasks
        optimizer = optim.Adam(model_DeepHisCoM.parameters(), lr=0.001)

        epochs = 50
        model_DeepHisCoM.train()
        for epoch in range(epochs):
            for data, target in train_loader:
                data, target = data.to(device), target.to(device)
                optimizer.zero_grad()           # Clear gradients for next train
                output = model_DeepHisCoM(data)            # Forward pass
                loss = criterion(output.view(-1), target.float())# Calculate loss
                loss.backward()                 # Backward pass
                optimizer.step()                # Update weights
        param = np.array(list(model_DeepHisCoM.fc_path_disease.parameters())[0].tolist()[0])
        param_save_matrix = np.vstack(param_save_list)

        # Calculate the percentage of elements in each column of param_save_matrix 
        # that are greater than the corresponding elements in param_save
        p_values = np.mean(param_save_matrix > param_save, axis=0)
        for i in range(len(p_values)):
            if i in pathway_list:
                p_value_all[SNR][sample_size]['diff'].append(p_values[i])
            else:
                p_value_all[SNR][sample_size]['same'].append(p_values[i])

    

# %%
with open('./results/p_value_diff_vs_same_DeepHisCom_real.pkl', 'wb') as f:
    pickle.dump(p_value_all, f)
# %%
type_I_error_avg_list_snr = []
power_avg_list_snr = []
for n_mean in SNR_list:
    type_I_error_avg_list_sample_size=[]
    power_avg_list_sample_size = []
    for sample_size in sample_size_list:
        p_value_togethor = p_value_all[n_mean][sample_size]['diff'] + p_value_all[n_mean][sample_size]['same']
        threshold_list = np.array(sorted(p_value_togethor))
        type_I_error_sum = 0
        power_sum = 0
        for threshold in threshold_list:
            predictions_diff = np.array(p_value_all[n_mean][sample_size]['diff']) <= threshold
            predictions_same = np.array(p_value_all[n_mean][sample_size]['same']) <= threshold

            TP = np.sum((predictions_diff == True))
            TN = np.sum((predictions_same == False))
            FP = np.sum((predictions_same == True))
            FN = np.sum((predictions_diff == False))
            # calculate Type II Error
            type_II_error = FN / (TP + FN) if (TP + FN) > 0 else 0
            if type_II_error > 0.1:
                continue
            else:
                type_I_error = FP / (FP + TN) if (FP + TN) > 0 else 0
                break
        print("Type I Error (False Positive Rate):", type_I_error)
        print("threshold:", threshold)
        type_I_error_sum += type_I_error
        # if type_I_error_avg_list_sample_size != []:
        #     if type_I_error_sum > type_I_error_avg_list_sample_size[-1]:
        #         type_I_error_sum = type_I_error_avg_list_sample_size[-1]
        power = 0
        for threshold in threshold_list[::-1]:
            predictions_diff = np.array(p_value_all[n_mean][sample_size]['diff']) <= threshold
            predictions_same = np.array(p_value_all[n_mean][sample_size]['same']) <= threshold

            #calculate TP, TN, FP, FN
            TP = np.sum((predictions_diff == True))
            TN = np.sum((predictions_same == False))
            FP = np.sum((predictions_same == True))
            FN = np.sum((predictions_diff == False))

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
        # if power_avg_list_sample_size != []:
        #     if power_sum < power_avg_list_sample_size[-1]:
        #         power_sum = power_avg_list_sample_size[-1]
        power_avg_list_sample_size.append(power_sum)
        type_I_error_avg_list_sample_size.append(type_I_error_sum)
    power_avg_list_snr.append(power_avg_list_sample_size)
    type_I_error_avg_list_snr.append(type_I_error_avg_list_sample_size)

# %%
with open('./results/type_I_error_avg_list_snr_sim_cls_HistCom_real.pkl', 'wb') as f:
    pickle.dump(type_I_error_avg_list_snr, f)
with open('./results/power_avg_list_snr_sim_cls_HistCom_real.pkl', 'wb') as f:
    pickle.dump(power_avg_list_snr, f)
# %%
