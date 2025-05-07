import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import h5py
import time
import random
import os
import sys
from glob import glob
from datetime import datetime
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import scipy.signal as signal
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torch.nn.functional as F
from torch.utils.data import random_split, DataLoader, TensorDataset, Dataset


# Check if a GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)


class QuakeXNet_1d(nn.Module):
    def __init__(self, num_classes=4, num_channels=3, dropout_rate=0.2):
        super(QuakeXNet_1d, self).__init__()
        # Define the layers of the CNN architecture
        self.conv1 = nn.Conv1d(in_channels=num_channels, out_channels=8, kernel_size=9, stride=1, padding=4)
        self.conv2 = nn.Conv1d(in_channels=8, out_channels=8, kernel_size=9, stride=2, padding=4)
        self.conv3 = nn.Conv1d(in_channels=8, out_channels=16, kernel_size=7, stride=1, padding=3)
        self.conv4 = nn.Conv1d(in_channels=16, out_channels=16, kernel_size=7, stride=2, padding=3)
        self.conv5 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=5, stride=1, padding=2)
        self.conv6 = nn.Conv1d(in_channels=32, out_channels=32, kernel_size=5, stride=2, padding=2)
        self.conv7 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)
        
        # Batch-normalization layers
        self.bn1 = nn.BatchNorm1d(8)
        self.bn2 = nn.BatchNorm1d(8)
        self.bn3 = nn.BatchNorm1d(16)
        self.bn4 = nn.BatchNorm1d(16)
        self.bn5 = nn.BatchNorm1d(32)
        self.bn6 = nn.BatchNorm1d(32)
        self.bn7 = nn.BatchNorm1d(64)
        
        # Dynamically calculate the size of the first fully connected layer
        self.fc_input_size = self._get_conv_output_size(num_channels, input_length=5000)
        self.fc1 = nn.Linear(self.fc_input_size, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.fc1_bn = nn.BatchNorm1d(128)
        self.fc2_bn = nn.BatchNorm1d(num_classes)
        
        # Define dropout
        self.dropout = nn.Dropout(dropout_rate)
        
    def _get_conv_output_size(self, num_channels, input_length):
        # Forward pass a dummy input through the conv layers to get the output size
        dummy_input = torch.randn(1, num_channels, input_length)
        with torch.no_grad():
            x = F.relu(self.bn1(self.conv1(dummy_input)))
            x = self.pool1(F.relu(self.bn2(self.conv2(x))))
            x = F.relu(self.bn3(self.conv3(x)))
            x = self.pool1(F.relu(self.bn4(self.conv4(x))))
            x = F.relu(self.bn5(self.conv5(x)))
            x = self.pool1(F.relu(self.bn6(self.conv6(x))))
            x = F.relu(self.bn7(self.conv7(x)))
        return x.numel()

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool1(F.relu(self.bn2(self.conv2(x))))
        x = self.dropout(x)
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool1(F.relu(self.bn4(self.conv4(x))))
        x = self.dropout(x)
        x = F.relu(self.bn5(self.conv5(x)))
        x = self.pool1(F.relu(self.bn6(self.conv6(x))))
        x = self.dropout(x)
        x = F.relu(self.bn7(self.conv7(x)))
        x = x.view(x.size(0), -1)  # Flatten before fully connected layer
        x = self.dropout(x)
        x = F.relu(self.fc1_bn(self.fc1(x)))
        x = self.fc2_bn(self.fc2(x))
        return x
   
class QuakeXNet_2d(nn.Module):
    def __init__(self, num_classes=4, num_channels=3, dropout_rate=0.2):
        super(QuakeXNet_2d, self).__init__()
        
        # Define the layers of the CNN architecture
        self.conv1 = nn.Conv2d(in_channels=num_channels, out_channels=8, kernel_size=(3, 3), stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=8, kernel_size=(3, 3), stride=2, padding=1)
        self.conv3 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(3, 3), stride=1, padding=1)
        self.conv4 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(3, 3), stride=2, padding=1)
        self.conv5 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3, 3), stride=1, padding=1)
        self.conv6 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), stride=2, padding=1)
        self.conv7 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), stride=1, padding=1)
        
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Batch-normalization layers
        self.bn1 = nn.BatchNorm2d(8)
        self.bn2 = nn.BatchNorm2d(8)
        self.bn3 = nn.BatchNorm2d(16)
        self.bn4 = nn.BatchNorm2d(16)
        self.bn5 = nn.BatchNorm2d(32)
        self.bn6 = nn.BatchNorm2d(32)
        self.bn7 = nn.BatchNorm2d(64)

        # Calculate the input size for the fully connected layer dynamically
        self.fc_input_size = self._get_conv_output_size(num_channels, (129, 38))
        self.fc1 = nn.Linear(self.fc_input_size, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.fc1_bn = nn.BatchNorm1d(128)
        self.fc2_bn = nn.BatchNorm1d(num_classes)

        # Define dropout
        self.dropout = nn.Dropout(dropout_rate)

    def _get_conv_output_size(self, num_channels, input_dims):
        # Forward pass a dummy input through the conv layers to get the output size
        dummy_input = torch.randn(1, num_channels, *input_dims)
        with torch.no_grad():
            x = F.relu(self.bn1(self.conv1(dummy_input)))
            x = self.pool1(F.relu(self.bn2(self.conv2(x))))
            x = F.relu(self.bn3(self.conv3(x)))
            x = self.pool1(F.relu(self.bn4(self.conv4(x))))
            x = F.relu(self.bn5(self.conv5(x)))
            x = F.relu(self.bn6(self.conv6(x)))
            x = F.relu(self.bn7(self.conv7(x)))
        return x.numel()

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))  # output size: (8, 129, 38)
        x = self.pool1(F.relu(self.bn2(self.conv2(x))))  # output size: (8, 64, 19)
        x = self.dropout(x)
        
        x = F.relu(self.bn3(self.conv3(x)))  # output size: (16, 64, 19)
        x = self.pool1(F.relu(self.bn4(self.conv4(x))))  # output size: (16, 32, 10)
        x = self.dropout(x)
        
        x = F.relu(self.bn5(self.conv5(x)))  # output size: (32, 32, 10)
        x = F.relu(self.bn6(self.conv6(x)))  # output size: (32, 16, 5)
        x = self.dropout(x)
        
        x = F.relu(self.bn7(self.conv7(x)))  # output size: (64, 16, 5)
        
        x = x.view(x.size(0), -1)  # Flatten before fully connected layer
        x = self.dropout(x)
        
        x = F.relu(self.fc1_bn(self.fc1(x)))  # classifier
        x = self.fc2_bn(self.fc2(x))  # classifier
        
        # Do not apply softmax here, as it will be applied in the loss function
        return x

class SeismicCNN_1d(nn.Module):
    def __init__(self, num_classes=4, num_channels = 3,dropout_rate=0.2):
        super(SeismicCNN_1d, self).__init__()
        # Define the layers of the CNN architecture
        self.conv1 = nn.Conv1d(in_channels= num_channels, out_channels=32, kernel_size=5)
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=5)
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.fc1 = nn.Linear(79808, 128)  # Adjust input size based on your data
        self.fc2 = nn.Linear(128, num_classes)
        self.softmax = nn.Softmax(dim=1)

        self.dropout = nn.Dropout(dropout_rate)

        self.bn1 = nn.BatchNorm1d(32)#, dtype=torch.float64)
        self.bn2 = nn.BatchNorm1d(64)#, dtype=torch.float64)
        self.fc1_bn = nn.BatchNorm1d(128)
        self.fc2_bn = nn.BatchNorm1d(num_classes)

    def forward(self, x):

        # x = self.pool1(F.relu(self.bn2(self.conv2(x)))) # feature extraction, output size of 8,1250 
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.dropout(x)
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.dropout(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1_bn(self.fc1(x)))
        x = self.dropout(x)
        x = self.fc2_bn(self.fc2(x))
        # do not apply softmax here, as it will be applied in the loss function
        return x
   
class SeismicCNN_2d(nn.Module):
    def __init__(self, num_classes=4, num_channels=3, dropout_rate=0.2):
        super(SeismicCNN_2d, self).__init__()
        
        # Define the layers of the CNN architecture for 2D spectrograms
        self.conv1 = nn.Conv2d(in_channels=num_channels, out_channels=32, kernel_size=(3, 3))
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3))
        self.pool = nn.MaxPool2d(kernel_size=(2, 2))
        
        # Calculate the size of the input to the first fully connected layer.
        conv_output_size = 64 * 30 * 8  # this must be adjusted based on the actual input size

        self.fc1 = nn.Linear(conv_output_size, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.softmax = nn.Softmax(dim=1)

        self.dropout = nn.Dropout(dropout_rate)

        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.fc1_bn = nn.BatchNorm1d(128)
        self.fc2_bn = nn.BatchNorm1d(num_classes)

    def forward(self, x):
        # Apply 2D convolution, batch normalization, ReLU, pooling, and dropout
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.dropout(x)
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.dropout(x)
        
        # Flatten the output for the fully connected layers
        x = x.view(x.size(0), -1)
        
        # Apply fully connected layers with batch normalization, ReLU, and dropout
        x = F.relu(self.fc1_bn(self.fc1(x)))
        x = self.dropout(x)
        x = self.fc2_bn(self.fc2(x))
        
        # Do not apply softmax here, as it will be applied in the loss function
        return x

def plot_confusion_matrix_and_cr(model, test_loader,classes = ['earthquake', 'explosion', 'surface','noise'],fname = '_confusion_matrix.png'):
    
    """
    inputs
    
    model: A trained neural network model in PyTorch. This model will be evaluated on the test data.
    test_loader: A PyTorch DataLoader containing batches of input data (features) and corresponding labels (ground truth) 
    from the test dataset. It iterates through the test set, providing data to the model for inference.
    inputs: 2D or 3D tensors (depending on the model type, usually spectrograms or seismic waveform windows in your case).
    labels: One-hot encoded labels corresponding to the classification categories (e.g., earthquake, explosion, noise, surface event).
    
    outputs
    Confusion Matrix (cm): A NumPy array representing the confusion matrix. It contains the counts of 
    actual vs. predicted labels for all classes (e.g., earthquakes predicted as explosions, etc.). The confusion matrix helps in identifying misclassification patterns.

    classification Report (report): A dictionary (or DataFrame) output from 
    sklearn.metrics.classification_report, containing precision, recall, F1-score, and support for each class. This provides a more comprehensive evaluation by detailing the performance of the model for each individual class.
    """
    
    mname = type(model).__name__

    with torch.no_grad(): # Context-manager that disabled gradient calculation.
        # Loop on samples in test set
        y_pred=[]#np.zeros(len(test_loader)*batch_size)
        y_test=[]#np.zeros(len(test_loader)*batch_size)
        for i,data in enumerate(test_loader):
            print(i,data[1].shape,data[0].shape)
            inputs, labels = data[0].to(device), data[1].to(device)
            inputs = inputs.float()
            labels = labels.float()

            outputs = model(inputs)
            y_pred.extend(outputs.argmax(1).cpu())
            y_test.extend(labels.argmax(1).cpu())
        
        y_pred=np.asarray(y_pred)
        y_test=np.asarray(y_test)
        # confusion matrix
        cm = confusion_matrix(y_test, y_pred)

        plt.figure()
        sns.heatmap(cm, annot=True, cmap='Blues', fmt='d', 
        xticklabels = classes, yticklabels = classes)
        plt.xlabel('Predicted', fontsize = 15)
        plt.ylabel('Actual', fontsize = 15)
        plt.title('Total samples: '+str(len(y_pred)), fontsize = 20)
        plt.savefig(f"./plot/{mname}_confusion_matrix.png", dpi=300, bbox_inches='tight')
        plt.show()
        
        
        # Calculate the classification report
        report = classification_report(y_test, y_pred, output_dict=True)
        # Set a pleasing style
        sns.set_style("whitegrid")
        # Create a figure and axes for the heatmap
        plt.figure()
        ax = sns.heatmap(pd.DataFrame(report).iloc[:3, :4], 
        annot=True, cmap='Blues', xticklabels=classes, vmin=0.5, vmax=1)
        # Set labels and title
        ax.set_xlabel('Metrics', fontsize=15)
        ax.set_ylabel('Classes', fontsize=15)
        ax.set_title('Classification Report', fontsize=18)
        # Adjust layout
        plt.tight_layout()
        plt.savefig(f"./plot/{mname}_classification_report.png", dpi=300, bbox_inches='tight')

        plt.show()
        
def extract_waveforms(cat, file_name, start=-20, input_window_length=100, fs=50, number_data=1000, num_channels=3, shifting=True,
                     lowcut=1, highcut=10):
    """
    Extract waveforms from file with exact number of returned samples.
    """   
    random.seed(1234) # set seed for reproducibility
    cat = cat.sample(frac=1).reset_index(drop=True)
    
    # If number_data is negative, use all data
    if number_data < 0:
        number_data = len(cat)
    
    # Open the file
    f = h5py.File(file_name, 'r')
    
    # Initialize arrays to store results
    x = np.zeros((number_data, 3, int(fs*input_window_length)))
    event_ids = []
    
    # Track the number of valid samples processed
    valid_count = 0
    processed = 0
    
    # Continue processing until we have enough valid samples or run out of data
    while valid_count < number_data and processed < len(cat):
        index = processed
        processed += 1
        
        # try:
        # Extract event ID
        event_id = cat['event_id'].values[index]+'_'+cat['station_network_code'].values[index]+'.'+cat['station_code'].values[index]
        
        # Read data
        bucket, narray = cat.loc[index]['trace_name'].split('$')
        xx, _, _ = iter([int(i) for i in narray.split(',:')])
        data = f['/data/%s' % bucket][xx, :, :]
        
        # Simple check for all-zero data before processing
        if np.mean(np.abs(data[0,0:10])) <= 0 & num_channels == 3:
            # Skip if all-zero data
            continue


        
        # Filter data
        nyquist = 0.5 * cat.loc[index,'trace_sampling_rate_hz']
        low = lowcut / nyquist
        high = highcut / nyquist
        b, a = signal.butter(4, [low, high], btype='band')
        
        # Apply taper and filter
        taper = signal.windows.tukey(data.shape[-1], alpha=0.1)
        data = np.array([np.multiply(taper, row) for row in data])
        filtered_signal = np.array([signal.filtfilt(b, a, row) for row in data])
        
        # Resample
        number_of_samples = int(filtered_signal.shape[1] * fs / cat.loc[index,'trace_sampling_rate_hz'])
        data = np.array([signal.resample(row, number_of_samples) for row in filtered_signal])
        
        # Determine window position
        if event_id.split("_")[1] != "noise":
            if np.isnan(cat.loc[index, 'trace_P_arrival_sample']):
                continue  # Skip if no P arrival
            
            # Random start between P-20 and P-5 if shifting enabled
            if shifting:
                ii = int(np.random.randint(start, -4) * fs)
            else:
                ii = int(start * fs)
            
            istart = int(cat.loc[index, 'trace_P_arrival_sample'] * fs / cat.loc[index,'trace_sampling_rate_hz']) + ii
            iend = istart + int(fs * input_window_length)
            
            # Adjust window if it goes out of bounds
            if istart < 0:
                istart = 0
                iend = int(fs * input_window_length)
            
            if iend > data.shape[-1]:
                istart = istart - (iend - data.shape[-1])
                iend = data.shape[-1]
        else:
            # For noise, start at the beginning
            istart = 0
            iend = istart + int(fs * input_window_length)
        
        # Normalize the data
        mmax = np.std(np.abs(data[:, istart:iend]))
        if mmax <= 0:  # Skip if normalization factor is zero
            continue
            
        # Check if there's useful data (non-zero)
        if np.mean(np.abs(data[:, istart:iend])) <= 0:
            continue
            
        # Store data
        x[valid_count, :, :iend-istart] = data[:, istart:iend] / mmax
        event_ids.append(event_id)
        
        valid_count += 1
        
        if valid_count % 100 == 0:
            print(f"Processed {processed}/{len(cat)}, collected {valid_count}/{number_data}")
            
    

        # except Exception as e:
        #     # Skip problematic data
        #     continue
    
    f.close()
    
    # If we didn't get enough samples, trim the arrays
    if valid_count < number_data:
        print(f"Warning: Only found {valid_count} valid samples out of {number_data} requested")
        x = x[:valid_count]
    
    # Convert event_ids to numpy array
    event_ids = np.array(event_ids)
    
    # Extract Z component only if requested
    if num_channels == 1:
        x2 = x[:, 2, :]
        del x        
        x = x2.reshape(x2.shape[0], 1, x2.shape[1])
    
    return x, event_ids


def extract_spectrograms(waveforms, fs, nperseg=256, overlap=0.5):
    noverlap = int(nperseg * overlap)  # Calculate overlap

    # Example of how to get the shape of one spectrogram
    f, t, Sxx = signal.spectrogram(waveforms[0, 0], nperseg=nperseg, noverlap=noverlap, fs=fs)

    # Initialize an array of zeros with the shape: (number of waveforms, channels, frequencies, time_segments)
    spectrograms = np.zeros((waveforms.shape[0], waveforms.shape[1], len(f), len(t)))

    for i in tqdm(range(waveforms.shape[0])):  # For each waveform
        for j in range(waveforms.shape[1]):  # For each channel
            _, _, Sxx = signal.spectrogram(waveforms[i, j], nperseg=nperseg, noverlap=noverlap, fs=fs)
            spectrograms[i, j] = Sxx  # Fill the pre-initialized array

    return spectrograms



def plot_model_training(loss_time, val_loss_time, val_accuracy_time,test_loss,test_acc, title = 'SeismicCNN 1D'):


    # Assuming loss_time, val_loss_time, val_accuracy_time, test_loss, test_accuracy are defined

    NN = np.count_nonzero(loss_time)
    fig, ax1 = plt.subplots(figsize=(8, 6))  # Increase figure size for better readability

    # Set font sizes
    plt.style.use('default')
    plt.rc('font', size=12)  # Global font size
    plt.rc('axes', titlesize=14)  # Title font size
    plt.rc('axes', labelsize=12)  # Axis label font size
    plt.rc('xtick', labelsize=10)  # X-axis tick label font size
    plt.rc('ytick', labelsize=10)  # Y-axis tick label font size

    # Plot Training and Validation Loss
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.plot(np.arange(1, NN), loss_time[:NN-1], color='tab:red', label='Training Loss', linewidth=2)
    ax1.plot(np.arange(1, NN), val_loss_time[:NN-1], color='tab:blue', label='Validation Loss', linewidth=2)
    #ax1.plot(NN+1, test_loss, 'p', color='tab:blue', label='Test Loss', markersize=10)
    ax1.set_ylim(0, 2)
    ax1.grid(True, linestyle='--', alpha=0.6)  # Add grid with some transparency

    # Twin axis for accuracy
    ax2 = ax1.twinx()
    ax2.set_ylabel('Accuracy (%)', fontsize=12, color='tab:green')
    ax2.plot(np.arange(1, NN), val_accuracy_time[:NN-1], color='tab:green', label='Validation Accuracy', linewidth=2)
    #ax2.plot(NN+1, test_accuracy, 's', color='tab:green', label='Test Accuracy', markersize=10)
    ax2.set_ylim(0, 100)
    ax2.tick_params(axis='y', labelcolor='tab:green')
    ax2.grid(False)

    # plot final accuracy
    ax2.plot(NN+1, test_acc, 'o', color='tab:green', label='Test Accuracy', markersize=10)

    # Title and legend
    plt.title(title, fontsize=14)
    fig.tight_layout()

    # Add legends for both axes
    ax1.legend(loc='upper left', fontsize=10)
    ax2.legend(loc='upper right', fontsize=10)

    # Save the figure with high resolution
    plt.savefig(f"./plot/{title}", dpi=300, bbox_inches='tight')

    plt.show()

# function to train model
def train_model(model, train_loader, val_loader,test_loader,  n_epochs=100,
                 learning_rate=0.001,criterion=nn.CrossEntropyLoss(),
                 augmentation=False,patience=10, model_path = './trained_models'):
    """
    Function to train and evaluate the defined model.

    Parameters:
        model (torch.nn.Module): The neural network model.
        train_loader (torch.utils.data.DataLoader): DataLoader for training data.
        val_loader (torch.utils.data.Dataset): Validation dataset.
        test_loader (torch.utils.data.DataLoader): DataLoader for validation data.
        optimizer (torch.optim.Optimizer): Optimizer for training the model.
        n_epochs (int): Number of training epochs.
        number_input (int): Number of points in the input data.
        num_channels (int): Number of channels in the input data.

    Returns:
        accuracy_list (list): List of accuracies computed from each epoch.
        train_loss_list (list): List of training losses from each epoch.
        val_loss_list (list): List of validation losses from each epoch.
        y_pred (list): List of predicted values.
        y_true (list): List of true values.
    """

    model_name = type(model).__name__
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # # Save loss and error for plotting
    loss_time = np.zeros(n_epochs)
    val_loss_time = np.zeros(n_epochs)
    val_accuracy_time = np.zeros(n_epochs)

    best_val_loss = float('inf')
    total = 0   # to store the total number of samples
    correct = 0 # to store the number of correct predictions
    

    # Create the directory for saving models
    directory = os.path.dirname(model_path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory)


    model_training_time = 0
    for epoch in tqdm(range(n_epochs)):
        running_loss = 0
        
        
        # putting the model in training mode
        model.train()
        
        initial_time = time.time()
        for data in train_loader:
            inputs, labels = data[0].to(device), data[1].to(device)
            inputs = inputs.float()

    
            # Data augmentation.
            if augmentation:
                # Find indices of noise labels in the entire data
                inoise = torch.where(labels.argmax(1) == 2)[0]

                # Determine the number of batches
                num_batches = inputs.shape[0]

                # Generate random numbers for augmentation decision and noise scaling
                random_decisions = torch.rand(num_batches, device=device) > 0.5
                noise_scales = torch.rand(num_batches, device=device) / 2

                # Generate a list of unique indices for noise samples
                unique_indices = torch.randperm(len(inoise), device=device)

                # Prepare noise for augmentation
                noises = torch.empty(num_batches, *inputs.shape[1:], device=device)
                for i, idx in enumerate(unique_indices):
                    noise = shuffle_phase_tensor(inputs[inoise[idx], :, :]).to(device)
                    noises[i % num_batches] = noise

                # Apply noise augmentation
                mask = random_decisions.unsqueeze(1).unsqueeze(2)  # Shape: (num_batches, 1, 1)
                scaled_noises = mask * noise_scales.unsqueeze(1).unsqueeze(2) * noises
                inputs += scaled_noises



            # Set the parameter gradients to zero
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            # computing the gradients
            loss.backward()

            # updating the parameters
            optimizer.step()

            running_loss += loss.item()
        final_time = time.time() - initial_time
        model_training_time += final_time
        

        # updating the training loss list
        loss_time[epoch] = running_loss/len(train_loader)

        # putting the model in evaluation mode. when you switch to model.eval(),
        #you’re preparing the model to test its knowledge without any further learning or adjustments.
        model.eval()
        
        with torch.no_grad(): # Context-manager that disabled gradient calculation.
            # Loop on samples in test set
            total = 0
            correct = 0
            running_test_loss = 0
            for data in val_loader:
                inputs, labels = data[0].to(device), data[1].to(device)
                inputs = inputs.float()
                labels = labels.float()

                outputs = model(inputs)
                running_test_loss += criterion(outputs, labels).item()

                correct += (outputs.argmax(1) == labels.argmax(1)).sum().item()
                total += labels.size(0)

    # Check for improvement
            if running_test_loss/len(val_loader) < best_val_loss:
                best_val_loss = running_test_loss/len(val_loader)
                epochs_no_improve = 0
                # Save the model if you want to keep the best one
                torch.save(model.state_dict(), model_path+model_name+'.pth')
            else:
                epochs_no_improve += 1
                # print(f'No improvement in validation loss for {epochs_no_improve} epochs.')

            if epochs_no_improve == patience:
                # print('Early stopping triggered.')
                
                break
        
        val_loss_time[epoch] = running_test_loss/len(val_loader)

        val_accuracy_time[epoch]=100 * correct / total
        # Print intermediate results on screen
        if (epoch+1) % 10 == 0:
            if val_loader is not None:
                print('[Epoch %d] loss: %.3f - accuracy: %.3f' %
                (epoch + 1, running_loss/len(train_loader), 100 * correct / total))
            else:
                print('[Epoch %d] loss: %.3f' %
                (epoch + 1, running_loss/len(train_loader)))


    model.load_state_dict(torch.load(model_path+model_name+'.pth'))
    # we now calculate the test accuracy
    # putting the model in evaluation mode. when you switch to model.eval(),
    #you’re preparing the model to test its knowledge without any further learning or adjustments.
    model.eval()
    with torch.no_grad(): # Context-manager that disabled gradient calculation.
        # Loop on samples in test set
        total = 0
        correct = 0
        running_test_loss = 0
        for data in test_loader:
            inputs, labels = data[0].to(device), data[1].to(device)
            inputs = inputs.float()
            labels = labels.float()

            outputs = model(inputs)
            running_test_loss += criterion(outputs, labels).item()

            correct += (outputs.argmax(1) == labels.argmax(1)).sum().item()
            total += labels.size(0)
        test_loss = running_test_loss/len(test_loader)
        test_accuracy = 100 * correct / total
        print('test loss: %.3f and accuracy: %.3f' % ( test_loss,test_accuracy))
    # Save the model if you want to keep the best one
   
    return loss_time, val_loss_time, val_accuracy_time, model_training_time,test_accuracy, test_loss





# Data Class
class PNWDataSet(Dataset): # create custom dataset
    def __init__(self, data,labels,num_classes): # initialize
        self.data = data 
        self.labels = labels
        self.num_classes = num_classes

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        sample_data = self.data[index]
        sample_labels = self.labels[index]
        
        # Convert labels to one-hot encoded vectors
        sample_labels = torch.nn.functional.one_hot(torch.tensor(sample_labels), num_classes=self.num_classes)
        
        return torch.Tensor(sample_data), sample_labels.float()  # return data as a tensor
    
    
    
# for additional data augmentation to generate noise with realistic PSD but random phase
def shuffle_phase_tensor(time_series):
    # Compute the Fourier transform
    new_time_series = torch.zeros(time_series.shape)
    for ichan in range(time_series.shape[0]):
        fourier_tensor = torch.fft.fft(torch.tensor(time_series[ichan,:]).float())
        # Get amplitude and phase
        amp_tensor = torch.abs(fourier_tensor)
        phase_tensor = torch.angle(fourier_tensor)
        
        # Shuffle the phase
        indices = torch.randperm(phase_tensor.size(-1)) 
        # in torch
        phase_tensor[1:len(phase_tensor)//2] = phase_tensor[indices[1:len(phase_tensor)//2]]
        phase_tensor[len(phase_tensor)//2+1:] = -torch.flip(phase_tensor[len(phase_tensor)//2+1:],dims=[0])  # Ensure conjugate symmetry
        
        # Reconstruct the Fourier transform with original amplitude and shuffled phase
        shuffled_fourier_tensor = amp_tensor * torch.exp(1j * phase_tensor)
        
        # Perform the inverse Fourier transform
        new_time_series[ichan,:] = torch.fft.ifft(shuffled_fourier_tensor).real


        window_length = new_time_series[ichan,:].size(-1)  # Taper along the last dimension
        hann_window = torch.hann_window(window_length).to(new_time_series.device)  # Ensure window is on the same device as tensor
        new_time_series[ichan,:] *= hann_window
    
    return new_time_series  # Return the real part

def shuffle_phase(time_series):
    # Compute the Fourier transform
    fourier_transform = np.fft.fft(time_series)
    
    # Get amplitude and phase
    amplitude = np.abs(fourier_transform)
    phase = np.angle(fourier_transform)
  
    # in numpy
    np.random.shuffle(phase[1:len(phase)//2])
    phase[len(phase)//2+1:] = -phase[len(phase)//2-1:0:-1]  # Ensure conjugate symmetry
    # in torch
    # Reconstruct the Fourier transform with original amplitude and shuffled phase
    shuffled_fourier = amplitude * np.exp(1j * phase)
    
    # Perform the inverse Fourier transform
    new_time_series = np.fft.ifft(shuffled_fourier)
    
    return new_time_series.real  # Return the real part
