"""
Configuration parameters for audio processing, datasets, and model training.
"""

# Audio processing parameters
SR = 22050            # Sampling rate
DURATION = 4          # Duration of the audio in seconds
N_MELS = 128          # Number of Mel bands
N_FFT = 2048          # FFT window size
HOP_LENGTH = 512      # Hop length for STFT
N_MFCC = 13           # Number of MFCC coefficients

# Model parameters
NUM_CLASSES = 10      # Number of audio classes

# Training parameters
BATCH_SIZE = 128      # Batch size for training
LEARNING_RATE = 0.001 # Initial learning rate
EPOCHS = 20           # Maximum number of training epochs
# XXX TEST: 
# EPOCHS = 3           # Maximum number of training epochs
EARLY_STOPPING_PATIENCE = 3 # Early stopping patience
EPOCHS_FOR_PRETRAINED = 2  # Epochs for fine-tuning pretrained models
NUMBER_WORKERS = 4    # Number of workers for data loading
# Scheduler parameters
SCHEDULER_STEP_SIZE = 5  # Step size for learning rate scheduler
SCHEDULER_GAMMA = 0.5     # Gamma for learning rate scheduler

# Model architecture parameters
DROPOUT_RATE = 0.5    # Dropout rate for CNN
FC_SIZE = 256         # Size of fully connected layer

# Other settings
RANDOM_SEED = 42      # Random seed for reproducibility
MODEL_SAVE_PATH = "best_model.pth"  # Path to save the best model
