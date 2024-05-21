# Import necessary modules
from torch.utils import data
import torch
from torch.nn.utils.rnn import pad_sequence
from .datasets.semeval_dataset import SemevalDataset

# Function to build the dataset from a given CSV file
def build_dataset(csv):
    datasets = SemevalDataset(csv)  # Initialize the SemevalDataset with the CSV file
    return datasets  # Return the dataset object

# Function to create a DataLoader object for training or testing
def make_data_loader(cfg, csv, is_train=True):
    if is_train:
        batch_size = cfg.SOLVER.BATCH_SIZE  # Use training batch size from config
        shuffle = True  # Shuffle the dataset for training
    else:
        batch_size = cfg.TEST.BATCH_SIZE  # Use testing batch size from config
        shuffle = False  # Do not shuffle the dataset for testing

    datasets = build_dataset(csv)  # Build the dataset using the provided CSV

    num_workers = cfg.DATALOADER.NUM_WORKERS  # Number of worker threads for data loading

    # Create a DataLoader object
    data_loader = data.DataLoader(
        dataset=datasets,  # The dataset to load
        batch_size=batch_size,  # Number of samples per batch
        shuffle=shuffle,  # Whether to shuffle the data
        num_workers=num_workers,  # Number of worker threads
        pin_memory=False,  # Whether to use pinned memory
        collate_fn=my_collate_fn  # Custom collate function for batching
    )

    return data_loader  # Return the DataLoader object

# Custom collate function for padding sequences and creating batches
def my_collate_fn(batch):
    inputs, values, explanations = zip(*batch)  # Unzip the batch into inputs, values, and explanations
    inputs_padded = pad_sequence([i.clone().detach() for i in inputs], batch_first=True, padding_value=1)  # Pad the input sequences
    values = torch.tensor(values) if not isinstance(values[0], torch.Tensor) else torch.stack(values)  # Convert values to tensors
    explanations = torch.tensor(explanations) if not isinstance(explanations[0], torch.Tensor) else torch.stack(explanations)  # Convert explanations to tensors

    return inputs_padded, values, explanations  # Return the padded inputs, values, and explanations as a batch
