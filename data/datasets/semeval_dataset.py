import os
import pandas as pd
from torch.utils.data import Dataset
from sklearn import preprocessing
import torch

class SemevalDataset(Dataset):
    def __init__(self, csv_file):
        """
        Initializes the SemevalDataset.

        Args:
            csv_file (str): Path to the CSV file containing the dataset.
        """
        # Load the pre-trained RoBERTa model from fairseq
        roberta = torch.hub.load('pytorch/fairseq', 'roberta.large')
        # Initialize the label encoder
        le = preprocessing.LabelEncoder()
        # Read the CSV file into a pandas DataFrame
        self.dataframe = pd.read_csv(csv_file)

        # Encode the 'explanation' column with integer labels
        self.dataframe['explanation'] = le.fit_transform(self.dataframe.explanation.values)
        # Encode the text pairs using RoBERTa
        self.tokens = [roberta.encode(x, y) for x, y in zip(self.dataframe['chunk1'], self.dataframe['chunk2'])]

    def __len__(self):
        """
        Returns the length of the dataset.

        Returns:
            int: Number of samples in the dataset.
        """
        return len(self.dataframe)

    def __getitem__(self, idx):
        """
        Retrieves a single sample from the dataset.

        Args:
            idx (int): Index of the sample to retrieve.

        Returns:
            tuple: Tuple containing the encoded text, value, and explanation tensors.
        """
        x_data = self.tokens  # Encoded text tokens
        value_data = torch.Tensor(self.dataframe.iloc[:, 2].values)  # Value column as tensor
        exp_data = torch.Tensor(self.dataframe.iloc[:, 3].values)  # Explanation column as tensor

        return x_data[idx], value_data[idx], exp_data[idx]
