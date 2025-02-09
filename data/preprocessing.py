from typing import List, Tuple
import pandas as pd

import torch
from torch.utils.data import DataLoader, TensorDataset

import statsmodels.api as sm
import statsmodels.formula.api as smf

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split

class Data():
    """
    Data class for handling and preprocessing credit transaction data
    
    This class loads credit transaction data from a CSV file, performs an optional train-test
    split, converts the data to PyTorch tensors, and creates DataLoaders for model training
    and evaluation. Additionally, it provides methods for data overview and exploratory data
    analysis (e.g., plotting histograms and fraud distribution)
    
    Attributes:
        batch_size (int): Batch size used for creating DataLoaders (default is 1000)
        credit_original_data (pd.DataFrame): The original dataset loaded from the CSV file
        features (np.ndarray): Array of feature values extracted from the dataset
        labels (np.ndarray): Array of label values extracted from the dataset
        X_train (np.ndarray): Training set features when data is split (available if split=True)
        X_test (np.ndarray): Testing set features when data is split (available if split=True)
        y_train (np.ndarray): Training set labels when data is split (available if split=True)
        y_test (np.ndarray): Testing set labels when data is split (available if split=True)
        train_loader (DataLoader): DataLoader for the training dataset (available if split=True)
        test_loader (DataLoader): DataLoader for the testing dataset (available if split=True)
        data_loader (DataLoader): DataLoader for the entire dataset when no split is performed (split=False)
    """
    batch_size: int = 1000

    def __init__(self, path: str):
        """
        Initialize the Data class by loading the CSV file and setting up train/test datasets

        Parameters:
            path: Path to the CSV file containing the credit data
        """
        # Load the original credit data from the CSV file
        self.credit_original_data = pd.read_csv(path)
    
    def _init_data(self, X, y) -> DataLoader:
        """
        Prepare a DataLoader from the given features (X) and labels (y)
        """
        m = X.shape[0]
        m = (m // self.batch_size) * self.batch_size

        X = X[:m, :]
        y = y[:m]

        X_tensor = torch.tensor(X, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.long)

        dataset = TensorDataset(X_tensor, y_tensor)
        data_loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        return data_loader

    def _init_train_test(self, split: bool = False):
        """
        Initialize data for training and testing from the original dataset
        Parameters:
            split: Determines if the data should be split into train and test sets. 
            True means the data is divided into both training and testing sets.
            False means the entire data is used as a single set.
        """
        # Extract features: select all columns except the first and last columns.
        # Extract labels: select the last column.
        self.features = self.credit_original_data.iloc[:, 1:-1].values
        self.labels = self.credit_original_data.iloc[:, -1].values

        if split:
            # If 'split' is True, divide the data into training (80%) and testing (20%) sets.
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                self.features,
                self.labels,
                test_size=0.2,
                random_state=42
            )

            self.train_loader = self._init_data(self.X_train, self.y_train)
            self.test_loader = self._init_data(self.X_test, self.y_test)
        else:
            # If 'split' is False, use the entire dataset as one unit
            self.data_loader = self._init_data(self.features, self.labels)

    def train_test_set(self, split: bool = False) -> Tuple[DataLoader, DataLoader] | DataLoader:
        """
        Retrieve the prepared DataLoader(s) based on whether the data was split

        Parameters:
            split: Indicates whether to split the data into training and testing sets.
            True to split; False to use the entire dataset.
        
        Returns:
            - Tuple(DataLoader, DataLoader) if split is True (training and testing loaders)
            - A single DataLoader if split is False
        """
        # Initialize data loading with the provided 'split' parameter
        self._init_train_test(split=split)
        if split:
            # If data is split, return both the training and testing DataLoaders
            return (self.train_loader, self.test_loader)
        else:
            # If data is not split, return the single DataLoader for the entire dataset
            return self.data_loader

    def overview(self):
        """
        Display an overview of the original credit data
        
        This method prints:
            - The first few rows of the dataset
            - Information about data types and non-null counts
            - Descriptive statistics of the dataset
        """
        print(self.credit_original_data.head())
        print(self.credit_original_data.info())
        print(self.credit_original_data.describe())

    def significant_var(self):
        """
        Fit a logistic regression model to identify significant variables in the dataset
        
        This method uses a generalized linear model (GLM) with a binomial family to fit a logistic regression,
        then prints the model summary showing the significance of each variable
        """
        # Define the logistic regression formula including variables V1 to V28 and Amount
        logit_equation = 'Class~V1+V2+V3+V4+V5+V6+V7+V8+V9+V10+V11+V12+V13+V14+V15+V16+V17+V18+V19+V20+V21+V22+V23+V24+V25+V26+V27+V28+Amount'
        # Fit the logistic regression model using statsmodels GLM with binomial family
        fit1 = smf.glm(formula=logit_equation,
                       data=self.credit_original_data,
                       family=sm.families.Binomial()).fit()
        # Print the summary of the fitted model
        print(fit1.summary())

    def histogram_distribution(self, columns_to_plot: List[str]):
        """
        Plot histograms with kernel density estimates (KDE) for specified columns
        
        Parameters:
            columns_to_plot: List of column names to plot histograms for
        """
        n_plots = len(columns_to_plot)
        # Calculate number of rows needed based on 3 plots per row
        fig, axes = plt.subplots(nrows=(n_plots + 2) // 3,
                                 ncols=3,
                                 figsize=(15, 10))
        fig.subplots_adjust(wspace=0.5, hspace=0.5)

        for i, col in enumerate(columns_to_plot):
            row_idx = i // 3  # Determine row index in subplot grid
            col_idx = i % 3   # Determine column index in subplot grid

            # Plot histogram with KDE for the given column using seaborn
            sns.histplot(data=self.credit_original_data,
                         x=col,
                         kde=True,
                         color='blue',
                         ax=axes[row_idx, col_idx])
            axes[row_idx, col_idx].set_title(f'Histogram of {col}')
            axes[row_idx, col_idx].set_xlabel(col)
            axes[row_idx, col_idx].set_ylabel('Frequency')

        # Remove any unused axes if the total number of columns is less than grid size
        for i in range(len(columns_to_plot), axes.size):
            fig.delaxes(axes.flatten()[i])

        # Display the figure with the subplots
        plt.show()

    def fraud_transactions_distribution(self):
        """
        Plot and display the distribution of fraudulent vs non-fraudulent transactions
        """
        # Create a count plot for the 'Class' variable
        plt.figure(figsize=(6, 4))
        sns.countplot(x='Class', data=self.credit_original_data, legend=False)
        plt.title('Distribution of Transactions')
        plt.xlabel('Class')
        plt.ylabel('Count')
        plt.show()

        # Calculate counts for fraud (Class==1) and non-fraud (Class==0) transactions
        fraud_count = (self.credit_original_data['Class'] == 1).sum()
        non_fraud_count = (self.credit_original_data['Class'] == 0).sum()
        print("Count of Fraudulent Transactions:", fraud_count)
        print("Count of Non-Fraudulent Transactions:", non_fraud_count)
        print("The positive class (frauds) account for {:>.5}% of all transactions".format(
            (100 * fraud_count) / (fraud_count + non_fraud_count)
        ))
