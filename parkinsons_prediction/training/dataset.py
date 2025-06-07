import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl

class ParkinsonsDataset(Dataset):
    """Dataset class for PyTorch Lightning."""
    
    def __init__(self, df, features, target_column="target_norm"):
        self.df = df.copy()
        self.features = features
        self.target_column = target_column
        
        # Prepare features and targets
        self.X = torch.tensor(df[features].values, dtype=torch.float32)
        self.y = torch.tensor(df[target_column].values, dtype=torch.float32)
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        return {
            'features': self.X[idx],
            'target': self.y[idx]
        }
    

class ParkinsonsDataModule(pl.LightningDataModule):
    """PyTorch Lightning Data Module."""
    
    def __init__(self, train_df, val_df, test_df, features, 
                 batch_size=128, num_workers=0, target_column="target_norm"):
        super().__init__()
        self.train_df = train_df
        self.val_df = val_df
        self.test_df = test_df
        self.features = features
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.target_column = target_column
        
    def setup(self, stage=None):
        """Setup datasets for different stages."""
        if stage == "fit" or stage is None:
            self.train_dataset = ParkinsonsDataset(
                self.train_df, self.features, self.target_column
            )
            self.val_dataset = ParkinsonsDataset(
                self.val_df, self.features, self.target_column
            )
            
        if stage == "test" or stage is None:
            self.test_dataset = ParkinsonsDataset(
                self.test_df, self.features, self.target_column
            )
    
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset, 
            batch_size=self.batch_size, 
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True
        )
    
    def val_dataloader(self):
        return DataLoader(
            self.val_dataset, 
            batch_size=self.batch_size * 2, 
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True
        )
    
    def test_dataloader(self):
        return DataLoader(
            self.test_dataset, 
            batch_size=self.batch_size * 2, 
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True
        )