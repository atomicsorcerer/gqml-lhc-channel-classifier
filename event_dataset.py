import numpy as np
import polars as pl
import torch
from torch.utils.data import Dataset


class EventDataset(Dataset):
	def __init__(self, file_path) -> None:
		"""
		Initializes an EventDataset for a given CSV file.
		
		Args:
			file_path: Path to the CSV file to be read.
		"""
		self.labels = pl.read_csv(file_path).get_column("label").apply(
			lambda s: [1.0, 0.0] if s else [0.0, 1.0])  # First position is signal, second position is background.
		
		columns = ['lepton_type_0', 'lepton_type_1', 'lepton_charge_0', 'lepton_charge_1', 'lepton_theta_diff',
		           'lepton_phi_diff', 'lepton_angular_dist', 'lepton_inv_mass', 'sys_inv_mass', 'met_et', 'met_phi']
		features_tmp = pl.read_csv(file_path).select(columns)
		
		self.features = features_tmp
	
	def __len__(self) -> int:
		"""
		Calculates the number of events in the dataset.
		
		Returns:
			Number of events in the dataset
		"""
		return len(self.labels)
	
	def __getitem__(self, idx) -> tuple[torch.Tensor, torch.Tensor]:
		"""
		Gets the features and label for a given index in the dataset.
		
		Args:
			idx: Index of the feature to be returned.

		Returns:
			tuple: Feature and label at the index (feature, label)
		"""
		label = torch.tensor([self.labels[idx][0]], dtype=torch.float32)
		feature = torch.from_numpy(np.float32(self.features[idx].to_numpy()))
		
		return feature, label
