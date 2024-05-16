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
		
		columns = ["lepton_eta_0", "lepton_phi_0", "lepton_pt_0",
		           "lepton_eta_1", "lepton_phi_1", "lepton_pt_1",
		           "jet_eta_0", "jet_phi_0", "jet_pt_0",
		           "jet_eta_1", "jet_phi_1", "jet_pt_1",
		           "jet_eta_2", "jet_phi_2", "jet_pt_2",
		           "jet_eta_3", "jet_phi_3", "jet_pt_3",
		           "jet_eta_4", "jet_phi_4", "jet_pt_4",
		           "jet_eta_5", "jet_phi_5", "jet_pt_5", ]
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
		label = torch.tensor([self.labels[idx]], dtype=torch.float32)
		feature = np.float32(self.features[idx].to_numpy())
		feature = torch.tensor(feature)
		
		return feature, label
