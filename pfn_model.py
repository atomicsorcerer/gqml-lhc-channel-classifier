import torch
from torch import nn


class ParticleFlowNetwork(nn.Module):
	def __init__(self, latent_space_dim) -> None:
		"""
		PFN model. Takes per-particle information and attempts to classify the event as signal or background.
		
		Args:
			latent_space_dim: The size of the latent space vector.
		"""
		super().__init__()
		
		self.latent_space_dim = latent_space_dim
		
		self.p_map = ParticleMapping(4, 24, latent_space_dim, [100, 100])
		
		self.stack = nn.Sequential(
			nn.Linear(latent_space_dim, 100),
			nn.ReLU(),
			nn.Linear(100, 100),
			nn.ReLU(),
			nn.Linear(100, 100),
			nn.ReLU(),
			nn.Linear(100, 100),
			nn.ReLU(),
			nn.Linear(100, 1),
		)
	
	def forward(self, x) -> torch.Tensor:
		"""
		Forward implementation for ParticleFlowNetwork.
		
		Args:
			x: Input tensor(s).
		
		Returns:
			torch.Tensor: Output tensor with two values each representing the probabilities of signal and background.
		"""
		x = self.p_map(x)
		x = self.stack(x)
		
		return x


class ParticleMapping(nn.Module):
	def __init__(self, input_size: int, total_features: int, output_dimension: int,
	             hidden_layer_dimensions=None) -> None:
		"""
		Maps each set of observables of a particle to a specific dimensional output and sums them together.
		
		>>> particle_map = ParticleMapping(4, 8, 8, hidden_layer_dimensions=[100, 50])
		>>> X = torch.Tensor([[[1, 2, 3, 4], [float("nan"), float("nan"), float("nan"), float("nan")]]])
		>>> X2 = torch.Tensor([[[5, 6, 7, 8], [float("nan"), float("nan"), float("nan"), float("nan")]]])
		>>> X3 = torch.Tensor([[[1, 2, 3, 4], [5, 6, 7, 8]]])
		>>> particle_map.forward(X) + particle_map.forward(X2) == particle_map.forward(X3)
		tensor([[True, True, True, True, True, True, True, True]])
		
		Args:
			input_size: The number of data points each particle has.
			output_dimension: The fixed number of output nodes.
			hidden_layer_dimensions: A list of numbers which set the sizes of hidden layers.
		
		Raises:
			TypeError: If hidden_layer_dimensions is not a list.
			ValueError: If hidden_layer_dimensions is an empty list.
			ValueError: Input tensor must be able to evenly split for the given input size.
		"""
		super().__init__()
		
		self.input_size = input_size
		self.output_dimension = output_dimension
		
		if hidden_layer_dimensions is None:
			hidden_layer_dimensions = [100]
		elif not isinstance(hidden_layer_dimensions, list):
			raise TypeError(f"Hidden layer dimensions must be a valid list. {hidden_layer_dimensions} is not valid.")
		elif len(hidden_layer_dimensions) == 0:
			raise ValueError("Hidden layer dimensions cannot be empty.")
		
		stack = nn.Sequential(nn.Linear(input_size, hidden_layer_dimensions[0] or output_dimension), nn.ReLU())
		
		for i in range(len(hidden_layer_dimensions)):
			stack.append(
				nn.Linear(hidden_layer_dimensions[i],
				          hidden_layer_dimensions[i] if i == len(hidden_layer_dimensions) - 1 else
				          hidden_layer_dimensions[
					          i + 1]))
			stack.append(nn.ReLU())
		
		stack.append(nn.Linear(hidden_layer_dimensions[-1], output_dimension))
		
		self.stack = stack
		
		if total_features % input_size != 0:
			raise ValueError(
				f"Each particle must have the same number of observables, which must be equal to the input size. "
				f"Total_features % input_size must be zero.")
		
		self.avg_pool_2d = torch.nn.AvgPool2d((total_features // input_size, 1))
	
	def forward(self, x: torch.Tensor) -> torch.Tensor:
		"""
		Forward implementation for ParticleMapping.
		
		Args:
			x: Input tensor(s).
		
		Returns:
			torch.Tensor: Output tensor with predefined dimensions.
		"""
		x = self.stack(x)
		x = torch.nan_to_num(x)
		x = self.sum_pool_2d(x)
		x = torch.squeeze(x, 1)
		
		return x
	
	def sum_pool_2d(self, x):
		"""
		Performs sum pooling.
		
		Args:
			x: Input tensor(s).

		Returns:
			torch.Tensor: Output tensor with predefined output dimensions.
		"""
		x = self.avg_pool_2d(x)
		x = torch.mul(x, self.input_size)
		
		return x
