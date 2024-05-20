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
		self.stack = nn.Sequential(
			ParticleMapping(4, latent_space_dim, [100, 100]),
			nn.Linear(latent_space_dim, 256),
			nn.ReLU(),
			nn.Linear(256, 256),
			nn.ReLU(),
			nn.Linear(256, 128),
			nn.ReLU(),
			nn.Linear(128, 128),
			nn.ReLU(),
			nn.Linear(128, 1),
			nn.Sigmoid()
		)
	
	def forward(self, x) -> torch.Tensor:
		"""
		Forward implementation for ParticleFlowNetwork.
		
		Args:
			x: Input tensor(s).

		Returns:
			torch.Tensor: Output tensor with two values each representing the probabilities of signal and background.
		"""
		logits = self.stack(x)
		
		return logits


class ParticleMapping(nn.Module):
	def __init__(self, input_size: int, output_dimension: int, hidden_layer_dimensions=None) -> None:
		"""
		Maps each set of observables of a particle to a specific dimensional output and sums them together.
		
		>>> particle_map = ParticleMapping(4, 8, hidden_layer_dimensions=[100, 50])
		>>> X = torch.Tensor([[1, 2, 3, 4]])
		>>> X2 = torch.Tensor([[5, 6, 7, 8]])
		>>> X3 = torch.Tensor([[1, 2, 3, 4, 5, 6, 7, 8]])
		>>> particle_map.forward(X) + particle_map.forward(X2) == particle_map.forward(X3)
		tensor([[True, True, True, True, True, True, True, True]])
		
		Args:
			input_size: The number of data points each particle has.
			output_dimension: The fixed number of output nodes.
			hidden_layer_dimensions: A list of numbers which set the sizes of hidden layers.
		
		Raises:
			TypeError: If hidden_layer_dimensions is not a list.
			ValueError: If hidden_layer_dimensions is an empty list.
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
		stack.append(nn.Sigmoid())
		
		self.stack = stack
	
	def forward(self, x: torch.Tensor) -> torch.Tensor:
		"""
		Forward implementation for ParticleMapping.
		
		Args:
			x: Input tensor(s).
		
		Returns:
			torch.Tensor: Output tensor with predefined dimensions.
		"""
		logits = []
		
		for tensor in x:
			tensor = self.individual_map(tensor)
			logits.append(tensor)
		
		logits = torch.stack(tuple(logits))
		print(logits.grad_fn)
		
		return logits
	
	def individual_map(self, x) -> torch.Tensor:
		"""
		Individual mapping for summand.
		
		>>> particle_map = ParticleMapping(4, 8, hidden_layer_dimensions=[100, 50])
		>>> X = torch.Tensor([1, 2, 3, 4, torch.nan, torch.nan, torch.nan, torch.nan])
		>>> len(particle_map.individual_map(X))
		8

		Args:
			x: Input tensor.

		Raises:
			ValueError: Input tensor must be able to evenly split for the given input size.

		Returns:
			torch.Tensor: Output tensor with predefined dimensions.
		"""
		x = x[~torch.isnan(x)]
		
		if len(x) % self.input_size != 0:
			raise ValueError(
				f"Each particle must have the same number of observables, which must be equal to the input size.")
		
		output = torch.zeros(self.output_dimension)
		
		for i in range(int(len(x) / self.input_size)):
			summand = self.stack(x[i * self.input_size:(i + 1) * self.input_size])
			
			for j, value in enumerate(summand):
				output[j] += value
		
		return torch.Tensor(output)
