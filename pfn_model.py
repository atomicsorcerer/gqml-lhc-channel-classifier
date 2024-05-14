import numpy as np
import torch
from torch import nn


class ParticleFlowNetwork(nn.Module):
	def __init__(self):
		super().__init__()
		self.stack = nn.Sequential(
			nn.Linear(33, 2048),
			nn.Sigmoid(),
			nn.Linear(2048, 2048),
			nn.Sigmoid(),
			nn.Linear(2048, 1024),
			nn.Sigmoid(),
			nn.Linear(1024, 512),
			nn.Sigmoid(),
			nn.Linear(512, 256),
			nn.Sigmoid(),
			nn.Linear(256, 256),
			nn.Sigmoid(),
			nn.Linear(256, 1),
			nn.Sigmoid()
		)
	
	def forward(self, x):
		logits = self.stack(x)
		return logits


class ParticleMapping(nn.Module):
	def __init__(self, input_size: int, output_dimension: int, hidden_layer_dimensions=None):
		"""
		Maps each set of observables of a particle to a specific dimensional output and sums them together.
		
		>>> ParticleMapping(4, 8, hidden_layer_dimensions=[100, 100])
		
		Args:
			input_size: The number of data points each particle has.
			output_dimension: The fixed number of output nodes.
			hidden_layer_dimensions: A list of numbers which set the sizes of hidden layers.
		"""
		super().__init__()
		
		self.input_size = input_size
		self.output_dimension = output_dimension
		
		if hidden_layer_dimensions is None:
			hidden_layer_dimensions = [100]
		
		stack = nn.Sequential(nn.Linear(input_size, hidden_layer_dimensions[0] or output_dimension), )
		
		for i in range(len(hidden_layer_dimensions)):
			stack.append(nn.Linear(hidden_layer_dimensions[i], hidden_layer_dimensions[i + 1] or output_dimension))
			stack.append(nn.ReLU())
		
		stack.append(nn.Linear(hidden_layer_dimensions[-1] or input_size, output_dimension))
		
		self.stack = stack
	
	def forward(self, x: torch.Tensor):
		"""
		Forward implementation for ParticleMapping.
		
		Args:
			x: Input tensor(s).
		
		Raises:
			ValueError: Input tensor must be able to evenly split for the given input size.
			
		Returns:
			torch.Tensor: Output tensor with predefined dimensions.
		"""
		if len(x) % self.input_size != 0:
			raise ValueError(
				"Each particle must have the same number of observables, which must be equal to the input size.")
		
		inputs = np.array_split(x.numpy(), self.input_size)
		outputs = self.stack(inputs)
		output = np.zeros(self.output_size)
		
		for mapping in outputs:
			for i in range(self.output_dimension):
				output[i] += mapping[i]
		
		return torch.Tensor(output)
