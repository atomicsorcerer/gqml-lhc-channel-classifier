import numpy as np
import torch
from torch.utils.data import DataLoader, random_split

from event_dataset import EventDataset
from pfn_model import ParticleFlowNetwork
from utils import train, test

data = EventDataset("Data/Processed Data/event_dataset.csv")

test_percent = 0.20
training_data, test_data = random_split(data, [1 - test_percent, test_percent])

batch_size = 32

train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=True)

model = ParticleFlowNetwork(latent_space_dim=8)

test_X = torch.Tensor([[2.2286e+00, -1.4301e+00, 3.4673e+04, 1, 2.0143e+00, -4.0211e-01,
                        2.6559e+04, 1, np.nan, np.nan, np.nan, np.nan,
                        np.nan, np.nan, np.nan, np.nan, np.nan,
                        np.nan, np.nan, np.nan, np.nan, np.nan,
                        np.nan, np.nan, np.nan, np.nan]])
print(next(model.stack.modules())[0](test_X))
# print(model.forward(test_X))

loss_function = torch.nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

epochs = 1
for t in range(epochs):
	print(f"Epoch {t + 1}\n-------------------------------")
	train(train_dataloader, model, loss_function, optimizer)
	test(test_dataloader, model, loss_function)

print("Finished Training!\n")
print(next(model.stack.modules())[0](test_X))
# print(model.forward(test_X))

do_save = input("Would you like to save the model? y/N -> ")
if do_save == "y":
	torch.save(model, "model.pth")
	print("Model Saved")
