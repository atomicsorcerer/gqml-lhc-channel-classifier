import torch
from torch.utils.data import DataLoader, random_split

from event_dataset import EventDataset
from pfn_model import ParticleFlowNetwork
from utils import train, test

data = EventDataset("Data/Processed Data/event_dataset.csv", limit=100_000)

test_percent = 0.20
training_data, test_data = random_split(data, [1 - test_percent, test_percent])

batch_size = 64

train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=True)

model = ParticleFlowNetwork(latent_space_dim=8)

print(list(model.named_parameters())[0])

loss_function = torch.nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

epochs = 10
for t in range(epochs):
	print(f"Epoch {t + 1}\n-------------------------------")
	train(train_dataloader, model, loss_function, optimizer)
	test(test_dataloader, model, loss_function)

print("Finished Training!\n")

do_save = input("Would you like to save the model? y/N -> ")
if do_save == "y":
	torch.save(model, "model.pth")
	print("Model Saved")
