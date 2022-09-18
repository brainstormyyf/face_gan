import torch
from torch.utils.data import DataLoader
from dataset import generate_random_seed, CelebADataset
from network import Discriminator, Generator


# check if CUDA is available
# if yes, set default tensor type to cuda

if torch.cuda.is_available():
  torch.set_default_tensor_type(torch.cuda.FloatTensor)
  print("using cuda:", torch.cuda.get_device_name(0))
  pass

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
# create Discriminator and Generator

D = Discriminator()
G = Generator()
D.to(device)
G.to(device)

# create Dataset object
celeba_dataset = CelebADataset('celeba_aligned.h5py')

# create Dataloader
data_loader = DataLoader(
    dataset=celeba_dataset,
    batch_size=1,
    shuffle=True,
    generator=torch.Generator(device='cuda')
)

# train Discriminator and Generator
epochs = 2

for epoch in range(epochs):
    print("epoch = ", epoch + 1)

    # train Discriminator and Generator
    for step, image_data_tensor in enumerate(data_loader):
        image_data_tensor = image_data_tensor.to(device)
        # train discriminator on true
        D.train(image_data_tensor, torch.ones(1).to(device))

        # train discriminator on false
        # use detach() so gradients in G are not calculated
        D.train(G.forward(generate_random_seed(100).to(device)).detach(), torch.zeros(1).to(device))

        # train generator
        G.train(D, generate_random_seed(100).to(device), torch.ones(1).to(device))

        pass

    pass

# plot discriminator error
D.plot_progress()

# plot generator error
G.plot_progress()

PATH_D = "trained_D.pth"
torch.save(D.state_dict(), PATH_D)

PATH_G = "trained_G.pth"
torch.save(G.state_dict(), PATH_G)
