import torch
import matplotlib.pyplot as plt
from network import Generator
from dataset import generate_random_seed

# load Generator
G = Generator()
PATH_G = "trained_G.pth"
G.load_state_dict(torch.load(PATH_G))

# plot several outputs from the trained generator
# plot a 3 column, 2 row array of generated images
f, axarr = plt.subplots(2,3, figsize=(16,8))
for i in range(2):
    for j in range(3):
        output = G.forward(generate_random_seed(100))
        img = output.detach().permute(0, 2, 3, 1).view(128, 128, 3).cpu().numpy()
        axarr[i, j].imshow(img, interpolation='none', cmap='Blues')
        pass
    pass