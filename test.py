from dataset import CelebADataset
celeba_dataset = CelebADataset('celeba_aligned.h5py')
print(len(celeba_dataset))
print(celeba_dataset[198].shape)