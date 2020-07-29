'''
created on 11/07/2020

@author: Ravindra Kumar
'''

from exp.Required_Modules_And_Packages import *

# Combined dataset class for both Image and Text modality
class ConcatDataset(Dataset):
    def __init__(self, image_modality, text_modality, label):
        self.x1 = image_modality
        self.x2 = text_modality
        self.y = label
        self.c = len(label.classes)
    def __len__(self):
        return len(self.y)
    def __getitem__(self, i):
        return (self.x1[i], self.x2[i]), self.y[i]

# Function for data loader to pick samples accordingly with appropriate padding
def my_collate(batch):
    x,y = zip(*batch)
    x1,x2 = zip(*x)
    x1,_ = zip(*x1)
    x1 = to_data(x1)
    x1 = torch.stack(x1)
    x2, y = pad_collate(list(zip(x2, y)), pad_idx=1, pad_first=True)
    return (x1,x2),y