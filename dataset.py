from os import listdir, sep
import torch
import random

from torch.utils.data import DataLoader
from torch.utils.data import Dataset

from skimage import io


def random_crop(data, crop_size):
    x = torch.randint(low=0, high=data.shape[0] - crop_size, size=(1,))
    y = torch.randint(low=0, high=data.shape[1] - crop_size, size=(1,))

    slice_x = slice(x, x + crop_size)
    slice_y = slice(y, y + crop_size)

    data = data[slice_x, slice_y, :]

    return data


def random_flip(x, flip_mode):
    f = []
    
    if 'h' in flip_mode and torch.randint(low=0, high=2, size=(1,)):
        f.append(1)

    if 'v' in flip_mode and torch.randint(low=0, high=2, size=(1,)):
        f.append(0)

    x = x.flip(dims=f)

    return x


class S7DatasetCN(Dataset):
    def __init__(self, directory, mode, factor,
                 crop_size, flip):
        self.directory = directory

        self.crop_size = crop_size
        self.flip = flip

        self.dirs = listdir(self.directory)
        random.seed(1234)
        random.shuffle(self.dirs)
        
        # :C i didnt make a mid pic for this
        try:
            for i in range(9):
                self.dirs.remove(f'20161109_224805_{i}')
        except ValueError:
            pass
        self.fname = 'medium_exposure'
        
        tmp_len = len(self.dirs)

        if mode == 'train':
            self.len = int(tmp_len * factor)
            self.dirs = self.dirs[:self.len]
        elif mode == 'test':
            self.len = tmp_len - int(tmp_len * factor)
            self.dirs = self.dirs[self.len:]

    def __len__(self):
        return self.len

    def __getitem__(self, idx):

        in_ = sep.join([self.directory, self.dirs[idx],
                        self.fname + '.bmp'])
        mid = sep.join([self.directory, self.dirs[idx],
                        self.fname + '.png'])
        out = sep.join([self.directory, self.dirs[idx],
                        self.fname + '.jpg'])

        in_ = torch.tensor(io.imread(in_).astype('float'))
        mid = torch.tensor(io.imread(mid).astype('float'))
        out = torch.tensor(io.imread(out).astype('float'))
#         print(in_.min(), in_.max())
#         print(mid.min(), mid.max())
#         print(out.min(), out.max())
        
#         norm = lambda x: (x / 128) - 1
        norm = lambda x: (x / 255)
        (in_, mid, out) = map(norm, (in_, mid, out))
#         print(in_.min(), in_.max())
#         print(mid.min(), mid.max())
#         print(out.min(), out.max())
#         print('='*40)

        if self.crop_size is not None:
            (in_, mid, out) = map(random_crop, (in_, mid, out), [self.crop_size]*3)

        if self.flip is not None:
            (in_, mid, out) = map(random_flip, (in_, mid, out), [self.flip]*3)

        (in_, mid, out) = map(lambda x: x.permute(2, 0, 1), (in_, mid, out))

        return (in_, mid, out)
    

def get_data(data_path, num_workers=0,
             batch_size=1, factor=0.9,
             crop_size=256, flip='hv'):

    # [TODO] get rid of those long args
    train_data = S7DatasetCN(
        directory=data_path,
        mode='train',
        factor=factor,
        crop_size=crop_size,
        flip=flip
    )

    test_data = S7DatasetCN(
        directory=data_path,
        mode='test',
        factor=factor,
        crop_size=crop_size,
        flip=flip
    )

    train_loader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )
    test_loader = DataLoader(
        test_data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )

    return train_loader, test_loader


if __name__ == '__main__':
    path = '/toleinik/data/S7-ISP-new'
    train, test = get_data(path)

    for i in train:
        print(i)
        break
