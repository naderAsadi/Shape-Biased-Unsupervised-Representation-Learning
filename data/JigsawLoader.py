import numpy as np
import torch
import torch.utils.data as data
import torchvision
import torchvision.transforms as transforms
from PIL import Image
from random import sample, random, randint

#from data.texture_transfer import TextureTransfer


def get_random_subset(names, labels, percent):
    """
    :param names: list of names
    :param labels:  list of labels
    :param percent: 0 < float < 1
    :return:
    """
    samples = len(names)
    amount = int(samples * percent)
    random_index = sample(range(samples), amount)
    name_val = [names[k] for k in random_index]
    name_train = [v for k, v in enumerate(names) if k not in random_index]
    labels_val = [labels[k] for k in random_index]
    labels_train = [v for k, v in enumerate(labels) if k not in random_index]
    return name_train, name_val, labels_train, labels_val


def _dataset_info(txt_labels):
    with open(txt_labels, 'r') as f:
        images_list = f.readlines()

    file_names = []
    labels = []
    for row in images_list:
        row = row.split(' ')
        file_names.append(row[0])
        labels.append(int(row[1]))

    return file_names, labels


def get_split_dataset_info(txt_list, val_percentage):
    names, labels = _dataset_info(txt_list)
    return get_random_subset(names, labels, val_percentage)


class JigsawDataset(data.Dataset):
    def __init__(self, names, labels, jig_classes=100, img_transformer=None, tile_transformer=None, patches=True, bias_whole_image=None):
        self.data_path = ""
        self.names = names
        self.labels = labels

        #self.texture_transfer = TextureTransfer(style_dir='./PACS-cats/art', img_size=75)

        self.N = len(self.names)
        self.permutations = self.__retrieve_permutations(jig_classes)
        self.grid_size = 3
        self.bias_whole_image = bias_whole_image
        if patches:
            self.patch_size = 64
        self._image_transformer = img_transformer
        self._augment_tile = tile_transformer
        if patches:
            self.returnFunc = lambda x: x
        else:
            def make_grid(x):
                return torchvision.utils.make_grid(x, self.grid_size, padding=0)
            self.returnFunc = make_grid

    def get_tile(self, img, n):
        w = float(img.size[0]) / self.grid_size
        y = int(n / self.grid_size)
        x = n % self.grid_size
        tile = img.crop([x * w, y * w, (x + 1) * w, (y + 1) * w])
        tile = self._augment_tile(tile)
        return tile
    
    def get_permuted_image(self, name):
        n_grids = self.grid_size ** 2
        tiles = [None] * n_grids
        for i in range(9):
            new_name = name[:-4] + '_' + str(i) + '_' + str(randint(0,2)) + name[-4:]
            img = Image.open(new_name.replace('//', '/')).convert('RGB')
            img = transforms.ToTensor()(img)
            tiles[i] = img
        tiles = torch.stack(tiles, 0)
        tiles = self.returnFunc(tiles)
        #print(tiles.shape)
        return transforms.ToPILImage()(tiles)
    
    def get_image(self, index, permuted=False, styled=False):
        framename = self.data_path + '/' + self.names[index]
        if permuted == True:
            if random() > 0.0: # each pile styled differently
                img = self.get_permuted_image(framename)
            else: # piles have similar style
                new_name = framename[:-4] + '_trans' + str(randint(0,4)) + framename[-4:]
                img = Image.open(new_name).convert('RGB')
        else:
            if styled == True:
                new_name = framename[:-4] + '_trans' + framename[-4:]
                img = Image.open(new_name).convert('RGB')
            else:
                img = Image.open(framename).convert('RGB')
        
        return self._image_transformer(img)
        
    def __getitem__(self, index):
        n_grids = self.grid_size ** 2
        tiles = [None] * n_grids
        order = np.random.randint(len(self.permutations) + 1)  # added 1 for class 0: unsorted
        if self.bias_whole_image:
            if self.bias_whole_image > random():
                order = 0
        if order == 0: #return whole image
            if random() > 1.0: # return styles
                img = self.get_image(index, permuted=False, styled=True)
            else: # return normal
                img = self.get_image(index, permuted=False, styled=False)
            for n in range(n_grids):
                tiles[n] = self.get_tile(img, n)
            data = tiles

        else: #return Jigsaw
            if random() > 0.5: # return styles piles
                img = self.get_image(index, permuted=True, styled=True)
            else: # return normal piles
                img = self.get_image(index, permuted=False, styled=False)
            for n in range(n_grids):
                tiles[n] = self.get_tile(img, n)
            data = [tiles[self.permutations[order - 1][t]] for t in range(n_grids)]

        data = torch.stack(data, 0)
        return self.returnFunc(data), int(order), int(self.labels[index])


    def __len__(self):
        return len(self.names)

    def __retrieve_permutations(self, classes):
        all_perm = np.load('permutations_%d.npy' % (classes))
        # from range [1,9] to [0,8]
        if all_perm.min() == 1:
            all_perm = all_perm - 1

        return all_perm

    # def __getitem__(self, index):
    #     img = self.get_image(index)
    #     n_grids = self.grid_size ** 2
    #     tiles = [None] * n_grids
    #     for n in range(n_grids):
    #         tiles[n] = self.get_tile(img, n)

    #     order = np.random.randint(len(self.permutations) + 1)  # added 1 for class 0: unsorted
    #     if self.bias_whole_image:
    #         if self.bias_whole_image > random():
    #             order = 0
    #     if order == 0:
    #         data = tiles
    #     else:
    #         data = [tiles[self.permutations[order - 1][t]] for t in range(n_grids)]
            
    #     data = torch.stack(data, 0)
    #     return self.returnFunc(data), int(order), int(self.labels[index])

    # def normalize(self, img):
    #     """
    #     This function normalizes the input image.

    #     Parameters:
    #         img (torch.tensor): The tensor containing image
    #     """
    #     #shape = img.shape
    #     #img = img.squeeze()
    #     img = transforms.Normalize([0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(img)
    #     return img#.view(shape)


class JigsawTestDataset(JigsawDataset):
    def __init__(self, *args, **xargs):
        super().__init__(*args, **xargs)

    def __getitem__(self, index):
        framename = self.data_path + '/' + self.names[index]
        img = Image.open(framename).convert('RGB')
        return self._image_transformer(img), 0, int(self.labels[index])


class JigsawTestDatasetMultiple(JigsawDataset):
    def __init__(self, *args, **xargs):
        super().__init__(*args, **xargs)
        self._image_transformer = transforms.Compose([
            transforms.Resize(255, Image.BILINEAR),
        ])
        self._image_transformer_full = transforms.Compose([
            transforms.Resize(225, Image.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        self._augment_tile = transforms.Compose([
            transforms.Resize((75, 75), Image.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def __getitem__(self, index):
        framename = self.data_path + '/' + self.names[index]
        _img = Image.open(framename).convert('RGB')
        img = self._image_transformer(_img)

        w = float(img.size[0]) / self.grid_size
        n_grids = self.grid_size ** 2
        images = []
        jig_labels = []
        tiles = [None] * n_grids
        for n in range(n_grids):
            y = int(n / self.grid_size)
            x = n % self.grid_size
            tile = img.crop([x * w, y * w, (x + 1) * w, (y + 1) * w])
            tile = self._augment_tile(tile)
            tiles[n] = tile
        for order in range(0, len(self.permutations)+1, 3):
            if order==0:
                data = tiles
            else:
                data = [tiles[self.permutations[order-1][t]] for t in range(n_grids)]
            data = self.returnFunc(torch.stack(data, 0))
            images.append(data)
            jig_labels.append(order)
        images = torch.stack(images, 0)
        jig_labels = torch.LongTensor(jig_labels)
        return images, jig_labels, int(self.labels[index])