class ACDC(Dataset):

    def __init__(self, img_list, target_list, img_num, crop_size=256, padding_size=100):
        self.img_list = img_list
        self.target_list = target_list
        self.img_num = img_num
        self.crop_size = crop_size
        self.padding_size = padding_size

    def __len__(self):
        return sum(self.img_num)

    def __getitem__(self, idx):
        img = torch.tensor(self.img_list[idx], dtype=torch.float32)
        target = torch.tensor(self.target_list[idx], dtype=torch.float32)

        img, target = self.center_crop(img, target, self.crop_size, self.padding_size)

        return img, target

    def center_crop(self, img, target, crop_size, padding_size):
        img = F.pad(img, pad=(self.padding_size, self.padding_size, self.padding_size, self.padding_size),
                    mode='constant', value=0)
        target = F.pad(target, pad=(self.padding_size, self.padding_size, self.padding_size, self.padding_size),
                       mode='constant', value=0)

        img = ff.center_crop(img, crop_size)
        target = ff.center_crop(target, crop_size)

        return img, target