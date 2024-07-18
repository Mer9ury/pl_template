from torchvision import transforms

def get_transform(resize=True):

    data_transforms = {'MIT5K':{
        'train': transforms.Compose([
            transforms.RandomCrop(256),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ]),
        'test': transforms.Compose([
            transforms.RandomCrop(256),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])}
    }

    return data_transforms