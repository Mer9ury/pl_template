from torchvision import transforms

def get_transform(resize=True):

    data_transforms = {'KonIQ':{
        'train': transforms.Compose([
            # transforms.Resize((224,224)),
            # transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

        ]),
        'test': transforms.Compose([
    #         transforms.Resize(input_size),
    #         transforms.CenterCrop(input_size),
            # transforms.Resize((224,224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])}
    }

    return data_transforms