from utils import *
from options import options


def main():
    # see options.py
    
    args = options()
    expr_path = GenerateEnvironment(args)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(device)

    train_transform, test_transform = None, None
    if args.dataset != "MNIST":
        stats = ((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4, padding_mode='reflect'),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(*stats)
        ])

        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(*stats)
        ])

    print(args.dataset)
    print(args.model)

    train_dataset = ShuffledDataset(args.dataset, './data', args.top_k * args.label_shuffle,
                                    train=True, download=True, transform=train_transform)
    test_dataset = ShuffledDataset(args.dataset, './data', 0, train=False, transform=test_transform)

    test_loader = DataLoader(test_dataset, args.batch_size, shuffle=False)

    # Begin noise detection
    model = torch.hub.load('pytorch/vision:v0.10.0', args.model, pretrained=True).to(device)
    if args.dataset == "MNIST":
        model.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False).to(device)
    elif args.dataset == "CIFAR10":
        print(model.fc)
        model.fc = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(2048, 1000)
        ).to(device)
    loss_recording = train(model, args.epochs, True, train_dataset, test_loader, device, args)

    # Report the noise detection results
    training_size = len(train_dataset)
    pred_indices = [t[1] for t in sorted(zip(loss_recording, range(len(train_dataset))), reverse=True,
                                         key=lambda x: x[0])[:int(training_size * args.top_k)]]
    if args.label_shuffle:
        changed_indices = train_dataset.get_shuffle_mapping().keys()
        noise_detected = list(set(changed_indices) & set(pred_indices))
        print("The model detected {} shuffled labele training samples out of {} total samples".format(len(noise_detected), len(changed_indices)))
    
    saved_dest = DumpNoisesToFile(pred_indices, args.dataset, expr_path)
    print("Indices of detected noises are saved to " + saved_dest)

    # cleanse the dataset and retrain
    model = torch.hub.load('pytorch/vision:v0.10.0', args.model, pretrained=True).to(device)
    if args.dataset == "MNIST":
        model.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False).to(device)
    elif args.dataset == "CIFAR10":
        model.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(2048, 1000)
        ).to(device)
    train_dataset.cleanse(pred_indices)
    train(model, args.epochs, False, train_dataset, test_loader, device, args)


if __name__ == '__main__':
    main()
