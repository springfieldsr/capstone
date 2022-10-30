from nlp_utils import *
from options import options


def main():
    args = options()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    df_train, df_val = create_dataframes(args.dataset)
    print(len(df_train), len(df_val))
    print(df_train.head())

    val_dataset = Dataset(df_val, 0, False)
    train_dataset = Dataset(df_train, args.top_k * args.label_shuffle, True)

    num_classes = 5 if args.dataset == 'bbc_text' else 20

    model = BertClassifier(num_classes=num_classes).to(device)
    loss_recording = train(model, args.epochs, True, train_dataset, val_dataset, device, args)

    training_size = len(train_dataset)
    pred_indices = [t[1] for t in
                    sorted(zip(loss_recording, range(len(train_dataset))), reverse=True, key=lambda x: x[0])[
                    :int(training_size * args.top_k)]]
    if args.label_shuffle:
        changed_indices = train_dataset.get_shuffle_mapping().keys()
        noise_detected = list(set(changed_indices) & set(pred_indices))
        print(
            "The model detected {} shuffled label training samples out of {} total samples".format(len(noise_detected),
                                                                                                   len(changed_indices)
                                                                                                   ))
    else:
        print("Found {} total noises, saving...".format(len(pred_indices)))
        np.save("results.npy", np.array(pred_indices))

    print("Removing noises and beginning training...")
    model = BertClassifier(num_classes=num_classes).to(device)
    train_dataset.cleanse(pred_indices)
    train(model, args.epochs, False, train_dataset, val_dataset, device, args)


if __name__ == '__main__':
    main()