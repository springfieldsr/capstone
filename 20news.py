import re
import string
import pandas as pd
from sklearn.datasets import fetch_20newsgroups
from nlp_utils import *

from options import Options


def main():
    args = Options()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    df_train, df_val = create_dataframes(args.dataset)
    print(len(df_train), len(df_val))
    print(df_train.head())

    val_dataset = Dataset(df_val, 0, False)
    train_dataset = Dataset(df_train, args.top_k * args.label_shuffle, True)

    num_classes = 5 if args.dataset == 'bbc_text' else 20

    model = BertClassifier(num_classes=num_classes).to(device)
    final_epochs = train(model, args.epochs, True, train_dataset, val_dataset, device, args)

    model = BertClassifier(num_classes=num_classes).to(device)
    loss_recording = train(model, final_epochs, False, train_dataset, val_dataset, device, args)

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


if __name__ == '__main__':
    main()