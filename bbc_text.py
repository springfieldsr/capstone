import pandas as pd
from nlp_utils import *

from options import options


def main():
    args = options()
    print(args)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_path = './data/bbc-text.csv'
    df = pd.read_csv(data_path)
    labels = {'business': 0,
              'entertainment': 1,
              'sport': 2,
              'tech': 3,
              'politics': 4
              }
    label_mapping = lambda x: labels[x]
    df["label"] = df["label"].map(label_mapping)

    df_train, df_val = np.split(df.sample(frac=1, random_state=42),
                                         [int(.8*len(df))])
    print(df.head())
    print(len(df_train),len(df_val))

    val_dataset = Dataset(df_val, 0, False)
    train_dataset = Dataset(df_train, 0.15, True)

    # LR = 1e-6

    # model = BertClassifier().to(device)
    # finished_epochs = train(model, args.epochs, True, train_dataset, val_dataset, device, args)

    model = BertClassifier(5).to(device)
    loss_recording = train(model, 5, False, train_dataset, val_dataset, device, args)

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