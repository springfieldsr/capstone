import torch
import random
import re
import string
import numpy as np
import pandas as pd
from transformers import BertTokenizer, BertModel, DistilBertModel, DistilBertTokenizer
from torch import nn

from torch.optim import Adam
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
from sklearn.datasets import fetch_20newsgroups

from const import PATIENCE_EPOCH

# tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-cased")


class Dataset(torch.utils.data.Dataset):

    def __init__(self, df, shuffle_percentage, train):

        self.labels = list(df['label'])
        self.label_category = list(set(self.labels))
        self.texts = [tokenizer(text,
                               padding='max_length', max_length = 512, truncation=True,
                                return_tensors="pt") for text in df['text']]
        percentage = train * shuffle_percentage
        dataset_len = len(self.labels)
        indices_to_shuffle = random.sample(range(dataset_len), int(percentage * dataset_len))
        self._create_shuffle_mapping(indices_to_shuffle)

    def _create_shuffle_mapping(self, indices):
        """
        Input:
        indices
            list of int to specify which samples to shuffle label
        """
        self.mapping = {}
        for index in indices:
            label = self.labels[index]
            new_label = label
            while new_label == label:
                new_label = random.choice(self.label_category)
            self.mapping[index] = new_label

    def get_shuffle_mapping(self):
        return self.mapping

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        sample, label = self.texts[idx], self.labels[idx]
        label = self.mapping.get(idx, label)
        return sample, label

    def cleanse(self, remove_list):
        remove_list = set(remove_list)
        self.texts = [text for idx, text in enumerate(self.texts) if idx not in remove_list]
        self.labels = [label for idx, label in enumerate(self.labels) if idx not in remove_list]
        self.mapping = {}


class BertClassifier(nn.Module):

    def __init__(self, num_classes, dropout=0.5):

        super(BertClassifier, self).__init__()

        # self.bert = BertModel.from_pretrained('bert-base-cased')
        self.bert = DistilBertModel.from_pretrained("distilbert-base-cased")
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(768, num_classes)
        self.relu = nn.ReLU()

    def forward(self, input_id, mask):

        # _, pooled_output = self.bert(input_ids= input_id, attention_mask=mask,return_dict=False)
        pooled_output = torch.mean(self.bert(input_ids= input_id, attention_mask=mask)['last_hidden_state'], dim=1)

        dropout_output = self.dropout(pooled_output)
        final_layer = self.linear(dropout_output)
        # final_layer = self.relu(linear_output)

        return final_layer


def train(model, epochs, pretrain, train_dataset, val_dataset, device, args):
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size)

    criterion = nn.CrossEntropyLoss(reduction='none')
    optimizer = Adam(model.parameters(), lr=args.lr)

    best_val_acc = 0
    patience = 0
    if not pretrain:
        loss_record = torch.zeros(len(train_dataset)).to(device)

    for epoch_num in range(epochs):
        model.train()

        # Manually shuffle the training dataset for loss recording later
        feed_indices = torch.randperm(len(train_dataset)).tolist()
        shuffled_dataset = Subset(train_dataset, feed_indices)
        train_dataloader = DataLoader(shuffled_dataset, args.batch_size, shuffle=False)

        total_acc_train = 0
        total_loss_train = 0
        sample_count = 0

        for train_input, train_label in tqdm(train_dataloader):
            train_label = train_label.to(device)
            mask = train_input['attention_mask'].to(device)
            input_id = train_input['input_ids'].squeeze(1).to(device)

            output = model(input_id, mask)

            batch_loss_list = criterion(output, train_label.long())

            # Begin loss recording at the assigned epoch
            if not pretrain and epoch_num >= epochs * args.recording_point:
                sample_indices = feed_indices[sample_count: sample_count + train_label.shape[0]]
                for idx, i in enumerate(sample_indices):
                    loss_record[i] += batch_loss_list[idx]

            batch_loss = torch.mean(batch_loss_list)
            total_loss_train += batch_loss.item()
            sample_count += train_label.shape[0]

            acc = (output.argmax(dim=1) == train_label).sum().item()
            total_acc_train += acc

            model.zero_grad()
            batch_loss.backward()
            optimizer.step()

        total_acc_val = 0
        total_loss_val = 0

        with torch.no_grad():
            model.eval()

            for val_input, val_label in val_dataloader:
                val_label = val_label.to(device)
                mask = val_input['attention_mask'].to(device)
                input_id = val_input['input_ids'].squeeze(1).to(device)

                output = model(input_id, mask)

                batch_loss_list = criterion(output, val_label.long())
                batch_loss = torch.mean(batch_loss_list)
                total_loss_val += batch_loss.item()

                acc = (output.argmax(dim=1) == val_label).sum().item()
                total_acc_val += acc

        val_acc = total_acc_val / len(val_dataset)
        print(
            f'Epochs: {epoch_num + 1} | Train Loss: {total_loss_train / len(train_dataset): .3f} \
                | Train Accuracy: {total_acc_train / len(train_dataset): .3f} \
                | Val Loss: {total_loss_val / len(val_dataset): .3f} \
                | Val Accuracy: {total_acc_val / len(val_dataset): .3f}')

        # Early stopping
        if pretrain:
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience = 0
            else:
                patience += 1
                if patience > PATIENCE_EPOCH:
                    print("Training early stops at epoch {}".format(epoch_num))
                    return epoch_num if pretrain else loss_record

    if pretrain:
        print("Training finished for total {} epochs".format(epoch_num))
    else:
        print("Noise detection training finished, returning loss recording...")
    return epochs if pretrain else loss_record


def create_dataframes(dataset_name='bbc_text'):
    if dataset_name == 'bbc_text':
        print('Dataset is bbc_text, processing...')
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

    elif dataset_name == '20news':
        print('Dataset is 20news group, processing...')
        dataset = fetch_20newsgroups(subset='all', remove=('headers', 'footers'))
        df = pd.DataFrame({"label": dataset.target, "text": dataset.data})

        alphanumeric = lambda x: re.sub(r"""\w*\d\w*""", '', x)
        punc_lower = lambda x: re.sub('[%s]' % re.escape(string.punctuation), '', x.lower())

        df['text'] = df['text'].map(alphanumeric).map(punc_lower)
    else:
        raise NotImplementedError

    df_train, df_val = np.split(df.sample(frac=1, random_state=42),
                                [int(.8 * len(df))])
    return df_train, df_val
