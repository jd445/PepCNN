import gc

import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm


def datainput(filepath):
    # db: list form of a sequence
    # strdb: string form of a sequence
    # data_label: list of the label of the sequence
    # itemset: the itemset of the sequence
    # max_sequence_length: the maximum length of the sequence in a dataset.
    max_sequence_length = 0
    file = open(filepath)
    db = []
    data_label = []
    itemset = ''
    for i in file:
        temp = i.replace("\n", "").split("\t")
        seq_db = temp[1].split(" ")
        max_sequence_length = max(max_sequence_length, len(seq_db))
        db.append(seq_db)
        data_label.append(int(temp[0]))
    strdb = []
    for i in range(len(db)):
        str = ""
        strdb.append(str.join(db[i]))
        itemset = itemset + str.join(db[i])
    itemset = set(itemset)
    itemset = list(itemset)
    itemset.sort()
    print(itemset)

    return db, strdb, data_label, itemset, max_sequence_length


class SequenceDataset(Dataset):
    def seq_picture(db, itemset, max_sequence_length):
        # convert the sequence to a target data, which is sequence picture.
        seq_picture = []
        np.zeros([len(itemset), max_sequence_length])
        for i in tqdm(db):
            temp_seq_picture = np.zeros([1, len(itemset), max_sequence_length])
            for j in range(len(itemset)):
                for k in range(len(i)):
                    if i[k] == itemset[j]:
                        temp_seq_picture[0, j, k] = 1
            seq_picture.append(temp_seq_picture)
        return seq_picture

    def __init__(self, db, itemset, max_sequence_length, data_label):
        super(SequenceDataset).__init__()
        self.X = SequenceDataset.seq_picture(db, itemset, max_sequence_length)
        self.y = data_label

    def __getitem__(self, index):
        return self.X[index].astype(np.float32), self.y[index]

    def __len__(self):
        return len(self.X)





def dataload(batch_size = 1024*8,data_path = 'Homo_sapiens_data.txt'):

    db, strdb, data_label, itemset, max_sequence_length_homo = datainput(data_path)
    model_path = './model.ckpt'  # the path where the checkpoint will be saved
    x_train, x_test, y_train, y_test = train_test_split(db, data_label, test_size=0.1)

    train_set = SequenceDataset(x_train, itemset, max_sequence_length_homo, y_train)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
    test_set = SequenceDataset(x_test, itemset, max_sequence_length_homo, y_test)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
    del x_train, x_test, y_train, y_test
    gc.collect()

    return train_loader, test_loader, len(itemset)