import numpy as np
from tqdm import tqdm


def create_vocab(data_path, save_vocab_path):
    """
    根据分词数据集构建词典
    :param data_path: 数据集路径
    :param save_vocab_path:
    :return:
    """
    vocab_words = []
    with open(data_path, "r", encoding="utf8") as fo:
        for line in fo:
            sentence_words = line.strip().split("\t")[1].split()
            for word in sentence_words:
                if word not in vocab_words:
                    vocab_words.append(word)
    with open(save_vocab_path, "w", encoding="utf8") as fw:
        for word in vocab_words:
            fw.write(word+"\n")


def read_vocab(vocab_path, special_words):
    with open(vocab_path, "r", encoding="utf8") as fo:
        vocab_words = [word.strip() for word in fo]
    vocab_words = special_words + vocab_words
    idx2vocab = {idx: word for idx, word in enumerate(vocab_words)}
    vocab2idx = {word: idx for idx, word in enumerate(vocab_words)}
    return idx2vocab, vocab2idx


def read_dataset(data_path, vocab2idx):
    all_datas, all_labels = [], []
    with open(data_path, "r", encoding="utf8") as fo:
        lines = (line.strip() for line in fo)
        for line in tqdm(lines):
            label, sentence = line.split("\t")
            label = int(label)
            sentence = sentence.strip()
            sent2idx = [vocab2idx[word] if word in vocab2idx else vocab2idx['<UNK>'] for word in sentence]
            all_datas.append(sent2idx)
            all_labels.append(label)
    return all_datas, all_labels


def batch_generator(all_data , batch_size, shuffle=False):
    """
    :param all_data : all_data整个数据集
    :param batch_size: batch_size表示每个batch的大小
    :param shuffle: 每次是否打乱顺序
    :return:
    """
    all_data = [np.array(d) for d in all_data]
    data_size = all_data[0].shape[0]
    if shuffle:
        p = np.random.permutation(data_size)
        all_data = [d[p] for d in all_data]

    batch_count = 0
    while True:
        if batch_count * batch_size + batch_size > data_size:
            batch_count = 0
            if shuffle:
                p = np.random.permutation(data_size)
                all_data = [d[p] for d in all_data]
        start = batch_count * batch_size
        end = start + batch_size
        batch_count += 1
        yield [d[start: end] for d in all_data]
