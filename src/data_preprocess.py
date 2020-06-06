import config
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import joblib
import numpy as np


def read_text():
    data = open(config.DATA_PATH).read()
    corpus = data.lower().split('\n')
    return corpus


def tokenizer_sequences():
    corpus = read_text()

    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(corpus)
    word_index = tokenizer.word_index
    vocab_size = len(word_index) + 1

    joblib.dump(tokenizer, f'{config.MODEL_PATH}tokenizer.pkl')
    joblib.dump(word_index, f"{config.MODEL_PATH}word_ind.pkl")
    joblib.dump(vocab_size, f"{config.MODEL_PATH}vocab_size.pkl")

    return corpus, tokenizer


def create_n_grams():
    corpus, tokenizer = tokenizer_sequences()

    input_sequences = []
    for line in corpus:
        token_list = tokenizer.texts_to_sequences([line])[0]
        for i in range(1, len(token_list)):
            n_gram_sequences = token_list[:i+1]
            input_sequences.append(n_gram_sequences)

    max_sequences_len = max([len(x) for x in input_sequences])
    input_sequences = np.array(pad_sequences(input_sequences,
                                             maxlen=max_sequences_len,
                                             padding=config.PAD_TYPE))

    joblib.dump(max_sequences_len, f"{config.MODEL_PATH}max_len.pkl")
    return input_sequences


def train_date():
    input_sequences = create_n_grams()
    vocab_size = joblib.load(f"{config.MODEL_PATH}vocab_size.pkl")
    x, labels = input_sequences[:, :-1], input_sequences[:, -1]
    y = tf.keras.utils.to_categorical(labels, num_classes=vocab_size)
    return x, y


if __name__ == '__main__':
    read_text()
