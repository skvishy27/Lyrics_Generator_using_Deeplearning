import config
import tensorflow as tf
import joblib
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Bidirectional, LSTM, Dense, Dropout


def rnn():

    vocab_size = joblib.load(f'{config.MODEL_PATH}vocab_size.pkl')
    max_length = joblib.load(f"{config.MODEL_PATH}max_len.pkl")

    model = Sequential()
    model.add(Embedding(vocab_size,
                        config.EMBEDDING_DIM,
                        input_length=max_length-1))
    model.add(Dropout(0.2))
    model.add(Bidirectional(LSTM(150, return_sequences=True)))
    model.add(Dropout(0.2))
    model.add(Bidirectional(LSTM(70)))
    model.add(Dropout(0.2))
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(16, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(vocab_size, activation='softmax'))

    return model


def rnn1():

    vocab_size = joblib.load(f'{config.MODEL_PATH}vocab_size.pkl')
    max_length = joblib.load(f"{config.MODEL_PATH}max_len.pkl")

    model = Sequential()
    model.add(Embedding(vocab_size,
                        config.EMBEDDING_DIM,
                        input_length=max_length-1))
    model.add(Bidirectional(LSTM(150)))
    model.add(Dense(vocab_size, activation='softmax'))

    return model
