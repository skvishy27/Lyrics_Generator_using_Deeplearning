from tensorflow.keras.optimizers import Adam
import tensorflow as tf
import numpy as np
import model
import config
import data_preprocess
import joblib


def train_fn():

    x, y = data_preprocess.train_date()

    rnn_model = model.rnn1()
    # print(model.summary())

    rnn_model.compile(optimizer=Adam(lr=config.LEARNING_RATE),
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])

    history = rnn_model.fit(x, y,
                            epochs=config.NUM_EPOCHS,
                            verbose=2)

    rnn_model.save(f"{config.MODEL_PATH}my_model.h5")
    np.save(f'{config.MODEL_PATH}my_history.npy', history.history)


if __name__ == '__main__':
    train_fn()
