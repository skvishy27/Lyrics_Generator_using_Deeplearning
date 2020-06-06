import config
import data_preprocess
import joblib
import tensorflow as tf
import time
from tensorflow.keras.preprocessing.sequence import pad_sequences


def predict(load_model, tokenizer, word_index, max_len, text):

    for _ in range(config.GENERATE_WORDS):
        token_list = tokenizer.texts_to_sequences([text])[0]
        token_list = pad_sequences([token_list],
                                   maxlen=max_len-1,
                                   padding=config.PAD_TYPE)
        predicted = load_model.predict_classes(token_list, verbose=0)
        output_words = ''
        for words, index in word_index.items():
            if index == predicted:
                output_words = words
                break
        text += ' ' + output_words
    return text


if __name__ == '__main__':
    tokenizer = joblib.load(f'{config.MODEL_PATH}tokenizer.pkl')
    word_index = joblib.load(f"{config.MODEL_PATH}word_ind.pkl")
    max_len = joblib.load(f"{config.MODEL_PATH}max_len.pkl")
    load_model = tf.keras.models.load_model(f"{config.MODEL_PATH}my_model.h5")

    generate_text = "I've got a bad feeling about this"
    prediction = predict(load_model, tokenizer, word_index, max_len, generate_text)
    print(prediction)
