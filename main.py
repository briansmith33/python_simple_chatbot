from tensorflow.keras.layers import Input, Embedding, LSTM, Dense, Flatten
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Model
from playsound import playsound
from threading import Thread
import speech_recognition
from gtts import gTTS
import pandas as pd
import numpy as np
import string
import random
import json
import os

tokenizer = Tokenizer(num_words=2000)
le = LabelEncoder()


class Chat(Thread):
    def __init__(self, intents="intents.json", model_name="chat.h5"):
        super().__init__()
        with open(intents) as content:
            self.data = json.load(content)
        self.model_name = model_name
        self.tags = []
        self.inputs = []
        self.responses = {}
        self.x_train = None
        self.y_train = None
        self.vocabulary = None
        self.input_shape = None
        self.output_length = None
        self.init()

    def init(self):
        for intent in self.data['intents']:
            self.responses[intent['tag']] = intent['responses']
            for lines in intent['patterns']:
                self.inputs.append(lines)
                self.tags.append(intent['tag'])

        self.data = pd.DataFrame({"inputs": self.inputs, "tags": self.tags})

        self.data['inputs'] = self.data['inputs'].apply(lambda wrd: [ltrs.lower() for ltrs in wrd if ltrs not in string.punctuation])
        self.data['inputs'] = self.data['inputs'].apply(lambda wrd: ''.join(wrd))

        tokenizer.fit_on_texts(self.data['inputs'])
        train = tokenizer.texts_to_sequences(self.data['inputs'])

        self.x_train = pad_sequences(train)

        self.y_train = le.fit_transform(self.data['tags'])

        self.input_shape = self.x_train.shape[1]

        self.vocabulary = len(tokenizer.word_index)

        self.output_length = le.classes_.shape[0]

    def create_model(self):
        i = Input(shape=(self.input_shape,))
        x = Embedding(self.vocabulary + 1, 10)(i)
        x = LSTM(10, return_sequences=True)(x)
        x = Flatten()(x)
        x = Dense(self.output_length, activation="softmax")(x)
        model = Model(i, x)

        model.compile(loss="sparse_categorical_crossentropy", optimizer='adam', metrics=['accuracy'])

        model.fit(self.x_train, self.y_train, epochs=200)
        model.save(self.model_name)

    @staticmethod
    def listen():
        print("listening...")
        listener = speech_recognition.Recognizer()
        with speech_recognition.Microphone() as src:
            listener.adjust_for_ambient_noise(src)

            spoken = listener.listen(src)
            return listener.recognize_google(spoken).lower()

    def predict(self, model, text):
        texts_p = []
        prediction_input = [letters.lower() for letters in text if letters not in string.punctuation]
        prediction_input = ''.join(prediction_input)
        texts_p.append(prediction_input)

        prediction_input = tokenizer.texts_to_sequences(texts_p)
        prediction_input = np.array(prediction_input).reshape(-1)
        prediction_input = pad_sequences([prediction_input], self.input_shape)

        output = model.predict(prediction_input)
        return output.argmax()

    def respond(self, prediction):
        response_tag = le.inverse_transform([prediction])[0]
        response = random.choice(self.responses[response_tag])
        audio = gTTS(text=response, lang="en", slow=False)
        audio_path = os.path.join(os.environ['TEMP'], "response.mp3")
        audio.save(audio_path)
        playsound(audio_path)
        os.remove(audio_path)

        if response_tag == "goodbye":
            exit(0)

    def run(self):
        model = load_model(self.model_name)

        while True:
            try:
                self.respond(self.predict(model, self.listen()))

            except KeyboardInterrupt:
                exit(0)

            except speech_recognition.UnknownValueError:
                continue
