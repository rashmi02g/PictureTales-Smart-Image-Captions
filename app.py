from flask import Flask, render_template, request
import keras
import tensorflow
import numpy as np
import pickle
# from tqdm.notebook import tqdm
import json
import os
from tensorflow.keras.preprocessing.text import tokenizer_from_json

from keras.models import load_model
from tensorflow.keras.applications.vgg16 import VGG16,preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model
from tensorflow.keras.utils import to_categorical,plot_model
from tensorflow.keras.layers import Input,Dense,LSTM,Embedding,Dropout,add,Flatten

file_path = r'C:\Users\Rashmi Gahlot\Desktop\image_caption\tokenizer.json'
# with open(file_path, 'rb') as file:
#     tokenizer = pickle.load(file)

with open(file_path) as file:
    tokenizer_json = json.load(file)
    tokenizer = tokenizer_from_json(tokenizer_json)


# vgg_model=load_model('vgg_model.h5')
vgg_model = VGG16() 
vgg_model = Model(inputs=vgg_model.inputs,             
                  outputs=vgg_model.layers[-2].output)

max_length=35
vocab_size=8485

# model=load_model('best_model.h5')
# model.load_weights('model_weights.h5')
#encoder model
#image feature layers
inputs1= Input(shape=(4096,))
fe1 = Dropout(0.4)(inputs1)
fe2 = Dense(256 , activation= 'relu')(fe1)
#sequence features layers
inputs2=Input(shape=(max_length,))
se1 = Embedding(vocab_size, 256, mask_zero=True)(inputs2)
se2 = Dropout(0.4)(se1)
se3 = LSTM(256)(se2)

#decoder model
decoder1 = add([fe2,se3])
decoder2 = Dense(256, activation='relu')(decoder1)
outputs = Dense(vocab_size, activation='softmax')(decoder2)

model = Model(inputs=[inputs1,inputs2],outputs=outputs)
model.compile(loss='categorical_crossentropy', optimizer='adam')

model.load_weights('gen_capbest_model_weights.h5')

app = Flask(__name__)
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 1




@app.route('/')
def index():
    return render_template('index.html')

@app.route('/after', methods=['GET', 'POST'])
def after():
    file = request.files['file1']
    file.save('static/file.jpg')


    def idx_to_word(integer, tokennizer):
        for word, index in tokenizer.word_index.items():
            if index == integer:
                return word
        return None

    def predict_caption(model, image, tokenizer, max_length):
        in_text='startseq'
        for i in range(max_length):
            sequence = tokenizer.texts_to_sequences([in_text])[0]
            sequence = pad_sequences([sequence], maxlen=max_length)
            yhat = model.predict([image, sequence], verbose=0)
            yhat = np.argmax(yhat)
            word = idx_to_word(yhat, tokenizer)
            if word is None:
                break
            in_text += " " + word
            if word == 'endseq':
                break
        return in_text

    image_path = 'static/file.jpg'
    image = load_img(image_path, target_size=(224, 224))
    image = img_to_array(image)
    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
    image = preprocess_input(image)
    feature = vgg_model.predict(image, verbose=0)
    final=predict_caption(model, feature, tokenizer, max_length)

    return render_template('predict.html',final=final)


if __name__ == "__main__":
    app.run(debug=True)