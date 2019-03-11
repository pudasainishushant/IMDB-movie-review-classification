#Import necessary modules 
import re
import numpy as np
import pandas as pd
from keras.preprocessing.text import Tokenizer
from keras.models import Sequential
from keras.layers import Dense, LSTM
from keras.layers.embeddings import Embedding
from keras.preprocessing.sequence import pad_sequences

# Load input dataset
df = pd.read_csv('../data/raw/imdb_master.csv', encoding='cp1252')

# Remove all non alpha numeric characters from the review column of dataset
df['review'] = df['review'].apply((lambda x: re.sub('[^a-zA-z0-9\s]','',x)))

# Only keep entries with `label` available.
df = df[df.label != 'unsup']

# Map `label` to 1 and 0.
df['label']=df.label.map({'pos': 1, 'neg': 0})

# Tokenize the reviews in the dataset
max_features = 20000
max_len = 100
tokenizer = Tokenizer(num_words=max_features, lower=True, split=' ')
tokenizer.fit_on_texts(df['review'])

# Create training input X. Only use df where `type` is train.
X= tokenizer.texts_to_sequences(df[df['type'] == 'train']['review'])
# Also, pad the sequence to max_len
X = pad_sequences(X, maxlen=max_len)


### Training

# Create a model with Embedding -> LSTM -> Dense 
model = Sequential()

# Input / Embdedding
model.add(Embedding(max_features,100,mask_zero=True))
model.add(LSTM(64,dropout=0.4, recurrent_dropout=0.4,return_sequences=True))
model.add(LSTM(32,dropout=0.5, recurrent_dropout=0.5,return_sequences=False))

# Output layer
model.add(Dense(1, activation='sigmoid'))
model.summary()


# Train the model
epochs = 2
batch_size = 32
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X,Y, epochs=epochs, batch_size=batch_size, verbose=1)

# Testing the model using the test data
X_test= tokenizer.texts_to_sequences(df[df['type'] == 'test']['review'])
X_test = np.array(X_test)
X_test = pad_sequences(X_test, maxlen=max_len)
y_test = df[df['type']=='test']['label'].values
preds= model.predict(X_test)
preds_binary = (preds > 0.5).astype(int)
from sklearn.metrics import accuracy_score
accuracy_score(y_test,preds_binary)


### Sample prediction
text1 = "I disliked the movie. The acting was worse."
text= np.array([text1])
model.save('new_model.model')
import tensorflow as tf
model = tf.keras.models.load_model('new_model.model')
text= tokenizer.texts_to_sequences(text)
text = pad_sequences(text, maxlen=max_len)
prediction = model.predict(text)
print(prediction)