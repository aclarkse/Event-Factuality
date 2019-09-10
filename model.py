import tensorflow as tf
import matplotlib.pyplot as plt
from keras import backend as K
from keras.models import Sequential
from keras.layers import Embedding
from keras.layers import Input, Dropout, GlobalMaxPooling1D, Dense
from keras.layers import LSTM, Bidirectional
from keras.layers import BatchNormalization
from keras.optimizers import Adam
from keras.callbacks import Callback, EarlyStopping
from sklearn.metrics import confusion_matrix, classification_report

# check to see if GPU usage is enabled
if len(K.tensorflow_backend._get_available_gpus()) > 0:
    from keras.layers import CuDNNLSTM as LSTM
    from keras.layers import CuDNNGRU as GRU

# Initialize session
session = tf.Session()
K.set_session(session)

# load in pre-trained word embeddings

# to load glove embeddings, uncomment these lines
#pickle_in = open("glove_100.pickle", "rb")
#embedding_matrix = pickle.load(pickle_in)

# to load elmo embeddings, uncomment these lines
#pickle_in = open("elmo_3072.pickle", "rb")
#embedding_matrix = pickle.load(pickle_in)

# load bert embeddings
pickle_in = open("bert_3072.pickle", "rb")
embedding_matrix = pickle.load(pickle_in)

# some configuration
SEQ_LEN = 71
EMBEDDING_DIM = 3072
HIDDEN_UNITS = SEQ_LEN*2
OUTPUT_SIZE = 6
EPOCHS = 10
BATCH_SIZE = 25
LR = 1e-3

embedding_layer = Embedding(
    vocabulary_size + 1,
    EMBEDDING_DIM,
    weights = [embedding_matrix],
    trainable = False
)

def build_model():
    """ Instantiates a stacked, bidirectional LSTM model."""
    model = Sequential()
    model.add(embedding_layer)
    model.add(Bidirectional(LSTM(units=HIDDEN_UNITS, return_sequences=True,
                                input_shape= (SEQ_LEN, EMBEDDING_DIM))))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))
    model.add(Bidirectional(LSTM(units=HIDDEN_UNITS, return_sequences=True)))
    model.add(Dropout(0.4))
    model.add(BatchNormalization())
    model.add(GlobalMaxPooling1D())
    model.add(Dense(OUTPUT_SIZE, activation = 'softmax'))
    model.compile(
        loss = "sparse_categorical_crossentropy",
        metrics = ["accuracy"],
        optimizer = Adam(lr=LR))

    return model

model = build_model()

# train the model
res = model.fit(X_train,
                Y_train,
                epochs = EPOCHS,
                callbacks = [EarlyStopping(monitor='val_loss', patience=3)],
                verbose = 1,
                batch_size = BATCH_SIZE,
                validation_data = (X_val, Y_val))

# plot results
pd.DataFrame(res.history).plot(figsize=(8, 5))
plt.grid(True)
plt.gca().set_ylim(0, 1)
plt.show()