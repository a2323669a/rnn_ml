import io
import keras
import numpy as np
from typing import Tuple
import jieba

class Simple_ED:
    def __init__(self, path, encoding):
        with io.open(path, encoding=encoding) as f:
           self.text = f.read()
        self.charset = list(set(self.text))
        self.characters = dict((i, c) for i, c in enumerate(self.charset))
        self.indexs = dict((c, i) for i, c in enumerate(self.charset))
        self.classes = len(self.charset)
        self.text_len = len(self.text)

    def encode(self,x :str) -> np.ndarray:
        row = len(x)
        code = np.zeros(shape=(row,self.classes))
        for i, c in enumerate(x):
            code[i, self.indexs[c]] = 1

        return code

    def decode(self,x :np.ndarray) -> str:
        return ''.join(self.characters[i] for i in np.argmax(x, axis=-1))

class GSeq(keras.utils.Sequence):
    def __init__(self, eder: Simple_ED, maxlen, batch_size):
        self.eder = eder
        self.maxlen = maxlen
        self.data_num = eder.text_len - maxlen
        self.batch_size = batch_size

        print("all data:{}, batch_size:{}, count:{}".format(self.data_num, self.batch_size, self.__len__()))

    def __len__(self):
        return self.data_num // self.batch_size

    def __getitem__(self, idx):
        x = np.zeros((self.batch_size, self.maxlen, self.eder.classes), dtype=np.bool)
        y = np.zeros((self.batch_size, self.eder.classes), dtype=np.bool)

        start = idx * self.batch_size
        ques = eder.text[start:start + maxlen]
        for i in range(self.batch_size):
            c = eder.text[start+maxlen+i]
            x[i] = eder.encode(ques)
            y[i] = eder.encode(c)

            ques = ques[1:] + c

        return x, y

class PCall(keras.callbacks.Callback):
    def __init__(self, eder: Simple_ED, maxlen):
        super().__init__()
        self.eder = eder
        self.maxlen = maxlen

    def on_epoch_end(self, epoch, logs=None):
        predict_gene(self.model, self.eder, self.maxlen)

def predict_gene(model :keras.models.Model,eder :Simple_ED, maxlen,
                 pre_str=None ,str_len=None):
    if pre_str == None:
        seed = np.random.randint(0,eder.text_len-maxlen)
        pre_str = eder.text[seed: seed+maxlen]
    if str_len == None:
        str_len = maxlen * 10

    gene_str = []
    prefix = pre_str
    for i in range(str_len):
        code = eder.encode(prefix)
        pred = model.predict(code.reshape((1, ) + code.shape), verbose=0)
        ci = sample(pred[0])
        c = eder.characters[ci]
        gene_str.append(c)
        prefix = prefix[1:] + c

    print("pre_str:{}\n{}".format(pre_str, ''.join(gene_str)))


def sample(preds, temperature=1.0):
    # helper function to sample an index from a probability array
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)

if __name__ == '__main__':
    eder = Simple_ED('./data/hqg.txt', 'gbk')
    maxlen = 40

    from keras.layers import Input, Dense, LSTM, GRU, TimeDistributed, RepeatVector
    from keras.losses import categorical_crossentropy
    from keras.callbacks import ModelCheckpoint
    from keras.models import Model
    import calls

    input = Input(shape=(maxlen, eder.classes))

    lstm1 = LSTM(128, name='lstm_1', return_sequences=False, stateful=False)(input)

    prediction = Dense(eder.classes, activation='softmax', name='classifier')(lstm1)

    model = Model(inputs = input, outputs = prediction)
    model.summary()

    model.compile(optimizer=keras.optimizers.Adam(),
                  loss = categorical_crossentropy,
                  metrics=['acc']
                  )

    seq = GSeq(eder, maxlen, batch_size=32)
    pcall = PCall(eder, maxlen)
    save_call = calls.SaveCall(filepath='rnn_g.h5', period=300, mode='train_mode', max_one=False)
    iepoch = save_call.load(model)

    model.fit_generator(seq, epochs=1, verbose=1, callbacks=[pcall, save_call], shuffle=True, steps_per_epoch=10, initial_epoch=iepoch)