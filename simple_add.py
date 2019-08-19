'''
realize two digits add two digits
'''
import numpy as np
import keras
from typing import Dict, Tuple

class Simple_ED:
    charset = '0123456789+ '
    characters = dict((i, c) for i, c in enumerate(charset))
    indexs :Dict[str, int] = dict((c, i) for i, c in enumerate(charset))
    classes = len(charset)

    def encode(self,x :str) -> np.ndarray:
        row = len(x)
        code = np.zeros(shape=(row,self.classes))
        for i, c in enumerate(x):
            code[i, self.indexs[c]] = 1

        return code

    def decode(self,x :np.ndarray) -> str:
        return ''.join(self.characters[i] for i in np.argmax(x, axis=-1))

def prepare_data(eder :Simple_ED,data_num = 50000, digits = 2,
                 vali_rate=0.1) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    exp_len = digits*2 + 1
    result_len = digits + 1

    data = []
    label = []
    for i in range(data_num):
        x = np.random.randint(1, 10**digits, size=(2,))
        y = np.sum(x)

        x_s = "{}+{}".format(x[0],x[1])
        x_s += " " * (exp_len - len(x_s))
        x_t = eder.encode(x_s)

        y_s = "{}".format(y)
        y_s += " " * (result_len - len(y_s))
        y_t = eder.encode(y_s)

        data.append(x_t)
        label.append(y_t)

    data = np.array(data)
    label = np.array(label)

    end = np.floor((1.-vali_rate) * data_num).astype('int')

    return data[:end], label[:end], data[end:], label[end:]

if __name__ == '__main__':
    digits = 2

    eder = Simple_ED()
    x_train, y_train, x_test, y_test = prepare_data(eder,digits = digits)

    import keras
    from keras.layers import Input, LSTM, RepeatVector, Dense, TimeDistributed
    from keras.callbacks import LambdaCallback

    inputs = keras.Input((digits*2+1,eder.classes),name='input')

    encode_lstm = LSTM(64,name='encode_lstm')(inputs)

    repeat = RepeatVector(digits+1)(encode_lstm)

    lstm2 = LSTM(64,return_sequences=True)(repeat)
    lstm3 = LSTM(128,return_sequences=True)(lstm2)

    prediction = TimeDistributed(
        Dense(eder.classes, activation='softmax', name='classifier')
    )(lstm3)

    model = keras.models.Model(inputs = inputs, outputs = prediction)
    model.summary()

    model.compile(optimizer='Adam', loss=keras.losses.categorical_crossentropy)

    model.fit(x_train, y_train, batch_size=32,epochs=10,verbose=2,
              callbacks=[keras.callbacks.ModelCheckpoint("simple.h5")],
              validation_data=(x_test,y_test),
              shuffle=True)

    print(model.evaluate(x_test,y_test))