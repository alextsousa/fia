from keras.models import load_model
import numpy as np
from keras.utils.np_utils import to_categorical

def predicao():

    fid = open('./dataset/test.csv', 'r')
    lines = fid.readlines()
    fid.close()

    dataset = []

    for line in lines:
        new_line = line.rstrip('\n')
        new_line = new_line.replace(',','.')
        new_line = new_line.split(';')

        #!!# Converte o dataset para float
        # dataset.append(list( new_line ))
        dataset.append(list(map(float, new_line )))
        #!!#
    dataset = np.array(dataset)

    n_test_patterns = 1000

    # Divide o dataset em entradas (X) e saídas (Y)
    X = dataset[0:n_test_patterns, 0:18]
    Y = dataset[0:n_test_patterns, 18]

    # !!# Normaliza o dataset
    X = X / np.amax(X, axis=0)
    # !!#

    # !!# Categoriza as saídas
    Y = to_categorical(Y, 2)
    # !!#

    model = load_model('./modelo/model.h5')

    pred = model.predict(x=X, batch_size=1, verbose=0)

    n_correct = 0
    n_wrong = 0

    for i in range(len(pred)):
        y_pred = int(round(pred[i][0]))

        if y_pred == int(Y[i][0]):
            n_correct += 1
        else:
            n_wrong +=1

    acc = float(n_correct) / (n_correct + n_wrong) * 100

    print("Acc: " + str(acc)  + "%")

if __name__ == "__main__":
    predicao()