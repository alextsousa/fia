#! /usr/bin/python
# -*- coding:utf-8 -*-

# Importações
from keras.models import Sequential
from keras.models import load_model
from keras.layers import Dense
from keras.layers.core import Dropout
import numpy as np
import matplotlib.pyplot as plt
import re
from datetime import datetime
from keras.utils import plot_model

#!!# Categoriza as saídas
from keras.utils.np_utils import to_categorical
#!!#

# Just disables the warning, doesn't enable AVX/FMA
# import os
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# 
# Remove acentos de string 
#
def remover_acentos(txt):
    #return normalize('NFKD', txt).encode('ASCII', 'ignore').decode('ASCII')
    string_nova = re.sub('[^a-zA-Z0-9áéíóúÁÉÍÓÚâêîôÂÊÎÔãõÃÕçÇ: ]', '', txt)
    return string_nova

# 
# Retorna um string da hora atual com ano, mes, dia, hora, minuto e segundo ex.: 20181020123301
#
def getDateTime():
    now = datetime.now()
    strDateTime  = str(now.year)
    strDateTime += str(now.month)
    strDateTime += str(now.day)
    strDateTime += str(now.hour)
    strDateTime += str(now.minute)
    strDateTime += str(now.second)

    return strDateTime

#
# Processa o dataset (substitui campos 'string' por inteiros)
#
def process_dataset(strDtAtual):
    # Carrega o dataset original
    fid = open('./dataset/DadosAlunos.csv', 'r')
    

    print('Reading file...')
    lines = fid.readlines()
    fid.close()

    # Remove o cabeçalho
    lines = lines[1:]

    # Carrega dados do ibge para código dos municípior
    ibgeDataset = open('./dados/ibge.csv', 'r')
    ibgeLines = ibgeDataset.readlines()
    ibgeDataset.close()

    # Colunas Dataset Ibge
    # 01 - Sigla da U.F.
    # 02 - Código da U.F.
    # 03 - Código Município
    # 04 - Município
    ibge = []
    for line in ibgeLines:
        ibge.append(line.split(';'))

    # String com dados verdadeiros da saida

    strOut = ['ABANDONO', 'DESISTENTE', 'TRANCADO']

    new_lines = []

    print( 'Processing...')
    

    for line in lines:

        fields = line.split(';')

        # Ajusta os valor das colunas

        new_fields = []

        # Coluna 00 matricula
        # Nova coluna 0
        # new_fields.append(fields[0])
        
        # Coluna 01 adicionada ao final
        
        #Coluna 02 transforma data de nascimento em idade
        # Nova coluna 1
        now = datetime.now()
        new_fields.append( now.year - int(fields[2][6:]))
        
        #Coluna 03 cpf retirada

        #Coluna 04 v ou f transformado para 1 ou 0
        # Nova coluna 2
        if fields[4] == 'V':
            new_fields.append(1)
        else:
            new_fields.append(0)

        # Coluna 05 email passada para final
        
        # Coluna 06 transforma municipio em código do ibge
        # Nova coluna 3
        codIbge = 0
        for ibge_Line in ibge:
            cidade_pesquisa = remover_acentos(fields[6].upper())
            if cidade_pesquisa.find(remover_acentos(ibge_Line[3].upper())) != -1:
                codIbge = ibge_Line[2]
                break

            #if remover_acentos(ibge_Line[3].upper()) == remover_acentos(fields[6].upper()):
                
        new_fields.append(codIbge)

        # Coluna 07 retirada

        # Colunas de 08 a 21
        # Novas colunas 4 a 17
        for x in range(8,22):
            # Corrige dados vazios para 0
            new_fields.append(fields[x])        

        # Coluna 22 saida sera tratado depois

        # Coluna 23
        # Nova coluna 18
        new_fields.append(fields[23])

        # Coluna 24 v ou f transformado para 1 ou 0
        # # Nova coluna 19            
        # if fields[24].upper() == 'V':
            # new_fields.append(1)
        # else:
            # new_fields.append(0)

        # Coluna 25 retirada

        # Tratando a saida
        # Nova coluna 20
        found = 0

        for strOutItem in strOut:
            if strOutItem == fields[22]:
                found = 1
        new_fields.append(found)

        # Adiciona no final a coluna nome
        # Nova coluna 21
        #new_fields.append(fields[1])

        # Adiciona no final a coluna email
        # Nova coluna 22
        #new_fields.append(fields[5])

        # Adiciona no final a coluna município
        # Nova coluna 23
        #new_fields.append(fields[6])

        # Corrige informações vazias
        new_array = []
        for item in new_fields:
            if item == '':
                new_array.append(0)
            else:
                new_array.append(item)

        new_lines.append(';'.join(map(str, new_array)) + '\n')
    
    print('Writing output file...')

    fid = open('./dataset/output.csv', 'w')
    fid.writelines(new_lines)
    fid.close()

    print('Done.')


#
# Particiona o dataset em treinamento e teste
#
def split_dataset(strDtAtual):
    train_filename = "./dataset/train.csv"
    test_filename = "./dataset/test.csv"

    fid = open("./dataset/output.csv", "r")
    lines = fid.readlines()
    fid.close()

    train_lines = list()
    test_lines = list()

    for line in lines:
        if np.random.random() > 0.2:
            train_lines.append(line)
        else:
            test_lines.append(line)

    fid = open(train_filename, 'w')
    fid.writelines(train_lines)
    fid.close()

    fid = open(test_filename, 'w')
    fid.writelines(test_lines)
    fid.close()

    print('Done.')


#
# Train net
#  
def train_net(strDtAtual):
    # Fixa o gerador de números aleatórios
    seed = 7
    np.random.seed(seed)

    # Número de padrões usados para treinamento
    n_patterns = 1000

    fid = open('./dataset/train.csv', 'r')
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

    # Divide o dataset em entradas (X) e saídas (Y)
    X = dataset[0:n_patterns, 0:18]
    Y = dataset[0:n_patterns, 18]

    #!!# Normaliza o dataset
    X = X / np.amax(X, axis=0)
    #!!#

    #!!# Categoriza as saídas
    Y = to_categorical(Y, 2)
    #!!#

    # Cria o modelo
    model = Sequential()
    model.add(Dense(1000, input_dim=18, kernel_initializer="uniform", activation='sigmoid'))
    model.add(Dropout(0.2))
    model.add(Dense(1000, kernel_initializer="uniform", activation='sigmoid'))
    model.add(Dropout(0.2))

    #!!# Saída pode ter 2 valores diferentes 0 ou 1
    # model.add(Dense(1, kernel_initializer="uniform", activation='softmax'))
    model.add(Dense(2, kernel_initializer="uniform", activation='softmax'))
    #!!#

    # Compila o modelo
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    # Treina o modelo
    history = model.fit(X, Y, epochs=15, batch_size=10)

    # Exporta o modelo
    model.save('./resultados/model.h5')

    print(history.history.keys())
    print(history.history['acc'])

    plt.plot(history.history['acc'])
    plt.plot(history.history['loss'])
    plt.savefig("./resultados/" + strDtAtual + "treinamento.png")
    plt.show()

    # Avalia o modelo
    scores = model.evaluate(X, Y)
    print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))


#
# Teste da rede com padrões desconhecidos
#  
def test_net(strDtAtual):

    print("Testing...")
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

    model = load_model('./resultados/model.h5')

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

    strDtAtual = getDateTime()

    exit = False
    option = 0

    while not exit :
        option = 0
        print('1 - Processar Dataset')
        print('2 - Separa Dataset')
        print('3 - Treinar rede')
        print('4 - Testar rede')
        print('9 - Sair')
        option = input('Digite uma opção: ')

        if option == '1':
            process_dataset(strDtAtual)
        elif option == '2':
            split_dataset(strDtAtual)
        elif option == '3':
            train_net(strDtAtual)
        elif option == '4':
            test_net(strDtAtual)
        elif option == '9':
            exit = True
        else:
            print("opção inválida \n")