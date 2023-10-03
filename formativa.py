#IMPORTAÇÃO DAS BIBLIOTECAS
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, confusion_matrix, f1_score, recall_score
from sklearn import tree
import matplotlib.pylab as plt
import seaborn as sns
import pandas as pd
#Função para criar a árvore de decisão
def makeTree(x,y,t):
    x_treino,x_teste,y_treino,y_teste = train_test_split(np.array(x).reshape(-1,1),np.array(y).reshape(-1,1),test_size=0.2,random_state=42) #Separa as váriaveis de teste e treino
    model = DecisionTreeClassifier() #Cria a arvore
    model.fit(x_treino,y_treino) #Ajuda o modelo com os dados de treino
    prev = model.predict(x_teste) #Cria uma previsão
    prec = precision_score(y_teste,prev, average='macro') #Métrica de precisão dos dados
    acurracy = accuracy_score(y_teste,prev) #Métrica da acurácia dos dados
    f1 = f1_score(y_teste,prev) #Métrica de qualidade do modelo
    recall = recall_score(y_teste,prev) #Métrica para falsos negativos
    matrix = confusion_matrix(y_teste,prev) #Matriz de confusão
    print(f'{t}\nAcurácia: {acurracy}\nMatriz: {matrix}\nPrecisão: {prec}\nF1: {f1}\nRecall: {recall}\n-------------------------') #Printa as métricas
    fig = plt.figure(figsize=(10,8)) #Gera uma figura
    tree.plot_tree(model,feature_names=[str(x) for x in x],class_names=[str(y) for y in y],filled=True) #Cria o modelo em formato de figura
    plt.title(t) #Adiciona um título a figura
    plt.show() #Mostra a figura
    
dados = pd.read_csv('formativa\dados_produtos.csv') # Lê os dados do csv
print(dados.info()) #Printa a informação de cada coluna
print(dados.describe) #Descreve os dados

tv,cafeteira,videogame,ipod,notebook,celular = [],[],[],[],[],[] #Variaveis para receber a nota
tvP,cafeteiraP,videogameP,ipodP,notebookP,celularP = [],[],[],[],[],[] #Variaveis para receber se comprou ou não
for i in range(len(dados['purchased'])): #For em todos os dados do dataFrame
    if dados['product_name'][i] == 'Televisão': #Testa o nome do produto
        tv.append(dados['rating'][i]) #Adiciona o rating
        tvP.append(dados['purchased'][i]) #Adiciona se foi comprado ou não
    elif dados['product_name'][i] == 'Cafeteira':
        cafeteira.append(dados['rating'][i])
        cafeteiraP.append(dados['purchased'][i])
    elif dados['product_name'][i] == 'Videogame':
        videogame.append(dados['rating'][i])
        videogameP.append(dados['purchased'][i])
    elif dados['product_name'][i] == 'iPod':
        ipod.append(dados['rating'][i])
        ipodP.append(dados['purchased'][i])
    elif dados['product_name'][i] == 'Notebook':
        notebook.append(dados['rating'][i])
        notebookP.append(dados['purchased'][i])
    else:
        celular.append(dados['rating'][i])
        celularP.append(dados['purchased'][i])
''
boxdata = [tv,cafeteira,videogame,ipod,notebook,celular] #Adiciona todos os dados de rating em uma variaveis
bp = sns.boxplot(boxdata) #Cria um boxplot da variavel
plt.xticks([0,1,2,3,4,5],['tv','cafeteira','videogame','ipod','notebook','celular']) #Adiciona os titulos em baixo do respectivo boxplot
plt.show() #Mostra o boxplot
print(dados.isna().sum()) #Vê se existem valores não numéricos
print(dados.isnull().sum()) #Vê se existem valores nulos
makeTree(tv,tvP,'Tv') #Cria a arvore de decisão com base dos dados de televisão
makeTree(cafeteira,cafeteiraP,'Cafeteria') #Cria a arvore de decisão com base dos dados de Cafeteria
makeTree(videogame,videogameP,'Videogame') #Cria a arvore de decisão com base dos dados de Videogame
makeTree(ipod,ipodP,'iPod') #Cria a arvore de decisão com base dos dados de iPod
makeTree(notebook,notebookP,'Notebook') #Cria a arvore de decisão com base dos dados de Notebook
makeTree(celular,celularP,'Celular') #Cria a arvore de decisão com base dos dados de Celular