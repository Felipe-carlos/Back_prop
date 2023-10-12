# Título:Neural Networks usando back propagation
# Autor: Felipe C. dos Santos
#

import random
import numpy as np
import matplotlib . pyplot as plt
import math
import csv
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.lines import Line2D

#--le os dados a serem lidos e os resultados
a = csv.reader(open('classification2.txt','r'),delimiter=',')
x1,x2,y =[],[],[]
for i in a:
    x1.append(float(i[0]))
    x2.append(float(i[1]))
    y.append(float(i[2]))

X=np.column_stack((x1, x2))
y=np.array(y).reshape(118,1)

# ---- Dividir os dados em treinamento e validação
np.random.seed(0)
ratio_de_dados = 0.4 #separa 40% dos dados para validação

shuffle_indicies = np.random.permutation(len(X))
X_train = X[shuffle_indicies[:math.ceil((1-ratio_de_dados)*len(X))]]
y_train = y[shuffle_indicies[:math.ceil((1-ratio_de_dados)*len(X))]]

X_validation = X[shuffle_indicies[math.ceil((1-ratio_de_dados)*len(X)):]]
y_validation = y[shuffle_indicies[math.ceil((1-ratio_de_dados)*len(X)):]]

x1_train = X_train[:, 0]
x2_train = X_train[:, 1]
x1_vali = X_validation[:, 0]
x2_vali = X_validation[:, 1]

#-----------arquitetura:
arquitetura = {
    'num_layers': 4,
    'n_L1': 2,
    'n_L2': 5,
    'n_L3': 5,
    'n_L4': 1,

}

def sigmoid(z):     #recebe um valor de z e retorna sigmoid aplicada a aquele ponto
    return 1/(1+math.exp(-z))
sigmoid_v =np.vectorize(sigmoid)

def tanh(z):         #recebe um valor de z e retorna tahn aplicada a aquele ponto
    return (math.exp(z)-math.exp(-z)) /(math.exp(z) + math.exp(-z))
tan_h_v = np.vectorize(tanh)

def deriv_sig(A):   #recebe um valor de z e retorna a derivada da sigmoid aplicada a aquele ponto
    return A*(1-A)
deriv_sig_v = np.vectorize(deriv_sig)

def deriv_tanh(z):  #recebe um valor de z e retorna a derivada da tanh aplicada a aquele ponto
    return 1 - ((math.exp(z)-math.exp(-z))**2 /(math.exp(z) + math.exp(-z))**2)
deriv_tanh_v = np.vectorize(deriv_tanh)



def init_weights(arquitetura):     #retorna um dicionario com os pesos inicializados - inicializaçãp xavier
    pesos = {}
    for l in range(1,arquitetura['num_layers']):

        pesos[f'w_{l}'] = np.random.rand(arquitetura[f'n_L{l+1}'],arquitetura[f'n_L{l}'])*2*(1/np.sqrt(arquitetura[f'n_L{l}']))-(1/np.sqrt(arquitetura[f'n_L{l}'])) #valores entre -1/sqrt(n) e 1/sqrt(n)
        pesos[f'b_{l}'] = np.random.rand(arquitetura[f'n_L{l + 1}'],1)*2*(1/np.sqrt(arquitetura[f'n_L{l}'])) -(1/np.sqrt(arquitetura[f'n_L{l}']))
    return pesos

def forward_prop(pesos,X,arquitetura    ): # entra com um dicionario com os pesos e X deve ser o par de valores x1 e x2
    fw_prop={}
    fw_prop[f'A_{1}'] = X.reshape(arquitetura['n_L1'],1)    #define A_1

    for l in range(2,arquitetura['num_layers']+1):
        fw_prop[f'Z_{l}'] = pesos[f'w_{l-1}'].dot(fw_prop[f'A_{l-1}']) + pesos[f'b_{l-1}']  #define zl como wl-1.al-1 + wb_l
        if l == arquitetura['num_layers']:                      #usa sigmoid apenas para ultima camada
             fw_prop[f'A_{l}'] = sigmoid_v(fw_prop[f'Z_{l}'])
        else:                                                   #usa tanh para as demais
            fw_prop[f'A_{l}'] = tan_h_v(fw_prop[f'Z_{l}'])

    return fw_prop                                              #Retorna dicionario com os valores de Z e A

def back_prop(pesos,fw_prop,y,arquitetura):     #entra com um dicionário com os pesos, resultados do forward e arquitetura
    bk_prop = {}
    l_max = arquitetura['num_layers']
    bk_prop[f'd_{l_max}'] = (fw_prop[f'A_{l_max}']-y) # gera o ultimo delta

    for l in range(arquitetura['num_layers']-1, 1,-1):
        bk_prop[f'd_{l}'] = pesos[f'w_{l}'].T.dot(bk_prop[f'd_{l+1}'])*deriv_tanh_v(fw_prop[f'Z_{l}']) #calcula os deltas
    for l in range(1,arquitetura['num_layers']):
        bk_prop[f'grad_{l}'] = bk_prop[f'd_{l+1}'].dot(fw_prop[f'A_{l}'].T)             #calcula o gradiente
        bk_prop[f'grad_b_{l}'] = bk_prop[f'd_{l + 1}']                                  #calcula o gradiente do bias
    return bk_prop         #retorna os deltas, e os gradientes para um ponto

def squeze(pesos,arquitetura): #recebe um dicionario com os pesos e espreme eles em um vetor (n,1)
    a=[]
    for l in range(1, arquitetura['num_layers']):
        a.append(pesos[f'b_{l}'].reshape(pesos[f'b_{l}'].size,1))
        a.append(pesos[f'w_{l}'].reshape(pesos[f'w_{l}'].size, 1))
    return np.vstack(a)

def resize_thetas(squezed_pesos,pesos):  #retorna um dicionaria com os pesos presentes no vetor squezed n,1
    pesos_redimensionados = {}
    off_set = 0
    for l in range(1, arquitetura['num_layers']):
        pesos_redimensionados[f'b_{l}']=squezed_pesos[off_set:off_set+pesos[f'b_{l}'].size].reshape(arquitetura[f'n_L{l + 1}'],1)
        off_set = off_set + pesos[f'b_{l}'].size
        pesos_redimensionados[f'w_{l}'] = squezed_pesos[off_set:off_set + pesos[f'w_{l}'].size].reshape(arquitetura[f'n_L{l+1}'],arquitetura[f'n_L{l}'])
        off_set = off_set + pesos[f'w_{l}'].size
    return pesos_redimensionados

def J(thetas,pesos,X,y,arquitetura):       #recebe os valores de theta já concatenados em um vetor (n,1)
    reshaped_thetas = resize_thetas(thetas,pesos)
    fw,_ ,_ = varer_dados(reshaped_thetas, X, y, arquitetura)
    output = []
    J_parcial = []
    lmax = arquitetura['num_layers']
    for i in range(len(y)):
        output.append(fw[i][f'A_{lmax}'])
        J = ((y[i].item()* math.log10(output[i].item())) + ((1-y[i].item())*math.log10(1-output[i].item())))
        J_parcial.append( J )
    return (-1/len(y))*sum(J_parcial)       #retorna o valor da função de custo - sem regularização

def prova_real(pesos,X,y,arquitetura):    #função para tirar a prova real do calculo do gradiente recebe os valores de theta já concatenados em um vetor (n,1)
    thetas = squeze(pesos,arquitetura)
    episilon=0.001
    grad_aprox=np.zeros((len(thetas),1))
    for i in range(len(thetas)):
        theta_plus= np.array(thetas)
        theta_minus = np.array(thetas)
        theta_plus[i] = theta_plus[i] + episilon
        theta_minus[i] = theta_minus[i] - episilon
        grad_aprox[i] = (J(theta_plus,pesos,X,y,arquitetura) - J(theta_minus,pesos,X,y,arquitetura))/(2*episilon)
    return grad_aprox           #retorna um vetor nx1 com os gradientes calculados numericamente relativos aos pesos

def varer_dados(pesos,X,y,arquitetura):     #recebe um dic com os pesos e arquitetura alem dos pontos X e y para fazer a varredura
    fw_prop = []  # lista com os valores de foward prop para todos os pontos
    bk_prop = []  # lista com os valores de backward prop para todos os pontos
    grad_acumulado = {}  # acumula os valores de gradiente calculados

    # inicia os vetores de atualização com 0
    for l in range(1, arquitetura['num_layers']):
        grad_acumulado[f'w_{l}'] = np.zeros((arquitetura[f'n_L{l + 1}'], arquitetura[f'n_L{l}']))
        grad_acumulado[f'b_{l}'] = np.zeros((arquitetura[f'n_L{l + 1}'], 1))
    #calcula fw, bk e o gradiente varendo todos os dados

    for i in range(len(y)):
        fw_prop.append(forward_prop(pesos, X[i], arquitetura))      #salva os dados da fw prop
        bk_prop.append(back_prop(pesos, fw_prop[i], y[i], arquitetura)) #salva os dados da bk prop
        for l in range(1, arquitetura['num_layers']):
            grad_acumulado[f'w_{l}'] = grad_acumulado[f'w_{l}'] + bk_prop[i][f'grad_{l}']
            grad_acumulado[f'b_{l}'] = grad_acumulado[f'b_{l}'] + bk_prop[i][f'grad_b_{l}']
    # ----gradiente
    grad_final = {}
    for l in range(1, arquitetura['num_layers']):
        grad_final[f'D_w_{l}'] = (1 / len(y)) * (grad_acumulado[f'w_{l}'] + (Lambda * pesos[f'w_{l}']))  ###regulariação - verificar se esta certo
        grad_final[f'D_b_{l}'] = (1 / len(y)) * grad_acumulado[f'b_{l}']
    return fw_prop, bk_prop, grad_final         #retorna vetores com os dicionarios equivalentes a cada ponto

def predicts(fw_prop,y,arquiteruta):  #recebe o vetor com os dicionarios com o valor da saida e y equivalente
    predict= []
    hip =[]
    l_max = arquiteruta['num_layers']
    for i in range(len(y)):
        hip.append(fw_prop[i][f'A_{l_max}'])
        if fw_prop[i][f'A_{l_max}']>= 0.5:
            predict.append(1)
        else:
            predict.append(0)

    return hip,predict  #retorna hipotese e as predições
def accuracy(pesos,y_vali,x_vali):  #calcula acuracia
    total =0
    forward, _, _ =varer_dados(pesos,x_vali,y_vali,arquitetura)
    _ , guess = predicts(forward,y_vali,arquitetura)
    for i in range(len(y_vali)):
        if guess[i]== y_vali[i]:
            total += 1
    return total/len(y_vali)
def grad_desc(pesos,grad_final,alfa,arquitetura):   #aplica o gradiente descent na atualização dos pesos

    pesos_atualizados = {}

    # ----gradiente

    for l in range(1, arquitetura['num_layers']):
        pesos_atualizados[f'w_{l}'] = pesos[f'w_{l}'] - alfa*grad_final[f'D_w_{l}']
        pesos_atualizados[f'b_{l}'] = pesos[f'b_{l}'] - alfa*grad_final[f'D_b_{l}']

    return pesos_atualizados        #retorna pesos atualizados
def imprime_prova_real(pesos):      #imprime a comparação dos gradientes calculados numericamente e do back propagation

    fw_prop, bk_prop, grad_final = varer_dados(pesos, X, y, arquitetura)


    # print(prova_real(pesos, X, y, arquitetura))
    a = []
    # --- empilha o vetor gradiente para comparação
    for l in range(1, arquitetura['num_layers']):
        a.append(grad_final[f'D_b_{l}'].reshape(pesos[f'b_{l}'].size, 1))
        a.append(grad_final[f'D_w_{l}'].reshape(pesos[f'w_{l}'].size, 1))
    comparação = np.vstack(a)
    numerico = prova_real(pesos, X, y, arquitetura)
    for i in range(len(comparação)):
        print("back_prop:", comparação[i], 'numerico:', numerico[i])
        print('erro:',(comparação[i] - numerico[i])/numerico[i]*100,'%')

def print_front(frame):         #imprime a fronteira de decisão
    pesos = frame[0]
    epoca = frame[1]
    # ------ Plotando fronteira de decisão
    x1s = np.linspace(-1, 1.5, 50)
    x2s = np.linspace(-1, 1.5, 50)
    z = np.zeros((len(x1s), len(x2s)))
    ax.clear()
    print_points()
    acu = round(accuracy(pesos, y_validation, X_validation) * 100, 2)

    for i in range(len(x1s)):
        for j in range(len(x2s)):
            x = np.array([x1s[i], x2s[j]]).reshape(2, -1)
            lmax = arquitetura['num_layers']
            a = forward_prop(pesos, x, arquitetura)
            z[i, j] = a[f'Z_{lmax}'].item()  
    plt.contour(x1s, x2s, z.T, 0)
    plt.title(f'Época = {epoca}', loc='right',fontweight='bold')
    plt.title(f'Acuracia nos dados de Validação = {acu}%', loc='left', fontsize=12)
    plt.xlabel("x1")
    plt.ylabel("x2")

def animation(frames):              #cria um gif com a evolução do resultado no tempo
    n_layers = arquitetura['num_layers']
    num_HN = arquitetura['n_L2']

    anim = FuncAnimation(fig, print_front, frames=frames)
    print_points()
    plt.show()
    #anim.save(f'Animações/ep{epochs}_layrs={n_layers}_Hn{num_HN}_alfa{alfa}_lambd{Lambda}_Ts_{ratio_de_dados}.gif', dpi=300, writer=PillowWriter(fps=10))

def print_points():     #printa os pontos x1 e x2
    # --- plota o scatter dos dados iniciais


    #scatter = plt.scatter(x1,x2,c=y)
    scatter_train = plt.scatter(x1_train, x2_train, c=y_train,marker='o',label='Dados de Treino')
    scatter_vali = plt.scatter(x1_vali, x2_vali, c=y_validation, marker='*',label='Dados de validação')

    legend_cores = plt.legend(handles=scatter_train.legend_elements()[0],title='Valores Reais', labels=['0', '1'],bbox_to_anchor=(1, 0),loc = 'lower left')

    legend_estilos = plt.legend(handles=[
        Line2D([0], [0], marker='o', color='w', markerfacecolor='white', markersize=10, label='Treino',
               markeredgecolor='black'),
        Line2D([0], [0], marker='*', color='w', markerfacecolor='white', markersize=10, label='Validação',
               markeredgecolor='black')],
                                title='Dados:',bbox_to_anchor=(1, 1),loc =  'upper left')
    #plt.legend(handles=scatter.legend_elements()[0], labels=['1','0'])

    ax.add_artist(legend_cores)
    ax.add_artist(legend_estilos)

    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.subplots_adjust(right=0.78)


## ------ Execução principal:
alfa = 0.5                    #learning rate
Lambda =0.0                   #regularização
pesos = init_weights(arquitetura)
epochs = 2




frame =[]               #usado para gerar a animação


for i in range(epochs):
    fw_prop, bk_prop, grad_final = varer_dados(pesos, X_train, y_train, arquitetura)
    pesos = grad_desc(pesos,grad_final, alfa, arquitetura)
    if i % 5 ==0:
        frame.append([pesos,i])



#------------Printa figura
fig,ax =plt.subplots(figsize=(8.5, 5))
n_layers = arquitetura['num_layers']
num_HN = arquitetura['n_L2']
print_points()
print_front([pesos,epochs])
#plt.savefig(f'Image/ep{epochs}_layrs={n_layers}_Hn{num_HN}_alfa{alfa}_lambd{Lambda}_Ts_{ratio_de_dados}.png')
#plt.show()
#------------Printa a prova real numérica do calculo do gradiente

imprime_prova_real(pesos)
#------------printa a acuracia

#print(accuracy(pesos,y_validation,X_validation))
#------------Printa somente os pontos

#print_points()

#------------ Animação
#fig,ax =plt.subplots(figsize=(8.5, 5))
#animation(frame)