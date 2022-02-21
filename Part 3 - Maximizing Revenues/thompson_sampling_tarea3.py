# Inteligencia Artifical aplicada a Negocios y Empresas
# Maximizaci�n de beneficios de una empresa de venta online con Muestreo de Thompson

#https://www.udemy.com/course/inteligencia-artificial-aplicada-a-negocios-y-empresas/learn/practice/1172458/instructor-solution#questions

# Importar las librer��as
import numpy as np
import matplotlib.pyplot as plt
import random

# Configuraci�n de los par�metros
#Numero de simulaciones o clientes o rondas
N = 10000
#Numero de estrategias
d = 9

# Creaci�n de la simulaci�n
#Damos aquí cuales son los ratios de conversión , o sea, la probabilidad de exito de la estrategia.
# conversion_rates = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
conversion_rates = [0.05, 0.13, 0.09, 0.16, 0.11, 0.04, 0.20, 0.08, 0.01]
max_conversion_rates=N*np.array(conversion_rates)

#Creamos el array de Nxd 
#Lo que hacemos es rellenar todo con ceros.
#Después recorremos el array fila a fila y para cada fila recorremos las d columnas 
#En cada una de ellas tiramos un numero al azar y si el numero es <= que el marcado entonces es cierto 1
X = np.array(np.zeros([N, d]))
for i in range(N):
    for j in range(d):
        if np.random.rand() <= conversion_rates[j]:
            X[i,j] = 1
            
            
# Implementaci�n de la Selecci�n Aleatoria y el Muestreo de Thompson
#Aqui guardaremos los resultados de cada ronda para la seleccion aleatoria y el muestreo de thompson
strategies_selected_rs = []
strategies_selected_ts = []
best_strategies_selected=[]
total_reward_rs_cum=[]
total_reward_ts_cum=[]
regret1 = []
regret2 = []
#Aqui guardaremos los resultados de las recompensas
total_reward_rs = 0
total_reward_ts = 0
#Cuantas veces se ha optenido una recompesa por cada uno de los anuncios y cuando no se ha optenido recompensa
#genera una lista con 9 ceros, o sea, d ceros
number_of_rewards_1 = [0] * d
number_of_rewards_0 = [0] * d
rewards_strategies = [0] * d
for n in range(0, N):
    # Best Strategy
    for i in range(0, d):
        rewards_strategies[i] = rewards_strategies[i] + X[n, i]
    
    # Selecci�n Aleatoria. Paso 1 (Realizamos los pasos para la selecci�n aleatoria)
    #Seleccionamos una estrategia al azar y la guardamos en la lista
    strategy_rs = random.randrange(d)
    strategies_selected_rs.append(strategy_rs)
    #vemos el valor de su recompensa
    reward_rs = X[n, strategy_rs]
    #Guardamos el valor de la recompensa
    total_reward_rs += reward_rs

    # Muestreo de Thompson Paso 1 (Realizamos los pasos para el muestreo de thompson)
    strategy_ts = 0
    max_random = 0
    #Hay que elegir cada una de las estrategias para seleccionar el valor Beta
    for i in range(0, d):
    
        #Paso1:
        #Calculamos el valor Beta para ese anuncio. La distribucción Beta https://es.wikipedia.org/wiki/Distribuci%C3%B3n_beta 
        # para su calculo necesita 2 valores
        #https://www.geeksforgeeks.org/random-betavariate-method-in-python/
        #El numero de anuncios con exito+1 y el numero de anuncios sin exito+1
        #Ver en el resumen el paso1 del algoritmo del muestreo de thompson
        random_beta = random.betavariate(number_of_rewards_1[i]+1, 
                                         number_of_rewards_0[i]+1)
        #Paso2
        #Tenemos que pasearnos por todas las estrategias y guardamos el valor Beta calculado.
        #Nos quedamos con el mayor y en que estrategia lo hemos encontrado.
        if random_beta > max_random: 
            max_random = random_beta
            strategy_ts = i
        
            
    #Paso3
    #En este paso actualizamos el contador de recompensas de la estrategia seleccionada para una simulacion n        
    reward_ts = X[n, strategy_ts]
    if reward_ts == 1:
        number_of_rewards_1[strategy_ts] += 1
    else:
        number_of_rewards_0[strategy_ts] += 1
    strategies_selected_ts.append(strategy_ts)
    total_reward_ts += reward_ts
    total_reward_rs_cum.append(total_reward_rs)
    total_reward_ts_cum.append(total_reward_ts)
    best_strategies_selected.append(max(total_reward_ts, total_reward_rs))

    total_reward_bs = max(rewards_strategies)  
    regret1.append(total_reward_bs - total_reward_rs)      
    regret2.append(total_reward_bs - total_reward_ts)  
        
    
    
    
# Calcular le retorno relativo y absoluto. 
#Suponemos que cada suscripcion se vende a 100 dolares
absolute_return = (total_reward_ts - total_reward_rs)*100
relative_return = (total_reward_ts - total_reward_rs) / total_reward_rs * 100
print("Rendimiento Absoluto: {:.0f} $".format(absolute_return))
print("Rendimiento Relativo: {:.0f} %".format(relative_return))
    
# Representaci�n del histograma de selecciones
#plt.plot(regret1, label="aleatoria")
#plt.plot(regret2, label="thompson")
plt.plot(total_reward_bs, label="total")
#max_conversion_rates=max_conversion_rates.tolist()
#plt.hist(max_conversion_rates)
plt.legend(loc = 'upper right')
plt.title("Curva de arrepentimiento de la estrategia thompson")
#plt.xlabel("Estrategia")
#plt.ylabel("Numero de veces que se ha seleccionado la estrategia")
plt.show()








