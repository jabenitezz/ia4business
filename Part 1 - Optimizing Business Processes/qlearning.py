# Inteligencia Artificial Aplicada a Negocios y Empresas
# Parte 1 - Optimizaci�n de los flujos de trabajo en un almacen con Q-Learning

# Importaci�n de las librer��as
import numpy as np

# Configuraci�n de los par�metros gamma y alpha para el algoritmo de Q-Learning
gamma = 0.75
alpha = 0.9

# PARTE 1 - DEFINICI�N DEL ENTORNO

# Definici�n de los estados
location_to_state = {'A': 0,
                     'B': 1,
                     'C': 2,
                     'D': 3,
                     'E': 4,
                     'F': 5,
                     'G': 6, 
                     'H': 7, 
                     'I': 8,
                     'J': 9,
                     'K': 10,
                     'L': 11}

# Definici�n de las acciones
actions = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]

# Definici�n de las recompensas
# Columnas:    A,B,C,D,E,F,G,H,I,J,K,L
R = np.array([[0,1,0,0,0,0,0,0,0,0,0,0], # A
              [1,0,1,0,0,1,0,0,0,0,0,0], # B
              [0,1,0,0,0,0,1,0,0,0,0,0], # C
              [0,0,0,0,0,0,0,1,0,0,0,0], # D
              [0,0,0,0,0,0,0,0,1,0,0,0], # E
              [0,1,0,0,0,0,0,0,0,1,0,0], # F
              [0,0,1,0,0,0,1,1,0,0,0,0], # G
              [0,0,0,1,0,0,1,0,0,0,0,1], # H
              [0,0,0,0,1,0,0,0,0,1,0,0], # I
              [0,0,0,0,0,1,0,0,1,0,1,0], # J
              [0,0,0,0,0,0,0,0,0,1,0,1], # K
              [0,0,0,0,0,0,0,1,0,0,1,0]])# L

# PARTE 2 - CONSTRUCCI�N DE LA SOLUCI�N DE IA CON Q-LEARNING

# Transformaci�n inversa de estados a ubicaciones
state_to_location = {state : location for location, state in location_to_state.items()}

# Crear la funci�n final que nos devuelva la ruta �ptima
def route(starting_location, ending_location):
    R_new = np.copy(R)
    ending_state = location_to_state[ending_location]
    R_new[ending_state, ending_state] = 1000
    

    Q = np.array(np.zeros([12, 12]))
    for i in range(1000):
        current_state = np.random.randint(0, 12)
        playable_actions = []
        for j in range(12):
            if R_new[current_state, j] > 0:
                playable_actions.append(j)
        next_state = np.random.choice(playable_actions)
        TD = R_new[current_state, next_state] + gamma*Q[next_state, np.argmax(Q[next_state,])] - Q[current_state, next_state]
        Q[current_state, next_state] = Q[current_state, next_state] + alpha*TD

    
    
    route = [starting_location]
    next_location = starting_location
    while(next_location != ending_location):
        starting_state = location_to_state[starting_location]
        next_state = np.argmax(Q[starting_state, ])
        next_location = state_to_location[next_state]
        route.append(next_location)
        starting_location = next_location
    return route







def route_intermediary(starting_location, intermediary_location, ending_location):
    R_new = np.copy(R)
    ending_state = location_to_state[ending_location]
    R_new[ending_state, ending_state] = 1000
    
    intermediary_state = location_to_state[intermediary_location]
    
    R_new[intermediary_state, intermediary_state] = 500

    Q = np.array(np.zeros([12, 12]))
    for i in range(1000):
        current_state = np.random.randint(0, 12)
        playable_actions = []
        for j in range(12):
            if R_new[current_state, j] > 0:
                playable_actions.append(j)
        next_state = np.random.choice(playable_actions)
        TD = R_new[current_state, next_state] + gamma*Q[next_state, np.argmax(Q[next_state,])] - Q[current_state, next_state]
        Q[current_state, next_state] = Q[current_state, next_state] + alpha*TD

    
    
    route = [starting_location]
    next_location = starting_location
    while(next_location != ending_location):
        starting_state = location_to_state[starting_location]
        next_state = np.argmax(Q[starting_state, ])
        next_location = state_to_location[next_state]
        route.append(next_location)
        starting_location = next_location
    return route











ruta=route_intermediary('E', 'B', 'G')
print("Ruta Elegida:")
print(ruta)



# PARTE 3 - PONER EL MODELO EN PRODUCCI�N
def best_route(starting_location, intermediary_location, ending_location):
    #Se quita el primer punto porque el punto intermedio se repite por eso se quita en la segunda.
    return route(starting_location, intermediary_location) + route(intermediary_location, ending_location)[1:]

# Imprimir la ruta final
print("Ruta Elegida:")
print(best_route('E', 'B', 'G'))










