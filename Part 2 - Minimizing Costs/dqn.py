# Inteligencia Artifical aplicada a Negocios y Empresas - Caso Práctico 2
# Creación de la red Q profunda

# Importar las librerías
import numpy as np

# IMPLEMENTAR EL ALGORITMO DE DEEP Q-LEARNING CON REPETICIÓN DE EXPERIENCIA

class DQN(object):
    
    # INTRODUCCIÓN E INICIALIZACIÓN DE LOS PARÁMETROS Y VARIABLES DEL DQN
    def __init__(self, max_memory = 100, discount_factor = 0.9):
        #Paramentros
        # max_memory : Numero de experiencias que queremos recordar para hacer repetición de experiencias
        # discount_factor : Factor de descuento en cada iteracion. 
        # Lo han puesto a este valor porque es el que más se ajusta a la experiencia
        self.memory = list()
        self.max_memory = max_memory
        self.discount_factor = discount_factor
        
    # CREACIÓN DE UN MÉTODO QUE CONSTRUYA LA MEMORIA DE LA REPETICIÓN DE EXPERIENCIA
    def remember(self, transition, game_over):
        #Va guardando transiciones en la memoria y el estado game_over
        self.memory.append([transition, game_over])
        #Se elimina el más antiguo si se llena
        if len(self.memory) > self.max_memory:
            del self.memory[0]
        
    # CREACIÓN DE UN MÉTODO QUE CONSTRUYA DOS BLOQUES DE ENTRADAS Y TARGETS EXTRATENDO TRANSICIONES
    def get_batch(self, model, batch_size = 10):
        #Paremetros
        # model: es el modelo de la red neuronal que lo utilizaremos para predecir
        # batch_size: numero de experiencias que queremos pasar al modelo y recuperar de la memoria
        len_memory = len(self.memory)
        #Cada elemento de la memoria (predicción) es una tupla de 3 elementos. El estado actual, la acción, y la recompensa.
        #se calcula con el primer elemento de la memoria. Ese elemento es una tupla de transition, game_over] y cojemos transition.
        #De esta transition cojer el primer elemento que es el estado, que contiene los tres elementos. (al hacerlo de esta forma es dinamico)
        #que sería un estado [scaled_temperature_ai, scaled_number_users, scaled_rate_data]
        num_inputs = self.memory[0][0][0].shape[1]
        #Del modelo recuperamos la última capa y sus dimensiones que será el tamaño de la salida.
        num_outputs = model.output_shape[-1]
        #Tenemos que crear unas matrices para el estado de la entrada y la salida.
        #Se crean unas matrices con una dimensión de el minimo entre la long de la memorioa y el batch_size. al principio puede no estar
        #llena la memoria.
        inputs = np.zeros((min(batch_size, len_memory), num_inputs))        
        targets = np.zeros((min(batch_size, len_memory), num_outputs))
        #Un bucle para extraer una serie de elementos de forma aleatoria de la memoria y con una dimensión de entre min(len_memory, batch_size)
        #esto se devuelve a enumerate que el primer elemento es el indice de la extración y el segundo el indice de la memoria
        for i, idx in enumerate(np.random.randint(0, len_memory, size=min(len_memory, batch_size))):
            current_state, action, reward, next_state = self.memory[idx][0]
            # --> Esto es el valor de la 
            #transicion self.memory.append([transition, game_over])
            game_over = self.memory[idx][1]
            inputs[i] = current_state
            #La predección de un modelo devuelve la predicción en el primer valor y luego una serie de valore propios de la predicción
            targets[i] = model.predict(current_state)[0]
            #De todas las acciones nos quedamos con la de mejor valor.
            Q_sa = np.max(model.predict(next_state)[0])
            if game_over:
                targets[i, action] = reward
            else:
                targets[i, action] = reward + self.discount_factor*Q_sa
        return inputs, targets
            
            
            
            