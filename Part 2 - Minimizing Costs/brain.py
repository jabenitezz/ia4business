# Inteligencia Artifical aplicada a Negocios y Empresas - Caso Práctico 2
# Creación del Cerebro

# Importar las librerías
from keras.layers import Input, Dense
from keras.models import Model
from keras.optimizers import Adam

# CONSTRUCCIÓN DEL CEREBRO

class Brain(object):
    #Cuando heredamos de object una clase estandar
    def __init__(self, learning_rate = 0.001, number_actions = 5):
        #Paramentros
        # learning_rate: --> radio de aprendizaje. Cuanto más pequeño es menos problemas tenemos de overfitting. 
        # Cambios mas pequeños entre iteracion e iteracion
        # number_actions: --> numero de acciones que puede tomar el cerebro
        self.learning_rate = learning_rate
        #El parametro de entrada es una matriz al menos de una columna con tres filas. Puede venir más de una columna con lo cual por eso
        #se deja el segundo argumento sin rellenar--> Se podría llegar a hacer un entremamiento por baches
        states = Input(shape = (3,))
        #Los valores de 64,32 en medio es en base a la experiencia. La funcion de activacion se utiliza la sigmoide para romper la linealidad
        #No el doble de usuarios da el doble de temperatura --> Por eso se utiliza la sigmoide o cualquier funcion de activacion.
        x = Dense(units = 64, activation = "sigmoid")(states)
        y = Dense(units = 32, activation = "sigmoid")(x)
        q_values = Dense(units = number_actions, activation = "softmax")(y)
        self.model = Model(inputs = states, output = q_values)
        #La funcion de perdidas va en funcion de los valores  a predecir. En este caso son 5 valores. También depende del optimizador. 
        #es una regresión y por lo tanto se utiliza mse
        self.model.compile(loss = "mse", optimizer = Adam(lr = learning_rate))