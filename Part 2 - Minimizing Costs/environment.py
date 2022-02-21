# Inteligencia Artifical aplicada a Negocios y Empresas - Caso Práctico 2
# Creación del Entorno

# Importar las librerías
import numpy as np

# CONSTRUIR EL ENTORNO EN UNA CLASE

class Environment(object):
    
    # INTRODUCIR E INICIALIZAR LOS PARÁMETROS Y VARIABLES DEL ENTORNO
    def __init__(self, optimal_temperature = (18.0, 24.0), initial_month = 0, initial_number_users = 10, initial_rate_data = 60):
        #Temperatura media de la localización del CPD
        self.monthly_atmospheric_temperature = [1.0, 5.0, 7.0, 10.0, 11.0, 20.0, 23.0, 24.0, 22.0, 10.0, 5.0, 1.0]
        self.initial_month = initial_month
        #Temperatura actual del mes en curso
        self.atmospheric_temperature = self.monthly_atmospheric_temperature[initial_month]
        self.optimal_temperature = optimal_temperature
        #Valores extremos de temperatura, nº usuarios, velocidd de transferencia de datos
        self.min_temperature = -20
        self.max_temperature = 80
        self.min_number_users = 10
        self.max_number_users = 100
        #Numero de usuarios que pueden cambiar en un minuto
        self.max_update_users = 5
        self.min_rate_data = 20
        self.max_rate_data = 300
        #Numero máximo de incrementos de velocidad por minuto
        self.max_update_data = 10
        #Numero de usuario inciales
        self.initial_number_users = initial_number_users
        #Numero de usuarios actuales
        self.current_number_users = initial_number_users
        #Velocidad inicial y actual
        self.initial_rate_data = initial_rate_data
        self.current_rate_data = initial_rate_data
        
        #La temperatura intrinseca esta basada en la temperatura que hay el numero de usuarios y la transmisiónn de datos
        self.intrinsec_temperature = self.atmospheric_temperature + 1.25*self.current_number_users+1.25*self.current_rate_data
        #Temperatura de la IA
        self.temperature_ai = self.intrinsec_temperature
        #temperatura sin la IA (media del rango de temeperaturas al principio)
        self.temperature_noai = (self.optimal_temperature[0]+self.optimal_temperature[1])/2.0
        #Medimos la energia electrica gastada con IA Y sin IA
        self.total_energy_ai = 0.0
        self.total_energy_noai = 0.0
        #Recompensa que tenemos hasta este momento
        self.reward = 0.0
        #Esta variable nos indica si el juego ha terminado o no
        self.game_over = 0
        #Esta nos indica si la inteligencia artificial esta en entrenamiento o no
        self.train = 1
    
    # CREAR UN MÉTODO QUE ACTUALICE EL ENTORNO JUSTO DESPUÉS DE QUE LA IA EJECUTE UNA ACCIÓN
    #Cada minuuto se realiza la actualizacion del entorno
    def update_env(self, direction, energy_ai, month):
        #Parametros
        #  direction: si tiene que bajar o subir la temperatura, 1 es para calentar y 0 para enfriar
        #  energy_ai: energia que gasta la IA
        #  month: mes en el que nos encontramos, utilizamos el mes actual para corregir la temperatura con los factores atmosféricos
        # OBTENCIÓN DE LA RECOMPENSA
        
        # Calcular la energía gastada por el sistema de refrigeración del server sin IA
        #Hay tres casos
        energy_noai = 0 #Caso en que la temeratura esta dentro del rango (no hay ni que enfriar ni calentar)
        #si la temperatura es menor que la temperatura optima hay que calentar y en un rango de la diferencia. --> tiende a llevar la temperatura al extremo inferiror
        if(self.temperature_noai  < self.optimal_temperature[0]):
            #Gastamos esta temperatura en calentar
            energy_noai = self.optimal_temperature[0] - self.temperature_noai
            self.temperature_noai = self.optimal_temperature[0]
        #Mismo caso si esta por encima.    
        elif(self.temperature_noai > self.optimal_temperature[1]):
            energy_noai = self.temperature_noai - self.optimal_temperature[1]
            self.temperature_noai = self.optimal_temperature[1]
        
        # Calcular la recompensa  -> Calclulamos cuanta energia ahorra el sistema con IA
        self.reward = energy_noai - energy_ai
        # Escalar la recompensa. Una forma de escalar la recompensa seria divide la recompensa entre el maximo de energia gastada por el sistema sin IA, o sea entre los máximos
        #y minimos de la temperatura, grado arriba o abajo energia gastada --> Normalizar los valores --> Ponerlos entre 0 y 1. Con lo cual diviendo entre 100 ya queda escalado
        #pues lo divide entre 1000
        self.reward = 1e-3*self.reward
        
        # OBTENCIÓN DEL SIGUIENTE ESTADO
        
        # Actualizar la temperatura atmosférica
        self.atmospheric_temperature = self.monthly_atmospheric_temperature[month]
        # Actualizar el número de usuarios
        self.current_number_users += np.random.randint(-self.max_update_users, self.max_update_users)
        #Como es una simulación se calcula el nuevo numero de usuarios con un incremento de lo usuarios que pueden entrar o salir
        #en un minuto. De forma aleatoria        
        #Cuidamos que el numero de usuarios está dentro del rango máximo y minimo. 
        if(self.current_number_users < self.min_number_users):
            self.current_number_users = self.min_number_users
        elif(self.current_number_users > self.max_number_users):
            self.current_number_users = self.max_number_users
        # Actualizar la tasa de transferencia de datos
        #Idem para la tasa de transferencia de datos
        self.current_rate_data += np.random.randint(-self.max_update_data, self.max_update_data)
        if(self.current_rate_data < self.min_rate_data):
            self.current_rate_data = self.min_rate_data
        elif(self.current_rate_data > self.max_rate_data):
            self.current_rate_data = self.max_rate_data
        # Calcular la variación de temperatura intrínseca
        past_intrinsic_temperature =  self.intrinsec_temperature
        self.intrinsec_temperature = self.atmospheric_temperature + 1.25*self.current_number_users+1.25*self.current_rate_data
        delta_intrinsec_temperaure = self.intrinsec_temperature - past_intrinsic_temperature
        # Calcular la variación de temperatura causada por la IA
        if(direction==-1):
            delta_temperature_ai = -energy_ai
        elif(direction == 1):
            delta_temperature_ai = energy_ai
        # Calcular la nueva temperatura del server cuando hay IA conectada (la IA lo que hace es modificar la temperatura sin que se tuviese IA)
        self.temperature_ai += delta_intrinsec_temperaure + delta_temperature_ai
        # Calcular la nueva temperatura del server cuando no hay IA conectada
        self.temperature_noai += delta_intrinsec_temperaure
        
        # OBTENCIÓN DEL GAME OVER--> Llegamos a este estado siempre que no consigamo nuestro objetivo. En este caso es pasar de los 
        #rangos de temperatura de trabajo
        if(self.temperature_ai < self.min_temperature):
            if(self.train == 1):
                self.game_over = 1
            else:
                #Si estamos en modo de ejecución pasamos la temperatura a la del borde del intervalo y computamos esta energia
                #como gastada.
                self.total_energy_ai += self.optimal_temperature[0] - self.temperature_ai
                self.temperature_ai = self.optimal_temperature[0]
        if(self.temperature_ai > self.max_temperature):
            if(self.train == 1):
                self.game_over = 1
            else:
                self.total_energy_ai += self.temperature_ai - self.optimal_temperature[1]
                self.temperature_ai = self.optimal_temperature[1]
                
        # ACTUALIZAR LOS SCORES
        
        # Calcular la energía total gastada por la IA
        self.total_energy_ai += energy_ai
        # Calcular la energía total gastada por el sistema de refrigeración del server sin IA
        self.total_energy_noai += energy_noai
        
        # ESCALAR EL SIGUIENTE ESTADO
        #Realizamos una normalizacion de los datos
        scaled_temperature_ai = (self.temperature_ai - self.min_temperature)/(self.max_temperature - self.min_temperature)
        scaled_number_users = (self.current_number_users - self.min_number_users)/(self.max_number_users - self.min_number_users)
        scaled_rate_data = (self.current_rate_data - self.min_rate_data)/(self.max_rate_data - self.min_rate_data)
        #Creamos el objeto que sera la entrada a la red neuronal. un vector con tres elementos escalados. Las redes neuronales necesitan matrices
        next_state = np.matrix([scaled_temperature_ai, scaled_number_users, scaled_rate_data])
        
        # DEVOLVER EL SIGUIENTE ESTADO, RECOMPENSA Y GAME OVER
        return next_state, self.reward, self.game_over
    
    # CREAR UN MÉTODO QUE REINICIE EL ENTORNO
    def reset(self, new_month):
        self.atmospheric_temperature = self.monthly_atmospheric_temperature[new_month]
        self.initial_month = new_month
        self.current_number_users = self.initial_number_users
        self.current_rate_data = self.initial_rate_data
        self.intrinsec_temperature = self.atmospheric_temperature + 1.25*self.current_number_users+1.25*self.current_rate_data
        self.temperature_ai = self.intrinsec_temperature
        self.temperature_noai = (self.optimal_temperature[0]+self.optimal_temperature[1])/2.0
        self.total_energy_ai = 0.0
        self.total_energy_noai = 0.0
        self.reward = 0.0
        self.game_over = 0
        self.train = 1


    # CREAR UN MÉTODO QUE NOS DE EN CUALQUIER INSTANTE EL ESTADO ACTUAL, LA ÚLTIMA RECOMPENSA Y EL VALOR DE GAME OVER
    def observe(self):
        scaled_temperature_ai = (self.temperature_ai - self.min_temperature)/(self.max_temperature - self.min_temperature)
        scaled_number_users = (self.current_number_users - self.min_number_users)/(self.max_number_users - self.min_number_users)
        scaled_rate_data = (self.current_rate_data - self.min_rate_data)/(self.max_rate_data - self.min_rate_data)
        current_state = np.matrix([scaled_temperature_ai, scaled_number_users, scaled_rate_data])
        
        return current_state, self.reward, self.game_over
        
        
        
        
        
        
        
        
        
    
