import numpy as np

class Neurona:
	
	def __init__(self, cant_entradas, f_activacion):
		self.cant_entradas = cant_entradas
		self.f_activ = f_activacion
		self.vector_w = self._generar_vector_w()
		self.b = self._generar_bias()
		self.vector_x = None
		self.y = None
		self.z = None
		
	def _generar_vector_w(self):
		return np.random.rand(self.cant_entradas) * 2 - 1
		#return [1] * self.cant_entradas
		#return [0] * self.cant_entradas
		
	def _generar_bias(self):
		#return np.random.rand() * 2 - 1
		#return 1
		return 0
	
	def _generar_delta(self, delta):
		#LOS NUEVOS DELTAS SE CALCULAN COMO d = d * (da/dz) = d * a´(z)
		return delta * self.f_activ.derivada(self.z)
		
	def _generar_delta_capa_siguiente(self, i, delta):
		#LOS DELTAS SIGUIENTES SE CALCULAN COMO d = d * (dz/dx) = d * w
		return delta * self.vector_w[i]
	
	def _actualizar_pesos(self, i, delta, learning_rate):
		#LOS NUEVOS PESOS SE CALCULAN COMO W = W - (dC/dw) * LR = W - deltas * (dz/dw) * LR = W - deltas * entradas * LR
		self.vector_w[i] -= delta * self.vector_x[i] * learning_rate
	
	def _actualizar_bias(self, delta, learning_rate):
		#LOS NUEVOS BIAS SE CALCULAN COMO b = b - (dC/db) * LR = b - deltas * (dz/db) * LR = b - deltas * 1 * LR
		self.b -= delta * learning_rate
	
	def procesar(self, entradas):
		'''
		PROCESA LAS ENTRADAS RECIBIDAS POR LA NEURONA. DEVUELVE UN UNICO RESULTADO
		PARAMETROS:
			ENTRADAS: VECTOR CON LOS PARAMETROS DE ENTRADA, DEBE SER DE LA LONGITUD DEFINIDA AL CREAR LA NEURONA
		COMPLEJIDAD: O(m) m:numero de entradas
		'''
		self.vector_x = entradas
		self.z = self.b + np.dot(self.vector_x, self.vector_w)
		self.y = self.f_activ.evaluar(self.z)
		return self.y
	
	def entrenar(self, delta, learning_rate, nuevos_deltas):
		'''
		ENTRENA A LA NEURONA SEGUN CIERTO DELTA DADO Y LEARNING RATE
		PARAMETROS:
			DELTA: UN VALOR CON EL RESULTADO DE LAS DERIVADAS PARCIALES DE LAS CAPAS SIGUIENTES DE LA RED. (DOUBLE)
			LEARNING_RATE: VELOCIDAD DE APRENDIZAJE. UN LR ALTO IMPLICA UNA MAYOR VELOCIDAD PARA ENCONTRAR EL RESULTADO, SIN EMBARGO PUEDE NO LLEGAR AL RESULTADO OPTIMO. (DOUBLE)
		COMPLEJIDAD: O(m) m:numero de entradas
		'''
		#GENERO EL DELTA DE ESTA CAPA
		#LOS NUEVOS DELTAS SE CALCULAN COMO d = d * (da/dz) = d * a´(z)
		delta = self._generar_delta(delta)
		for i in range(self.cant_entradas):
			#CALCULO LOS DELTAS PARA LA CAPA SIGUIENTE
			#LOS DELTAS SIGUIENTES SE CALCULAN COMO d = d * (dz/dx) = d * w
			nuevos_deltas[i] += self._generar_delta_capa_siguiente(i, delta)
			#CALCULO LOS NUEVOS PESOS
			#LOS NUEVOS PESOS SE CALCULAN COMO W = W - (dC/dw) * LR = W - deltas * (dz/dw) * LR = W - deltas * entradas * LR
			self._actualizar_pesos(i, delta, learning_rate)
		#CALCULO LOS NUEVOS BIAS
		#LOS NUEVOS BIAS SE CALCULAN COMO b = b - (dC/db) * LR = b - deltas * (dz/db) * LR = b - deltas * 1 * LR
		self._actualizar_bias(delta, learning_rate)
		
	def entrenarRapido(self, delta, learning_rate):
		'''
		ENTRENA A LA NEURONA SEGUN CIERTO DELTA DADO Y LEARNING RATE
		PARAMETROS:
			DELTA: UN VALOR CON EL RESULTADO DE LAS DERIVADAS PARCIALES DE LAS CAPAS SIGUIENTES DE LA RED. (DOUBLE)
			LEARNING_RATE: VELOCIDAD DE APRENDIZAJE. UN LR ALTO IMPLICA UNA MAYOR VELOCIDAD PARA ENCONTRAR EL RESULTADO, SIN EMBARGO PUEDE NO LLEGAR AL RESULTADO OPTIMO. (DOUBLE)
		COMPLEJIDAD: O(m) m:numero de entradas
		'''
		#GENERO EL DELTA DE ESTA CAPA
		#LOS NUEVOS DELTAS SE CALCULAN COMO d = d * (da/dz) = d * a´(z)
		delta = self._generar_delta(delta)
		#CALCULO LOS DELTAS PARA LA CAPA SIGUIENTE
		#LOS DELTAS SIGUIENTES SE CALCULAN COMO d = d * (dz/dx) = d * w
		nuevos_deltas = np.multiply(delta, self.vector_w)
		#CALCULO LOS NUEVOS PESOS
		#LOS NUEVOS PESOS SE CALCULAN COMO W = W - (dC/dw) * LR = W - deltas * (dz/dw) * LR = W - deltas * entradas * LR
		self.vector_w = np.subtract(self.vector_w, np.multiply(delta * learning_rate, self.vector_x))
		#CALCULO LOS NUEVOS BIAS
		#LOS NUEVOS BIAS SE CALCULAN COMO b = b - (dC/db) * LR = b - deltas * (dz/db) * LR = b - deltas * 1 * LR
		self.b -= delta * learning_rate
		return nuevos_deltas
	
	def obtener_pesos(self):
		return self.vector_w.tolist()
		
	def definir_pesos(self, pesos):
		self.vector_w = pesos
		
	def __str__(self):
		precision = 1
		string = 'N=f('
		for w, x in zip(self.vector_w, self.vector_x):
			string = string + '[' + str(round(w,precision)) + '] * [' + str(round(x,precision)) + '] + '
		string = string + '[' + str(round(self.b,precision)) + '])=' + str(round(self.y,precision))
		return string
