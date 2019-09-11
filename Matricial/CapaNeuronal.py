import numpy as np

class CapaNeuronal:

	def __init__(self, cant_neuronas, cant_entradas, f_activacion):
		'''
		CONSTRUCTOR DE LA CAPA NEURONAL
		PARAMETROS:
			CANT_NEURONAS: CANTIDAD DE NEURONAS DE LA CAPA (INT)
			CANT_ENTRADAS: CANTIDAD DE ENTRADAS DE LA CAPA (INT)
			FACTIV: FUNCION DE ACTIVACION DERIVABLE
		'''
		self.cant_entradas = cant_entradas
		self.cant_neuronas = cant_neuronas
		self.f_activacion = f_activacion
		self.matriz_w = self._generar_matriz_de_pesos()
		self.matriz_b = self._generar_matriz_de_bias()
		self_matriz_x = None
		self.matriz_z = None
		self.matriz_y = None
		return

	def _generar_matriz_de_pesos(self):
		return np.matrix(np.random.rand(self.cant_neuronas, self.cant_entradas) * 2 - 1)
		
	def _generar_matriz_de_bias(self):
		return np.matrix(np.zeros((self.cant_neuronas, 1)))
			
	def procesar(self, entradas):
		'''
		PROCESA LAS ENTRADAS RECIBIDAS POR LA CAPA NEURONAL. DEVUELVE UN VECTOR DE RESULTADOS (LA LONGITUD DEL VECTOR ES LA MISMA QUE LA CANTIDAD DE NEURONAS DE LA CAPA)
		PARAMETROS:
			ENTRADAS: VECTOR CON LOS PARAMETROS DE ENTRADA, DEBE SER DE LA LONGITUD ESPECIFICADA AL CREAR LA CAPA
		COMPLEJIDAD: O(n*m) n:numero de neuronas, m:numero de entradas
		'''
		#GUARDAMOS EL VECTOR X (ES DE Nx1)
		self.matriz_x = entradas
		#GENERAMOS EL VECTOR Z (ES DE Nx1)
		#HACEMOS WX=Z
		self.matriz_z = (self.matriz_w @ self.matriz_x) + self.matriz_b
		#GENERAMOS EL VECTOR Y (ES DE Nx1)
		#HACEMOS f(Z)=Y
		self.matriz_y = self.f_activacion.evaluar(self.matriz_z)
		return self.matriz_y
	
	def entrenar(self, deltas, learning_rate):
		'''
		ENTRENA A LAS NEURONAS DE LA CAPA SEGUN CIERTOS DELTAS DADOS Y LEARNING RATE
		PARAMETROS:
			DELTAS: UNA LISTA CON LOS RESULTADOS DE LAS DERIVADAS PARCIALES DE LAS CAPAS SIGUIENTES DE LA RED. DEBE HABER TANTAS COMO NEURONAS EN ESTA CAPA. (LIST(DOUBLES))
			LEARNING_RATE: VELOCIDAD DE APRENDIZAJE. UN LR ALTO IMPLICA UNA MAYOR VELOCIDAD PARA ENCONTRAR EL RESULTADO, SIN EMBARGO PUEDE NO LLEGAR AL RESULTADO OPTIMO. (DOUBLE)
		COMPLEJIDAD: O(n*m) n:numero de neuronas, m:numero de entradas
		'''
		#GENERO EL DELTA DE ESTA CAPA
		#LOS NUEVOS DELTAS SE CALCULAN COMO d = d * (da/dz) = d * a´(z)
		deltas = np.multiply(deltas, self.f_activacion.derivada(self.matriz_z))
		#CALCULO LOS DELTAS PARA LA CAPA SIGUIENTE
		#LOS DELTAS SIGUIENTES SE CALCULAN COMO d = d * (dz/dx) = d * w
		matriz_d = np.multiply(deltas, self.matriz_w)
		#CALCULO LOS NUEVOS PESOS
		#LOS NUEVOS PESOS SE CALCULAN COMO W = W - (dC/dw) * LR = W - deltas * (dz/dw) * LR = W - deltas * entradas * LR
		self.matriz_w -= (self.matriz_x @ deltas.T).T * learning_rate
		#CALCULO LOS NUEVOS BIAS
		#LOS NUEVOS BIAS SE CALCULAN COMO b = b - (dC/db) * LR = b - deltas * (dz/db) * LR = b - deltas * 1 * LR
		self.matriz_b -= deltas * learning_rate
		#EL RESULTADO ES LA SUMATORIA DE LOS ELEMENTOS DE CADA COLUMNA DE LA MATRIZ DE DELTAS
		return matriz_d.sum(0).T
