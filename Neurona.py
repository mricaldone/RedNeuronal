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
		return np.random.rand() * 2 - 1
		#return 1
		#return 0
	
	def procesar(self, entradas):
		self.vector_x = entradas
		self.z = self.b
		for x, w in zip(self.vector_x, self.vector_w):
			self.z = self.z + x * w
		self.y = self.f_activ.evaluar(self.z)
		return self.y
		
	def actualizar_pesos(self, delta, learning_rate):
		#LOS NUEVOS PESOS SE CALCULAN COMO W = W - (dC/dw) * LR = W - deltas * (dz/dw) * LR = W - deltas * entradas * LR
		for i in range(len(self.vector_w)):
			self.vector_w[i] = self.vector_w[i] - delta * self.vector_x[i] * learning_rate
		
	def actualizar_bias(self, delta, learning_rate):
		#LOS NUEVOS BIAS SE CALCULAN COMO b = b - (dC/db) * LR = b - deltas * (dz/db) * LR = b - deltas * 1 * LR
		self.b = self.b - delta * learning_rate
	
	def generar_delta(self, delta):
		#LOS NUEVOS DELTAS SE CALCULAN COMO d = d * (da/dz) = d * a´(z)
		return delta * self.f_activ.derivada(self.z)
		
	def generar_deltas_capa_siguiente(self, delta):
		#LOS DELTAS SIGUIENTES SE CALCULAN COMO d = d * (dz/dx) = d * w
		deltas = []
		for w in self.vector_w:
			deltas.append(delta * w)
		return deltas 
