import numpy as np

class Neurona:
	
	def __init__(self, cant_entradas, f_activacion, f_costo):
		self.cant_entradas = cant_entradas
		self.f_activ = f_activacion
		self.f_costo = f_costo
		self.vector_w = self._generar_vector_w()
		self.b = self._generar_bias()
		self.vector_x = None
		self.y = None
		self.z = None
		
	def _generar_vector_w(self):
		return np.random.rand(self.cant_entradas)
		
	def _generar_bias(self):
		return np.random.rand()
	
	def procesar(self, entradas):
		self.vector_x = entradas
		self.z = self.b
		for x, w in zip(self.vector_x, self.vector_w):
			self.z = self.z + x * w
		self.y = self.f_activ.evaluar(self.z)
		return self.y
	
	#PROXIMA A RETIRAR:
	def obtener_pesos(self):
		return self.vector_w
	
	def actualizar_pesos(self, delta, learning_rate):
		#LOS NUEVOS PESOS SE CALCULAN COMO W = W - (dC/dW) * LR = W - deltas * (da/dZ) * (dZ/dW) * LR = W - deltas * a´(Z) * entradas * LR
		for i in range(len(self.vector_w)):
			self.vector_w[i] = self.vector_w[i] - delta * self.f_activ.derivada(self.z) * self.vector_x[i] * learning_rate
		
	def actualizar_bias(self, delta, learning_rate):
		#LOS NUEVOS BIAS SE CALCULAN COMO b = b - (dC/db) * LR = b - deltas * (da/dZ) * (dZ/db) * LR = b - deltas * a´(Z) * LR
		self.b = self.b - delta * self.f_activ.derivada(self.z) * learning_rate
		
	def generar_delta(self, valor_esperado):
		#EL DELTA SE CALCULA COMO d = (dC/da) * (da/dz) = C'(a) * a'(z)
		#DERIVADA DEL COSTE
		dc = self.f_costo.derivada(valor_esperado, self.y)
		#DERIVADA DE LA ACTIVACION
		da = self.f_activ.derivada(self.z)
		return da * dc