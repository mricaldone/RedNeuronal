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
		return np.random.rand(self.cant_entradas)
		
	def _generar_bias(self):
		return np.random.rand()
	
	def procesar(self, entradas):
		self.vector_x = entradas
		self.z = self.b
		#print(self.vector_x, self.vector_w)
		for x, w in zip(self.vector_x, self.vector_w):
			self.z = self.z + x * w
		self.y = self.f_activ.evaluar(self.z)
		#print(self.y)
		return self.y
	
	def obtener_entradas(self):
		return self.vector_x
	
	def obtener_pesos(self):
		return self.vector_w
	
	def obtener_bias(self):
		return self.b
	
	def obtener_z(self):
		return self.z
	
	def obtener_derivada_activacion(self):
		return self.f_activ.derivada(self.z)
	
	def actualizar_pesos(self, w):
		self.vector_w = w
		
	def actualizar_bias(self, b):
		self.b = b