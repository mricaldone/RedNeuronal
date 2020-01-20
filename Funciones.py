import numpy as np

class Sigmoide:
	
	def evaluar(self, x):
		return 1 / (1 + np.exp(-x))
		
	def derivada(self, x):
		t = self.evaluar(x)
		return t - np.multiply(t,t)
	
class Tanh:
	
	def evaluar(self, x):
		return np.tanh(x)
	
	def derivada(self, x):
		tanh = self.evaluar(x)
		return 1 -  np.multiply(tanh, tanh)
		
class Relu:
	
	def evaluar(self, x):
		return np.matrix(np.where(x > 0, x, 0))
	
	def derivada(self, x):
		return np.where(x > 0, 1, 0)
	
class LeakyRelu:
	
	def __init__(self, coeficiente):
		self.a = coeficiente
	
	def evaluar(self, x):
		return np.matrix(np.where(x > 0, self.a * x, 0))
	
	def derivada(self, x):
		return np.where(x > 0, self.a, 0)
	
class ErrorCuadraticoMedio:
	
	def evaluar(self, x, y):
		#Y: VALOR REAL. X: VALOR ESTIMADO
		return 0.5 * (y - x) ** 2
	
	def derivada(self, x, y):
		return x - y
