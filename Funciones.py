class Sigmoide:
	
	def __init__(self):
		self.e = 2.718281828459045235360
	
	def evaluar(self, x):
		e = self.e
		x = round(x, 2)
		return 1 / (1 + e ** -x)
		
	def derivada(self, x):
		t = self.evaluar(x)
		return t - t ** 2
	
class Tanh:
	
	def __init__(self):
		self.e = 2.718281828459045235360
	
	def evaluar(self, x):
		e = self.e
		return (e**x - e**-x) / (e**x + e**-x)
	
	def derivada(self, x):
		tanh = self.evaluar(x)
		return 1 - tanh * tanh
		
class Relu:
	
	def evaluar(self, x):
		return x if  x > 0 else 0
	
	def derivada(self, x):
		return 1 if x > 0 else 0
	
class ErrorCuadraticoMedio:
	
	def evaluar(self, x, y):
		#Y: VALOR REAL. X: VALOR ESTIMADO
		return 0.5 * (y - x) ** 2
	
	def derivada(self, x, y):
		return x - y
