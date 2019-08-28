from Test import *

class Sigmoide:
	
	def __init__(self):
		self.e = 2.718281828459045235360
	
	def evaluar(self, x):
		e = self.e
		return 1 / (1 + e ** -x)
		
	def derivada(self, x):
		e = self.e ** -x
		return e / (1 + e) ** 2
	
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
		
def testFuncionSigmoide():
	print('TEST SIGMOIDE')
	f = Sigmoide()
	r = f.evaluar(0)
	Test.test(r,0.5)
	
def testFuncionSigmoideD():
	print('TEST SIGMOIDE DERIVADA')
	f = Sigmoide()
	r = f.derivada(0)
	Test.test(r,0.25)
	
def testFuncionTanh():
	print('TEST TANH')
	f = Tanh()
	r = f.evaluar(0)
	Test.test(r,0)
	
def testFuncionTanhD():
	print('TEST TANH DERIVADA')
	f = Tanh()
	r = f.derivada(0)
	Test.test(r,1)
	
def testFuncionRelu():
	print('TEST RELU')
	f = Relu()
	r = f.evaluar(0)
	Test.test(r,0)
	r = f.evaluar(-10)
	Test.test(r,0)
	r = f.evaluar(100)
	Test.test(r,100)
	
def testFuncionReluD():
	print('TEST RELU DERIVADA')
	f = Relu()
	r = f.derivada(0)
	Test.test(r,0)
	r = f.derivada(-10)
	Test.test(r,0)
	r = f.derivada(100)
	Test.test(r,1)

testFuncionSigmoide()
testFuncionTanh()
testFuncionRelu()
testFuncionSigmoideD()
testFuncionTanhD()
testFuncionReluD()
