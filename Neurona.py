import numpy
from Utils import *
from Test import *

class Neurona:
	
	def __init__(self, cant_entradas, factiv):
		'''
		CONSTRUCTOR DE LA NEURONA
		PARAMETROS:
			CANT_ENTRADAS: CANTIDAD DE ENTRADAS DE LA NEURONA (INT)
			FACTIV: FUNCION DE ACTIVACION DERIVABLE
		'''
		self.cant_entradas = cant_entradas
		self.coeficientes = numpy.ones(self.cant_entradas)
		self.bias = 0
		self.factiv = factiv
		
	def procesar(self, entradas):
		'''
		PROCESA LAS ENTRADAS RECIBIDAS POR LA NEURONA. DEVUELVE UN UNICO RESULTADO
		PARAMETROS:
			ENTRADAS: VECTOR CON LOS PARAMETROS DE ENTRADA, DEBE SER DE LA LONGITUD ESPECIFICADA AL CREAR LA NEURONA
		'''
		#if len(entradas) != len(self.cant_entradas):
		#	raise 
		pi = producto_interno(entradas, self.coeficientes)
		return self.factiv.evaluar(pi + self.bias)

from Funciones import *

def testNeurona():
	print('TEST NEURONA')
	f = Relu()
	neurona = Neurona(2, f)
	x1 = 5
	x2 = 7
	r = neurona.procesar([x1,x2])
	Test.test(r,12)
	
testNeurona()
		
