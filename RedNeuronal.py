from CapaNeuronal import *
from Funciones import *
from Test import *

class RedNeuronal:
	
	def __init__(self, cant_entradas, estructura, factiv):
		'''
		CONSTRUCTOR DE LA RED NEURONAL
		PARAMETROS:
			CANT_ENTRADAS: CANTIDAD DE ENTRADAS DE LA RED (INT)
			ESTRUCTURA: VECTOR CUYAS COMPONENTES SON LA CANTIDAD DE NEURONAS POR CAPA (ARRAY(INT))
			FACTIV: FUNCION DE ACTIVACION DERIVABLE
		'''
		self.factiv = factiv
		self.cant_entradas = cant_entradas
		self.capas = []
		#A CONTINUACION SE CONSTRUYE LA ESTRUCTURA DE LA RED
		#LA CAPA INICIAL TENDRA LA MISMA CANTIDAD DE ENTRADAS QUE LA RED
		cant_entradas_de_la_capa = self.cant_entradas
		for cant_neuronas in estructura:
			capa_neuronal = CapaNeuronal(cant_neuronas, cant_entradas_de_la_capa, factiv)
			self.capas.append(capa_neuronal)
			#LA PROXIMA CAPA TENDRA TANTAS ENTRADAS COMO NEURONAS EN LA CAPA ACTUAL
			cant_entradas_de_la_capa = cant_neuronas
	
	def procesar(self, entradas):
		'''
		PROCESA LAS ENTRADAS A TRAVES DE LA RED. DEVUELVE UN VECTOR DE RESULTADOS (LA LONGITUD DEL VECTOR ES LA MISMA QUE LA CANTIDAD DE NEURONAS DE LA ULTIMA CAPA)
		PARAMETROS:
			ENTRADAS: VECTOR CON LOS PARAMETROS DE ENTRADA, DEBE SER DE LA LONGITUD ESPECIFICADA AL CREAR LA RED
		'''
		#if len(entradas) != len(self.cant_entradas):
		#	raise CantidadDeEntradasRedNeuronalError
		r = entradas
		for capa in self.capas:
			r = capa.procesar(r)
		return r
		
	def entrenar(self, entradas, valor_esperado):
		#r = self.procesar(entradas)
		#error = valor_esperado - r
		#delta = self.factiv.derivada(error)
		#for capa in self.capas[:-1]:
		#	capa.modificar_pesos(delta)
		return

def testRedNeuronal():
	print('TEST RED NEURONAL')
	f = Sigmoide()
	rn = RedNeuronal(2, [1], f)
	r = rn.procesar([4,6])
	Test.test(r,[f.evaluar(10)])
	
	rn = RedNeuronal(2, [2,1], f)
	r = rn.procesar([2,9])
	esp = [f.evaluar(f.evaluar(11) + f.evaluar(11))]
	Test.test(r,esp)
	print(r)
	print('PRUEBA COMPUERTA AND')
	rn = RedNeuronal(2, [1], f)
	for i in range(100):
		rn.entrenar([1,1],[1])
		rn.entrenar([1,0],[0])
		rn.entrenar([0,1],[0])
		rn.entrenar([0,0],[0])
	print(rn.procesar([1,1]))
	print(rn.procesar([1,0]))
	print(rn.procesar([0,1]))
	print(rn.procesar([0,0]))
	print('PRUEBA COMPUERTA OR')
	rn = RedNeuronal(2, [1], f)
	for i in range(100):
		rn.entrenar([1,1],[1])
		rn.entrenar([1,0],[1])
		rn.entrenar([0,1],[1])
		rn.entrenar([0,0],[0])
	print(rn.procesar([1,1]))
	print(rn.procesar([1,0]))
	print(rn.procesar([0,1]))
	print(rn.procesar([0,0]))
	print('PRUEBA COMPUERTA XAND')
	rn = RedNeuronal(2, [2,1], f)
	for i in range(100):
		rn.entrenar([1,1],[1])
		rn.entrenar([1,0],[0])
		rn.entrenar([0,1],[0])
		rn.entrenar([0,0],[1])
	print(rn.procesar([1,1]))
	print(rn.procesar([1,0]))
	print(rn.procesar([0,1]))
	print(rn.procesar([0,0]))
	print('PRUEBA COMPUERTA XOR')
	rn = RedNeuronal(2, [2,1], f)
	for i in range(100):
		rn.entrenar([1,1],[0])
		rn.entrenar([1,0],[1])
		rn.entrenar([0,1],[1])
		rn.entrenar([0,0],[0])
	print(rn.procesar([1,1]))
	print(rn.procesar([1,0]))
	print(rn.procesar([0,1]))
	print(rn.procesar([0,0]))
		
testRedNeuronal()
