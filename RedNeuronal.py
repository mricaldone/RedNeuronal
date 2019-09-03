from CapaNeuronal import *
from Funciones import *
from Test import *

class RedNeuronal:
	
	def __init__(self, cant_entradas, estructura, f_activacion):
		'''
		CONSTRUCTOR DE LA RED NEURONAL
		PARAMETROS:
			CANT_ENTRADAS: CANTIDAD DE ENTRADAS DE LA RED (INT)
			ESTRUCTURA: VECTOR CUYAS COMPONENTES SON LA CANTIDAD DE NEURONAS POR CAPA (ARRAY(INT))
			FACTIV: FUNCION DE ACTIVACION DERIVABLE
		'''
		self.factiv = f_activacion
		self.fcosto = ErrorCuadraticoMedio()
		self.cant_entradas = cant_entradas
		self.capas = self._generar_capas(estructura)
		return
	
	def _generar_capas(self, estructura):
		#A CONTINUACION SE CONSTRUYE LA ESTRUCTURA DE LA RED
		#LA CAPA INICIAL TENDRA LA MISMA CANTIDAD DE ENTRADAS QUE LA RED
		capas = []
		cant_entradas_de_la_capa = self.cant_entradas
		for cant_neuronas in estructura:
			capa_neuronal = CapaNeuronal(cant_neuronas, cant_entradas_de_la_capa, self.factiv, self.fcosto)
			capas.append(capa_neuronal)
			#LA PROXIMA CAPA TENDRA TANTAS ENTRADAS COMO NEURONAS EN LA CAPA ACTUAL
			cant_entradas_de_la_capa = cant_neuronas
		return capas
	
	def procesar(self, entradas):
		'''
		PROCESA LAS ENTRADAS A TRAVES DE LA RED. DEVUELVE UN VECTOR DE RESULTADOS (LA LONGITUD DEL VECTOR ES LA MISMA QUE LA CANTIDAD DE NEURONAS DE LA ULTIMA CAPA)
		PARAMETROS:
			ENTRADAS: VECTOR CON LOS PARAMETROS DE ENTRADA, DEBE SER DE LA LONGITUD ESPECIFICADA AL CREAR LA RED
		'''
		for capa in self.capas:
			entradas = capa.procesar(entradas)
		return entradas
		
	def _ultima_capa(self):
		return self.capas[len(self.capas) - 1]
		
	def _generar_deltas(self, entradas, y_esperados):
		self.procesar(entradas)
		return self._ultima_capa().generar_deltas(y_esperados)
	
	def _propagar_deltas(self, deltas, learning_rate):
		for capa in reversed(self.capas):
			deltas = capa.procesar_deltas(deltas, learning_rate)
		return
		
	def entrenar(self, entradas, valores_esperados, learning_rate = 0.01):
		#OBTENGO LOS DELTAS QUE LE VOY A TRANSFERIR A LA ULTIMA CAPA
		deltas = self._generar_deltas(entradas, valores_esperados)
		#TRANSFIERO LOS DELTAS DESDE LA ULTIMA CAPA HASTA LA PRIMERA
		self._propagar_deltas(deltas, learning_rate)
		return
		
	def entrenar_set(self, conjunto_de_entradas, conjunto_de_valores_esperados, epochs = 1000, learning_rate = 0.05):
		for i in range(epochs):
			for entradas, valores_esperados in zip(conjunto_de_entradas, conjunto_de_valores_esperados):
				self.entrenar(entradas, valores_esperados, learning_rate)
		return i

def testRedNeuronal():
	LEARNING_RATE = 10
	EPOCHS = 100000
	F = Sigmoide()
	print('TEST RED NEURONAL')
	rn = RedNeuronal(4, [2,2,4], F)
	rn.entrenar([1,1,1,1],[1,1,1,1])
	#return
	print('PRUEBA COMPUERTA AND')
	rn = RedNeuronal(2, [1], F)
	datos = [[1,1],[1,0],[0,1],[0,0]]
	esperados = [[1],[0],[0],[0]]
	rn.entrenar_set(datos, esperados, EPOCHS, LEARNING_RATE)
	print(rn.procesar([1,1]))
	print(rn.procesar([1,0]))
	print(rn.procesar([0,1]))
	print(rn.procesar([0,0]))
	print('PRUEBA COMPUERTA OR')
	rn = RedNeuronal(2, [1], F)
	datos = [[1,1],[1,0],[0,1],[0,0]]
	esperados = [[1],[1],[1],[0]]
	rn.entrenar_set(datos, esperados, EPOCHS, LEARNING_RATE)
	print(rn.procesar([1,1]))
	print(rn.procesar([1,0]))
	print(rn.procesar([0,1]))
	print(rn.procesar([0,0]))
	print('PRUEBA COMPUERTA XAND')
	rn = RedNeuronal(2, [2,1], F)
	datos = [[1,1],[1,0],[0,1],[0,0]]
	esperados = [[1],[0],[0],[1]]
	rn.entrenar_set(datos, esperados, EPOCHS, LEARNING_RATE)
	print(rn.procesar([1,1]))
	print(rn.procesar([1,0]))
	print(rn.procesar([0,1]))
	print(rn.procesar([0,0]))
	print('PRUEBA COMPUERTA XOR')
	rn = RedNeuronal(2, [2,1], F)
	datos = [[1,1],[1,0],[0,1],[0,0]]
	esperados = [[0],[1],[1],[0]]
	rn.entrenar_set(datos, esperados, EPOCHS, LEARNING_RATE)
	print(rn.procesar([1,1]))
	print(rn.procesar([1,0]))
	print(rn.procesar([0,1]))
	print(rn.procesar([0,0]))
	
testRedNeuronal()
