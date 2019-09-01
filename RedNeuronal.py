from CapaNeuronal import *
from Funciones import *
from Test import *
from Matriz import *

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
		self.fcosto = ErrorCuadraticoMedio()
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
		return
	
	def procesar(self, entradas):
		'''
		PROCESA LAS ENTRADAS A TRAVES DE LA RED. DEVUELVE UN VECTOR DE RESULTADOS (LA LONGITUD DEL VECTOR ES LA MISMA QUE LA CANTIDAD DE NEURONAS DE LA ULTIMA CAPA)
		PARAMETROS:
			ENTRADAS: VECTOR CON LOS PARAMETROS DE ENTRADA, DEBE SER DE LA LONGITUD ESPECIFICADA AL CREAR LA RED
		'''
		#if len(entradas) != len(self.cant_entradas):
		#	raise CantidadDeEntradasRedNeuronalError
		r = Matriz(len(entradas),1,entradas)
		for capa in self.capas:
			r = capa.procesar(r)
		return r.getColumn(0)
		
	def _ultima_capa(self):
		return self.capas[len(self.capas) - 1]
		
	def _generar_deltas(self, entradas, y_esperados):
		y_obtenidos = self.procesar(entradas)
		z_obtenidos = self._ultima_capa().obtener_z().getColumn(0)
		deltas = []
		for valor_obtenido, valor_esperado, z_obtenido in zip(y_obtenidos, y_esperados, z_obtenidos):
			#DERIVADA DEL COSTE
			dc = self.fcosto.derivada(valor_esperado, valor_obtenido)
			#DERIVADA DE LA ACTIVACION
			da = self.factiv.derivada(z_obtenido)
			deltas.append(da * dc)
		return Matriz(len(deltas), 1, deltas)
		
	def _propagar_deltas(self, deltas):
		capa_anterior = None
		for capa in reversed(self.capas):
			if capa == self._ultima_capa():
				capa.asignar_deltas(deltas)
			else:
				matriz_de_pesos = capa_anterior.obtener_matriz_de_pesos()	
				deltas = capa.calcular_deltas(deltas, matriz_de_pesos)
				capa.asignar_deltas(deltas)
			capa_anterior = capa
		return
		
	def _actualizar_pesos(self, learning_rate):
		for capa in reversed(self.capas):
			capa.actualizar_pesos(learning_rate)
		return
		
	def entrenar(self, entradas, valores_esperados, learning_rate = 0.01):
		#OBTENGO LOS DELTAS QUE LE VOY A TRANSFERIR A LA ULTIMA CAPA
		deltas = self._generar_deltas(entradas, valores_esperados)
		#TRANSFIERO LOS DELTAS DESDE LA ULTIMA CAPA HASTA LA PRIMERA
		self._propagar_deltas(deltas)
		#ACTUALIZO PESOS DESDE LA ULTIMA CAPA HASTA LA PRIMERA
		self._actualizar_pesos(learning_rate)
		return
		
	def entrenar_set(self, conjunto_de_entradas, conjunto_de_valores_esperados, epochs = 1000, learning_rate = 0.01):
		for i in range(epochs):
			for entradas, valores_esperados in zip(conjunto_de_entradas, conjunto_de_valores_esperados):
				self.entrenar(entradas, valores_esperados, learning_rate)
		return i

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
	print('TEST RED NEURONAL 2')
	f = Sigmoide()
	rn = RedNeuronal(4, [2,2,4], f)
	rn.entrenar([1,1,1,1],[1,1,1,1])
	print('PRUEBA COMPUERTA AND')
	rn = RedNeuronal(2, [1], f)
	datos = [[1,1],[1,0],[0,1],[0,0]]
	esperados = [[1],[0],[0],[0]]
	rn.entrenar_set(datos, esperados, 10000, 0.05)
	#for i in range(1000):
	#	rn.entrenar([1,1],[1])
	#	rn.entrenar([1,0],[0])
	#	rn.entrenar([0,1],[0])
	#	rn.entrenar([0,0],[0])
	#	print(i)
	#	print(rn.procesar([1,1]))
	#	print(rn.procesar([1,0]))
	#	print(rn.procesar([0,1]))
	#	print(rn.procesar([0,0]))
	#	input()
	print(rn.procesar([1,1]))
	print(rn.procesar([1,0]))
	print(rn.procesar([0,1]))
	print(rn.procesar([0,0]))
	print('PRUEBA COMPUERTA OR')
	rn = RedNeuronal(2, [1], f)
	for i in range(10000):
		rn.entrenar([1,1],[1],0.05)
		rn.entrenar([1,0],[1],0.05)
		rn.entrenar([0,1],[1],0.05)
		rn.entrenar([0,0],[0],0.05)
	print(rn.procesar([1,1]))
	print(rn.procesar([1,0]))
	print(rn.procesar([0,1]))
	print(rn.procesar([0,0]))
	return
	print('PRUEBA COMPUERTA XAND')
	rn = RedNeuronal(2, [2,1], f)
	for i in range(1):
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
	for i in range(1):
		rn.entrenar([1,1],[0])
		rn.entrenar([1,0],[1])
		rn.entrenar([0,1],[1])
		rn.entrenar([0,0],[0])
	print(rn.procesar([1,1]))
	print(rn.procesar([1,0]))
	print(rn.procesar([0,1]))
	print(rn.procesar([0,0]))
	
testRedNeuronal()
