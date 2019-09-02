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
	
	def _propagar_deltas(self, deltas, learning_rate):
		for i,capa in enumerate(reversed(self.capas)):
			n = capa.len_entradas()
			m = capa.len_salidas()
			#DERIVADA DE Z RESPECTO DE CADA ACTIVACION DE LA CAPA ANTERIOR (ES DECIR LAS ENTRADAS): LOS PESOS DE ESTA CAPA
			matriz_w = capa.obtener_pesos()
			#DERIVADA DE Z RESPECTO DE CADA PESO DE ESTA CAPA: LAS ENTRADAS DE ESTA CAPA
			vector_a = capa.obtener_entradas()
			#DERIVADA DE Z RESPECTO DE CADA PARAMETRO DE BIAS: UNO
			vector_b = capa.obtener_bias()
			
			#print(deltas)
			#CALCULO LOS NUEVOS BIAS
			#LOS NUEVOS BIAS SE CALCULAN COMO b = b - (dC/db) * LR = b - (deltas * (dZ/db)) * LR = b - deltas * LR 
			vector_r = Matriz(m, 1, [-learning_rate])
			nuevo_vector_b = vector_b.sum_mat(deltas.mul_directa_mat(vector_r))
			capa.actualizar_bias(nuevo_vector_b)
			#print(nuevo_vector_b)
			
			#CALCULO LOS NUEVOS PESOS
			#LOS NUEVOS PESOS SE CALCULAN COMO W = W - (dC/dW) * LR = W - deltas * (dZ/dW) * LR = W - deltas * entradas * lr
			matriz_d = deltas.expandirColumnas(n)
			matriz_a = vector_a.expandirColumnas(m).transpuesta()
			matriz_r = Matriz(m, n, [-learning_rate])
			nueva_matriz_w = matriz_w.sum_mat(matriz_d.mul_directa_mat(matriz_a.mul_directa_mat(matriz_r)))
			capa.actualizar_pesos(nueva_matriz_w)
			#print(matriz_w)
			#print(nueva_matriz_w)
			
			#CALCULO LOS NUEVOS DELTAS PARA LA CAPA SIGUIENTE
			#LOS NUEVOS DELTAS SE CALCULAN COMO d = d * (dZ/da) = d * W
			#print(matriz_d)
			nueva_matriz_d = matriz_d.mul_directa_mat(matriz_w)
			#print(nueva_matriz_d)
			deltas = nueva_matriz_d.transpuesta().vector_medio()
			#print(deltas)
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
	print('TEST RED NEURONAL')
	LEARNING_RATE = 0.05
	EPOCHS = 100
	F = Sigmoide()
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
	rn.entrenar_set(datos, esperados, EPOCHS, 0.5)
	print(rn.procesar([1,1]))
	print(rn.procesar([1,0]))
	print(rn.procesar([0,1]))
	print(rn.procesar([0,0]))
	
testRedNeuronal()
