from Funciones import *
from Test import *
from Neurona import *
import numpy as np

class CapaNeuronal:
	
	def __init__(self, cant_neuronas, cant_entradas, f_activacion, f_costo):
		'''
		CONSTRUCTOR DE LA CAPA NEURONAL
		PARAMETROS:
			CANT_NEURONAS: CANTIDAD DE NEURONAS DE LA CAPA (INT)
			CANT_ENTRADAS: CANTIDAD DE ENTRADAS DE LA CAPA (INT)
			FACTIV: FUNCION DE ACTIVACION DERIVABLE
		'''
		self.cant_entradas = cant_entradas
		self.cant_neuronas = cant_neuronas
		self.neuronas = self._generar_neuronas(f_activacion, f_costo)
		return
	
	def _generar_neuronas(self, f_activacion, f_costo):
		neuronas = []
		for i in range(self.cant_neuronas):
			neuronas.append(Neurona(self.cant_entradas, f_activacion, f_costo))
		return neuronas
			
	def procesar(self, entradas):
		'''
		PROCESA LAS ENTRADAS RECIBIDAS POR LA CAPA NEURONAL. DEVUELVE UN VECTOR DE RESULTADOS (LA LONGITUD DEL VECTOR ES LA MISMA QUE LA CANTIDAD DE NEURONAS DE LA CAPA)
		PARAMETROS:
			ENTRADAS: VECTOR CON LOS PARAMETROS DE ENTRADA, DEBE SER DE LA LONGITUD ESPECIFICADA AL CREAR LA CAPA
		'''
		vector_r = []
		for neurona in self.neuronas:
			vector_r.append(neurona.procesar(entradas))
		return vector_r
	
	def procesar_deltas(self, deltas, learning_rate):
		#print(deltas)
		n = self.len_entradas()
		m = self.len_salidas()
		sumatoria = [0] * n
		#DERIVADA DE Z RESPECTO DE CADA ACTIVACION DE LA CAPA ANTERIOR (ES DECIR LAS ENTRADAS): LOS PESOS DE ESTA CAPA
		#DERIVADA DE Z RESPECTO DE CADA PESO DE ESTA CAPA: LAS ENTRADAS DE ESTA CAPA
		#DERIVADA DE Z RESPECTO DE CADA PARAMETRO DE BIAS: UNO
		for neurona, delta in zip(self.neuronas, deltas):
			#CALCULO LOS NUEVOS BIAS
			#LOS NUEVOS BIAS SE CALCULAN COMO b = b - (dC/db) * LR = b - deltas * (da/dZ) * (dZ/db) * LR = b - deltas * A´(Z) * LR
			neurona.actualizar_bias(delta, learning_rate)
			#CALCULO LOS NUEVOS PESOS
			#LOS NUEVOS PESOS SE CALCULAN COMO W = W - (dC/dW) * LR = W - deltas * (da/dZ) * (dZ/dW) * LR = W - deltas * A´(Z) * entradas * LR
			neurona.actualizar_pesos(delta, learning_rate)
			#CALCULO LOS NUEVOS DELTAS PARA LA CAPA SIGUIENTE
			#LOS NUEVOS DELTAS SE CALCULAN COMO d = d * (dZ/da) = d * W
			aux = neurona.generar_proximos_deltas(delta)
			for i in range(n):
				sumatoria[i] = sumatoria[i] + aux[i]
		return sumatoria
	
	def generar_deltas(self, y_esperados):
		deltas = []
		for neurona, y_esperado in zip(self.neuronas, y_esperados):
			deltas.append(neurona.generar_delta(y_esperado))
		return deltas
		
	def len_entradas(self):
		return self.cant_entradas
		
	def len_salidas(self):
		return self.cant_neuronas

from Funciones import *

def testCapaNeuronal():
	print('TEST CAPA NEURONAL')
	f = Relu()
	capa = CapaNeuronal(2,2,f)
	m = Matriz(2,1,[3,11])
	r = capa.procesar(m)
	Test.test(r.getColumn(0)[0],14)
	Test.test(r.getColumn(0)[1],14)
	
#testCapaNeuronal()
