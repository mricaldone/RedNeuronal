from Neurona import *

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
	
	def entrenar(self, deltas, learning_rate):
		#print(deltas)
		sumatoria = [0] * self.cant_entradas
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
			nuevos_deltas = neurona.generar_deltas_de_capa_intermedia(delta)
			for i, nuevo_delta in enumerate(nuevos_deltas):
				sumatoria[i] = sumatoria[i] + nuevo_delta
		return sumatoria
	
	def generar_deltas_de_capa_final(self, y_esperados):
		deltas = []
		for neurona, y_esperado in zip(self.neuronas, y_esperados):
			deltas.append(neurona.generar_delta_de_capa_final(y_esperado))
		return deltas
