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
	
	#def generar_deltas_de_capa_final(self, y_esperados):
	#	deltas = []
	#	for neurona, y_esperado in zip(self.neuronas, y_esperados):
	#		deltas.append(neurona.generar_delta_de_capa_final(y_esperado))
	#	return deltas
	
	def entrenar_capa_intermedia(self, deltas, learning_rate):
		sumatoria = [0] * self.cant_neuronas
		for delta in deltas:
			for i,d in enumerate(delta):
				sumatoria[i] = sumatoria[i] + d
		deltas = sumatoria
		nuevos_deltas = []
		for neurona, delta in zip(self.neuronas, deltas):
			#CALCULO LOS NUEVOS DELTAS PARA LA CAPA SIGUIENTE
			#LOS NUEVOS DELTAS SE CALCULAN COMO d = d * (da/dz) = d * a´(z)
			delta = neurona.generar_delta_de_capa_intermedia(delta)
			#CALCULO LOS NUEVOS PESOS
			#LOS NUEVOS PESOS SE CALCULAN COMO W = W - (dC/dw) * LR = W - deltas * (dz/dw) * LR = W - deltas * entradas * LR
			neurona.actualizar_pesos(delta, learning_rate)
			#CALCULO LOS NUEVOS BIAS
			#LOS NUEVOS BIAS SE CALCULAN COMO b = b - (dC/db) * LR = b - deltas * (dz/db) * LR = b - deltas * 1 * LR
			neurona.actualizar_bias(delta, learning_rate)
			#CALCULO LOS DELTAS PARA LA CAPA SIGUIENTE
			#LOS DELTAS SIGUIENTES SE CALCULAN COMO d = d * (dz/dx) = d * w
			nuevos_deltas.append(neurona.generar_deltas_capa_siguiente(delta))
		return nuevos_deltas
		
	def entrenar_capa_final(self, y_esperados, learning_rate):
		deltas = []
		for neurona, y_esperado in zip(self.neuronas, y_esperados):
			#CALCULO LOS DELTAS FINALES PARA LA CAPA SIGUIENTE
			#LOS DELTAS FINALES SE CALCULAN COMO d = (dC/da) (da/dz) = C´(a) * a´(z)
			delta = neurona.generar_delta_de_capa_final(y_esperado)
			#CALCULO LOS NUEVOS PESOS
			#LOS NUEVOS PESOS SE CALCULAN COMO W = W - (dC/dW) * LR = W - (dC/da) * (da/dz) * (dz/dw) * LR = W - deltas * entradas * LR
			neurona.actualizar_pesos(delta, learning_rate)
			#CALCULO LOS NUEVOS BIAS
			#LOS NUEVOS BIAS SE CALCULAN COMO b = b - (dC/db) * LR = b - (dC/da) * (da/dz) * (dz/db) * LR = b - deltas * 1 * LR
			neurona.actualizar_bias(delta, learning_rate)
			#CALCULO LOS DELTAS PARA LA CAPA SIGUIENTE
			#LOS DELTAS SIGUIENTES SE CALCULAN COMO d = d * (dz/dx) = d * w
			deltas.append(neurona.generar_deltas_capa_siguiente(delta))
		return deltas
