from Neurona import *

class CapaNeuronal:

	def __init__(self, cant_neuronas, cant_entradas, f_activacion):
		'''
		CONSTRUCTOR DE LA CAPA NEURONAL
		PARAMETROS:
			CANT_NEURONAS: CANTIDAD DE NEURONAS DE LA CAPA (INT)
			CANT_ENTRADAS: CANTIDAD DE ENTRADAS DE LA CAPA (INT)
			FACTIV: FUNCION DE ACTIVACION DERIVABLE
		'''
		self.cant_entradas = cant_entradas
		self.cant_neuronas = cant_neuronas
		self.neuronas = self._generar_neuronas(f_activacion)
		return

	def _generar_neuronas(self, f_activacion):
		neuronas = []
		for i in range(self.cant_neuronas):
			neuronas.append(Neurona(self.cant_entradas, f_activacion))
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
	
	def _sumar_columnas(self, matriz):
		sumatoria = [0] * self.cant_entradas
		for fila in matriz:
			for i, valor in enumerate(fila):
				sumatoria[i] = sumatoria[i] + valor
		return sumatoria
	
	def entrenar_capa(self, deltas, learning_rate):
		print('CAPA')
		nuevos_deltas = []
		for neurona, delta in zip(self.neuronas, deltas):
			print('NEURONA')
			#CALCULO LOS NUEVOS DELTAS PARA LA CAPA SIGUIENTE
			#LOS NUEVOS DELTAS SE CALCULAN COMO d = d * (da/dz) = d * a´(z)
			delta = neurona.generar_delta(delta)
			#CALCULO LOS DELTAS PARA LA CAPA SIGUIENTE
			#LOS DELTAS SIGUIENTES SE CALCULAN COMO d = d * (dz/dx) = d * w
			nuevos_deltas.append(neurona.generar_deltas_capa_siguiente(delta))
			#CALCULO LOS NUEVOS PESOS
			#LOS NUEVOS PESOS SE CALCULAN COMO W = W - (dC/dw) * LR = W - deltas * (dz/dw) * LR = W - deltas * entradas * LR
			neurona.actualizar_pesos(delta, learning_rate)
			#CALCULO LOS NUEVOS BIAS
			#LOS NUEVOS BIAS SE CALCULAN COMO b = b - (dC/db) * LR = b - deltas * (dz/db) * LR = b - deltas * 1 * LR
			neurona.actualizar_bias(delta, learning_rate)
		return self._sumar_columnas(nuevos_deltas)
