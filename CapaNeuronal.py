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
	
	def entrenar(self, deltas, learning_rate):
		'''
		ENTRENA A LAS NEURONAS DE LA CAPA SEGUN CIERTOS DELTAS DADOS Y LEARNING RATE
		PARAMETROS:
			DELTAS: UNA LISTA CON LOS RESULTADOS DE LAS DERIVADAS PARCIALES DE LAS CAPAS SIGUIENTES DE LA RED. DEBE HABER TANTAS COMO NEURONAS EN ESTA CAPA. (LIST(DOUBLES))
			LEARNING_RATE: VELOCIDAD DE APRENDIZAJE. UN LR ALTO IMPLICA UNA MAYOR VELOCIDAD PARA ENCONTRAR EL RESULTADO, SIN EMBARGO PUEDE NO LLEGAR AL RESULTADO OPTIMO. (DOUBLE)
		'''
		sumatoria = [0] * self.cant_entradas
		for neurona, delta in zip(self.neuronas, deltas):
			deltas_neurona = neurona.entrenar(delta, learning_rate)
			for i, valor in enumerate(deltas_neurona):
				sumatoria[i] = sumatoria[i] + valor
		return sumatoria
