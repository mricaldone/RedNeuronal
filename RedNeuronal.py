from CapaNeuronal import *
from Funciones import *

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
			capa_neuronal = CapaNeuronal(cant_neuronas, cant_entradas_de_la_capa, self.factiv)
			capas.append(capa_neuronal)
			#LA PROXIMA CAPA TENDRA TANTAS ENTRADAS COMO NEURONAS EN LA CAPA ACTUAL
			cant_entradas_de_la_capa = cant_neuronas
		return capas
		
	def _ultima_capa(self):
		return self.capas[len(self.capas) - 1]
	
	def _aplicar_derivada_de_coste(self, valores_obtenidos, valores_esperados):
		vector_dcoste = []
		for valor_obtenido, valor_esperado in zip(valores_obtenidos, valores_esperados):
			vector_dcoste.append(self.fcosto.derivada(valor_obtenido, valor_esperado))
		return vector_dcoste
	
	def _entrenar_capas(self, deltas, learning_rate):
		for capa in reversed(self.capas):
			deltas = capa.entrenar(deltas, learning_rate)
	
	def _calcular_tolerancia(self, valores_esperados, valores_obtenidos):
		tolerancias = []
		for valor_obtenido, valor_esperado in zip(valores_obtenidos, valores_esperados):
			tolerancias.append(abs(valor_obtenido - valor_esperado))
		return max(tolerancias)
	
	def procesar(self, entradas):
		'''
		PROCESA LAS ENTRADAS A TRAVES DE LA RED. DEVUELVE UN VECTOR DE RESULTADOS (LA LONGITUD DEL VECTOR ES LA MISMA QUE LA CANTIDAD DE NEURONAS DE LA ULTIMA CAPA)
		PARAMETROS:
			ENTRADAS: VECTOR CON LOS PARAMETROS DE ENTRADA, DEBE SER DE LA LONGITUD ESPECIFICADA AL CREAR LA RED
		'''
		for capa in self.capas:
			entradas = capa.procesar(entradas)
		return entradas
	
	def entrenar(self, entradas, valores_esperados, learning_rate = 0.01):
		#PROCESO TODAS LAS ENTRADAS
		valores_obtenidos = self.procesar(entradas)
		#print('R',valores_obtenidos)
		#APLICO LA DERIVADA DE LA FUNCION DE COSTE
		deltas = self._aplicar_derivada_de_coste(valores_obtenidos, valores_esperados)
		#ENTRENO LAS CAPAS DE ATRAS HACIA ADELANTE
		self._entrenar_capas(deltas, learning_rate)
		return self._calcular_tolerancia(valores_esperados, valores_obtenidos)
		
	def entrenar_set(self, conjunto_de_entradas, conjunto_de_valores_esperados, epochs = 10000, learning_rate = 0.5, tolerancia = 0):
		for i in range(epochs):
			stop = True
			for entradas, valores_esperados in zip(conjunto_de_entradas, conjunto_de_valores_esperados):
				if self.entrenar(entradas, valores_esperados, learning_rate) > tolerancia:
					stop = False
			if stop:
				break
		return i + 1
