from Funciones import *
from Test import *
from Matriz import *
import numpy as np

class CapaNeuronal:
	
	def __init__(self, cant_neuronas, cant_entradas, factiv):
		'''
		CONSTRUCTOR DE LA CAPA NEURONAL
		PARAMETROS:
			CANT_NEURONAS: CANTIDAD DE NEURONAS DE LA CAPA (INT)
			CANT_ENTRADAS: CANTIDAD DE ENTRADAS DE LA CAPA (INT)
			FACTIV: FUNCION DE ACTIVACION DERIVABLE
		'''
		self.factiv = factiv
		self.cant_entradas = cant_entradas
		self.cant_neuronas = cant_neuronas
		self.matriz_w = self._inicializar_matriz_de_pesos()
		self.vector_b = self._inicializar_vector_de_bias()
		self.vector_x = None #REPRESENTA LAS ENTRADAS
		self.vector_z = None #REPRESENTA LA SALIDA DEL PERCEPTRON
		self.vector_y = None #REPRESENTA LA SALIDA DE LA NEURONA
		self.vector_d = None #REPRESENTA LOS DELTAS DE CADA NEURONA
		return
			
	def _inicializar_matriz_de_pesos(self):
		#return Matriz(self.cant_neuronas, self.cant_entradas, [1])
		dim = self.cant_neuronas * self.cant_entradas
		return Matriz(self.cant_neuronas, self.cant_entradas, np.random.rand(dim))
		
	def _inicializar_vector_de_bias(self):
		#return Matriz(self.cant_neuronas, 1, [0])
		return Matriz(self.cant_neuronas, 1, np.random.rand(self.cant_neuronas))
			
	def procesar(self, entradas):
		'''
		PROCESA LAS ENTRADAS RECIBIDAS POR LA CAPA NEURONAL. DEVUELVE UN VECTOR DE RESULTADOS (LA LONGITUD DEL VECTOR ES LA MISMA QUE LA CANTIDAD DE NEURONAS DE LA CAPA)
		PARAMETROS:
			ENTRADAS: VECTOR CON LOS PARAMETROS DE ENTRADA, DEBE SER DE LA LONGITUD ESPECIFICADA AL CREAR LA CAPA
		'''
		self.vector_x = entradas
		self.vector_z = self.matriz_w.mul_mat(self.vector_x).sum_mat(self.vector_b)
		self.vector_y = self.vector_z.evaluar(self.factiv.evaluar)
		return self.vector_y
	
	def obtener_z(self):
		return self.vector_z
	
	def obtener_pesos(self):
		return self.matriz_w
		
	def obtener_entradas(self):
		return self.vector_x
	
	def obtener_bias(self):
		return self.vector_b
		
	def len_entradas(self):
		return self.cant_entradas
		
	def len_salidas(self):
		return self.cant_neuronas
		
	def actualizar_bias(self, vector_b):
		self.vector_b = vector_b
	
	def actualizar_pesos(self, matriz_w):
		self.matriz_w = matriz_w

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
