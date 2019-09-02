from Funciones import *
from Test import *
from Neurona import *
import numpy as np

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
		
		#self.factiv = factiv
		#self.matriz_w = self._inicializar_matriz_de_pesos()
		#self.vector_b = self._inicializar_vector_de_bias()
		#self.vector_x = None #REPRESENTA LAS ENTRADAS
		#self.vector_z = None #REPRESENTA LA SALIDA DEL PERCEPTRON
		#self.vector_y = None #REPRESENTA LA SALIDA DE LA NEURONA
		#self.vector_d = None #REPRESENTA LOS DELTAS DE CADA NEURONA
		return
	
	def _generar_neuronas(self, f_activacion):
		neuronas = []
		for i in range(self.cant_neuronas):
			neuronas.append(Neurona(self.cant_entradas, f_activacion))
		return neuronas
	
	#def _inicializar_matriz_de_pesos(self):
	#	#return Matriz(self.cant_neuronas, self.cant_entradas, [1])
	#	dim = self.cant_neuronas * self.cant_entradas
	#	return Matriz(self.cant_neuronas, self.cant_entradas, np.random.rand(dim))
		
	#def _inicializar_vector_de_bias(self):
	#	#return Matriz(self.cant_neuronas, 1, [0])
	#	return Matriz(self.cant_neuronas, 1, np.random.rand(self.cant_neuronas))
			
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
		#self.vector_x = entradas
		#self.vector_z = self.matriz_w.mul_mat(self.vector_x).sum_mat(self.vector_b)
		#self.vector_y = self.vector_z.evaluar(self.factiv.evaluar)
		#return self.vector_y
	
	def _obtener_pesos(self):
		matriz_w = []
		for neurona in self.neuronas:
			matriz_w.append(neurona.obtener_pesos())
		return matriz_w
	
	def _obtener_entradas(self):
		matriz_a = []
		for neurona in self.neuronas:
			matriz_a.append(neurona.obtener_entradas())
		return matriz_a
	
	def _obtener_bias(self):
		vector_b = []
		for neurona in self.neuronas:
			vector_b.append(neurona.obtener_bias())
		return vector_b
	
	def obtener_vector_z(self):
		vector_z = []
		for neurona in self.neuronas:
			vector_z.append(neurona.obtener_z())
		return vector_z
	
	def _obtener_vector_da(self):
		vector_da = []
		for neurona in self.neuronas:
			vector_da.append(neurona.obtener_derivada_activacion())
		return vector_da
	
	def _actualizar_bias(self, vector_b):
		for neurona, b in zip(self.neuronas, vector_b):
			neurona.actualizar_bias(b)
			
	def _actualizar_pesos(self, matriz_w):
		for neurona, w in zip(self.neuronas, matriz_w):
			neurona.actualizar_pesos(w)
	
	def procesar_deltas(self, deltas, learning_rate):
		n = self.len_entradas()
		m = self.len_salidas()
		#DERIVADA DE Z RESPECTO DE CADA ACTIVACION DE LA CAPA ANTERIOR (ES DECIR LAS ENTRADAS): LOS PESOS DE ESTA CAPA
		matriz_w = self._obtener_pesos()
		#DERIVADA DE Z RESPECTO DE CADA PESO DE ESTA CAPA: LAS ENTRADAS DE ESTA CAPA
		matriz_a = self._obtener_entradas()
		#DERIVADA DE Z RESPECTO DE CADA PARAMETRO DE BIAS: UNO
		vector_b = self._obtener_bias()
		
		vector_z = self.obtener_vector_z()
		vector_da = self._obtener_vector_da()
		
		#print(deltas)
		#CALCULO LOS NUEVOS BIAS
		#LOS NUEVOS BIAS SE CALCULAN COMO b = b - (dC/db) * LR = b - deltas * (da/dZ) * (dZ/db) * LR = b - deltas * A´(Z) * LR
		nuevo_vector_b = [] 
		for b, d, da in zip(vector_b, deltas, vector_da):
			nuevo_vector_b.append(b - d * da * learning_rate)
		self._actualizar_bias(nuevo_vector_b)
		#print(vector_b)
		#print(nuevo_vector_b)
		
		#CALCULO LOS NUEVOS PESOS
		#LOS NUEVOS PESOS SE CALCULAN COMO W = W - (dC/dW) * LR = W - deltas * (da/dZ) * (dZ/dW) * LR = W - deltas * A´(Z) * entradas * LR
		nueva_matriz_w = []
		for fila_w, d, da, fila_a in zip(matriz_w, deltas, vector_da, matriz_a):
			nuevos_pesos = []
			for w, x in zip(fila_w, fila_a):
				nuevos_pesos.append(w - d * da * x * learning_rate)
			nueva_matriz_w.append(nuevos_pesos)
		self._actualizar_pesos(nueva_matriz_w)		
		#print(matriz_w)
		#print(nueva_matriz_w)
		
		#CALCULO LOS NUEVOS DELTAS PARA LA CAPA SIGUIENTE
		#LOS NUEVOS DELTAS SE CALCULAN COMO d = d * (dZ/da) = d * W
		nueva_matriz_d = []
		for fila_w, d in zip(matriz_w, deltas):
			nuevos_deltas = []
			for w in fila_w:
				nuevos_deltas.append(d * w)
			nueva_matriz_d.append(nuevos_deltas)
		nuevo_vector_d = [0] * n
		for fila_d in nueva_matriz_d:
			for i,d in enumerate(fila_d):
				nuevo_vector_d[i] = nuevo_vector_d[i] + d
				if i == m - 1:
					nuevo_vector_d[i] = nuevo_vector_d[i] / m
		#print(nueva_matriz_d)
		#print(nuevo_vector_d)
		return nuevo_vector_d
		#print(deltas)
	
	#def obtener_z(self):
	#	return self.vector_z
	
	#def obtener_pesos(self):
	#	return self.matriz_w
		
	#def obtener_entradas(self):
	#	return self.vector_x
	
	#def obtener_bias(self):
	#	return self.vector_b
		
	def len_entradas(self):
		return self.cant_entradas
		
	def len_salidas(self):
		return self.cant_neuronas
		
	#def actualizar_bias(self, vector_b):
	#	self.vector_b = vector_b
	
	#def actualizar_pesos(self, matriz_w):
	#	self.matriz_w = matriz_w

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
