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
	
	#def _producto_interno(self, vect_a, vect_b):
	#	r = 0
	#	for a,b in zip(vect_a,vect_b):
	#		r = r + a * b
	#	return r
	
	#def _mul_mat_vec(self, mat, vect):
	#	r = []
	#	for v in mat:
	#		r.append(self._producto_interno(v,vect))
	#	return r
		
	#def _mul_directa_vectores(self, va, vb):
	#	print(va,vb)
	#	r = []
	#	for a,b in zip(va,vb):
	#		r.append(a * b)
	#	return r
		
	#def _sum_directa_vectores(self, va, vb):
	#	r = []
	#	for a,b in zip(va,vb):
	#		r.append(a + b)
	#	return r
	
	#def _mul_directa_vec_escalar(self, vector, escalar):
	#	r = []
	#	for v in vector:
	#		r.append(v * escalar)
	#	return r
	
	#def _mul_directa_mat_vec(self, mat, vec):
	#	r = []
	#	for vm,v in zip(mat,vec):
	#		r.append(self._mul_directa_vec_escalar(vm,v))
	#	return r
		
	#def _evaluar_vector(self, funcion, vector):
	#	y = []
	#	for x in vector:
	#		y.append(funcion(x))
	#	return y
			
	def procesar(self, entradas):
		'''
		PROCESA LAS ENTRADAS RECIBIDAS POR LA CAPA NEURONAL. DEVUELVE UN VECTOR DE RESULTADOS (LA LONGITUD DEL VECTOR ES LA MISMA QUE LA CANTIDAD DE NEURONAS DE LA CAPA)
		PARAMETROS:
			ENTRADAS: VECTOR CON LOS PARAMETROS DE ENTRADA, DEBE SER DE LA LONGITUD ESPECIFICADA AL CREAR LA CAPA
		'''
		self.vector_x = entradas
		#print(self.matriz_w)
		#print(self.vector_x)
		#print(self.matriz_w.mul_mat(self.vector_x))
		#print(self.vector_b)
		self.vector_z = self.matriz_w.mul_mat(self.vector_x).sum_mat(self.vector_b)
		self.vector_y = self.vector_z.evaluar(self.factiv.evaluar)
		return self.vector_y
	
	def asignar_deltas(self, deltas):
		self.vector_d = deltas
		
	def calcular_deltas(self, deltas_capa_anterior, matriz_de_pesos_capa_anterior):
		#self.deltas = deltas
		w = matriz_de_pesos_capa_anterior.transpuesta()
		#print(w)
		d = deltas_capa_anterior
		#print(d)
		da = self.vector_z.evaluar(self.factiv.derivada)
		#print(da)
		w_da = w.mul_mat(da)
		#print(w_da)
		return w_da.mul_directa_mat(d)
		
		#nuevos_deltas = []
		#for i,neurona in enumearate(self.neuronas):
			#pesos de la siguiente capa asociados a esta neurona
		#	pesos_asociados_a_esta_neurona = []
		#	for j in len(W):
		#		pesos_asociados_a_esta_neurona.append(W[j][i]) #En realidad esto es como obtener una fila de la matriz transpuesta de W
		#	nuevo_delta = neurona.calcular_delta(deltas, pesos_asociados_a_esta_neurona)
		#	nuevos_deltas.append(nuevo_delta)
		#return nuevos_deltas
		
	def actualizar_pesos(self, learning_rate):
		#dim_m = self.matriz_w.dim_m()
		lr = Matriz(self.vector_d.dim_n(), self.vector_d.dim_m(), [-learning_rate])
		deltas = self.vector_d.mul_directa_mat(lr)
		aux = self.vector_x.mul_mat(deltas.transpuesta())
		self.matriz_w = self.matriz_w.sum_mat(aux.transpuesta())
		#ACTUALIZAR BIAS
		self.vector_b = self.vector_b.sum_mat(deltas)
		#print(self.vector_b)
		return
	
	def obtener_z(self):
		return self.vector_z
	
	def obtener_matriz_de_pesos(self):
		return self.matriz_w
		
	#def __len__(self):
	#	return self.cant_neuronas

from Funciones import *

def testCapaNeuronal():
	print('TEST CAPA NEURONAL')
	f = Relu()
	capa = CapaNeuronal(2,2,f)
	m = Matriz(2,1,[3,11])
	r = capa.procesar(m)
	Test.test(r.getColumn(0)[0],14)
	Test.test(r.getColumn(0)[1],14)
	
testCapaNeuronal()
