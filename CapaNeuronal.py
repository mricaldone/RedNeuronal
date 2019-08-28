from Neurona import *
from Test import *

class CapaNeuronal:
	
	def __init__(self, cant_neuronas, cant_entradas, factiv):
		'''
		CONSTRUCTOR DE LA CAPA NEURONAL
		PARAMETROS:
			CANT_NEURONAS: CANTIDAD DE NEURONAS DE LA CAPA (INT)
			CANT_ENTRADAS: CANTIDAD DE ENTRADAS DE LA CAPA (INT)
			FACTIV: FUNCION DE ACTIVACION DERIVABLE
		'''
		self.cant_entradas = cant_entradas
		self.factiv = factiv
		self.cant_neuronas = cant_neuronas
		self.neuronas = []
		for i in range(self.cant_neuronas):
			self.neuronas.append(Neurona(self.cant_entradas, self.factiv))
			
	def procesar(self, entradas):
		'''
		PROCESA LAS ENTRADAS RECIBIDAS POR LA CAPA NEURONAL. DEVUELVE UN VECTOR DE RESULTADOS (LA LONGITUD DEL VECTOR ES LA MISMA QUE LA CANTIDAD DE NEURONAS DE LA CAPA)
		PARAMETROS:
			ENTRADAS: VECTOR CON LOS PARAMETROS DE ENTRADA, DEBE SER DE LA LONGITUD ESPECIFICADA AL CREAR LA CAPA
		'''
		#if len(entradas) != len(self.cant_entradas):
		#	raise CantidadDeEntradasCapaNeuronalError
		resultados = []
		for neurona in self.neuronas:
			resultados.append(neurona.procesar(entradas))
		return resultados
		
	def __len__(self):
		return self.cant_neuronas

from Funciones import *

def testCapaNeuronal():
	print('TEST CAPA NEURONAL')
	f = Relu()
	capa = CapaNeuronal(2,2,f)
	x1 = 3
	x2 = 11
	r = capa.procesar([x1,x2])
	Test.test(r[0],14)
	
testCapaNeuronal()
