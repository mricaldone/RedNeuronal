from RedNeuronal import *

def main():
	LEARNING_RATE = 10
	EPOCHS = 1000
	TOLERANCIA = 0.1
	F = Sigmoide()
	#F = Relu()
	print('TEST RED NEURONAL')
	rn = RedNeuronal(2, [1], F)
	rn.entrenar([1,1],[0])
	rn = RedNeuronal(4, [2,2,4], F)
	rn.entrenar([1,1,1,1],[1,1,1,1])
	rn = RedNeuronal(4, [4], F)
	rn.entrenar([1,1,1,1],[1,1,1,1])
	rn = RedNeuronal(1, [1], F)
	rn.entrenar([1,1],[1])
	print('PRUEBA COMPUERTA YES')
	rn = RedNeuronal(1, [1], F)
	datos = [[1],[0]]
	esperados = [[1],[0]]
	print('EPOCHS', rn.entrenar_set(datos, esperados, EPOCHS, LEARNING_RATE, TOLERANCIA))
	print(rn.procesar([1]))
	print(rn.procesar([0]))
	print('PRUEBA COMPUERTA NOT')
	rn = RedNeuronal(1, [1], F)
	datos = [[1],[0]]
	esperados = [[0],[1]]
	print('EPOCHS', rn.entrenar_set(datos, esperados, EPOCHS, LEARNING_RATE, TOLERANCIA))
	print(rn.procesar([1]))
	print(rn.procesar([0]))
	print('PRUEBA COMPUERTA AND')
	rn = RedNeuronal(2, [1], F)
	datos = [[1,1],[1,0],[0,1],[0,0]]
	esperados = [[1],[0],[0],[0]]
	print('EPOCHS', rn.entrenar_set(datos, esperados, EPOCHS, LEARNING_RATE, TOLERANCIA))
	print(rn.procesar([1,1]))
	print(rn.procesar([1,0]))
	print(rn.procesar([0,1]))
	print(rn.procesar([0,0]))
	print('PRUEBA COMPUERTA OR')
	rn = RedNeuronal(2, [1], F)
	datos = [[1,1],[1,0],[0,1],[0,0]]
	esperados = [[1],[1],[1],[0]]
	print('EPOCHS', rn.entrenar_set(datos, esperados, EPOCHS, LEARNING_RATE, TOLERANCIA))
	print(rn.procesar([1,1]))
	print(rn.procesar([1,0]))
	print(rn.procesar([0,1]))
	print(rn.procesar([0,0]))
	print('PRUEBA COMPUERTA XAND')
	rn = RedNeuronal(2, [2,1], F)
	datos = [[1,1],[1,0],[0,1],[0,0]]
	esperados = [[1],[0],[0],[1]]
	print('EPOCHS', rn.entrenar_set(datos, esperados, EPOCHS, LEARNING_RATE, TOLERANCIA))
	print(rn.procesar([1,1]))
	print(rn.procesar([1,0]))
	print(rn.procesar([0,1]))
	print(rn.procesar([0,0]))
	print('PRUEBA COMPUERTA XOR')
	rn = RedNeuronal(2, [2,1], F)
	datos = [[1,1],[1,0],[0,1],[0,0]]
	esperados = [[0],[1],[1],[0]]
	print('EPOCHS', rn.entrenar_set(datos, esperados, EPOCHS, LEARNING_RATE, TOLERANCIA))
	print(rn.procesar([1,1]))
	print(rn.procesar([1,0]))
	print(rn.procesar([0,1]))
	print(rn.procesar([0,0]))
	
main()
