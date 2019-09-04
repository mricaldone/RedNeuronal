from RedNeuronal import *
from mnist import MNIST
import random

def test():
	LEARNING_RATE = 10
	EPOCHS = 1000
	TOLERANCIA = 0.5
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

def activar_entradas(entradas, factiv):
	r = []
	for entrada in entradas:
		ne = []
		for valor in entrada:
			ne.append(valor/10)
		r.append(ne)
	return r
	
def mnist_test():
	LEARNING_RATE = 0.1
	EPOCHS = 1000
	TOLERANCIA = 0.1
	F = Sigmoide()
	
	mndata = MNIST('samples')

	images, labels = mndata.load_training()
	labels = [labels]
	labels = activar_entradas(labels, F)
	
	rn = RedNeuronal(784, [784,8,1], F)
	print('EPOCHS', rn.entrenar_set(images, labels, EPOCHS, LEARNING_RATE, TOLERANCIA))
	
	images, labels = mndata.load_testing()
	
	while True:
		index = random.randrange(0, len(images))
		r = rn.procesar(images[index])
		print(mndata.display(images[index]))
		print("Esperado", labels[index])
		print("Resultado:", r[0] * 10)
		input()
	
def pruebas_neuronales():
	LEARNING_RATE = 1
	EPOCHS = 1000
	TOLERANCIA = 0.1
	F = Sigmoide()
	rn = RedNeuronal(2, [2,1], F)
	e = rn.entrenar_set([[0,0],[0,1],[1,0],[1,1]], [[0],[1],[1],[0]], EPOCHS, LEARNING_RATE, TOLERANCIA)
	print('EPOCHS', e)
	
def main():
	pruebas_neuronales()
	
main()
