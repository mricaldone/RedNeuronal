from RedNeuronal import *
from mnist import MNIST
from PIL import Image
import numpy as np
import random
import time

def test():
	LEARNING_RATE = 10
	EPOCHS = 1000
	TOLERANCIA = 0.1
	F = Sigmoide()
	#F = Relu()
	print('TEST RED NEURONAL')
	print('PRUEBA 1')
	rn = RedNeuronal(2, [1], F)
	rn.entrenar([1,1],[0])
	print('PRUEBA 2')
	rn = RedNeuronal(4, [2,2,4], F)
	rn.entrenar([1,1,1,1],[1,1,1,1])
	print('PRUEBA 3')
	rn = RedNeuronal(4, [4], F)
	rn.entrenar([1,1,1,1],[1,1,1,1])
	print('PRUEBA 4')
	rn = RedNeuronal(2, [1], F)
	rn.entrenar([1,1],[1])
	print('PRUEBA COMPUERTA YES')
	rn = RedNeuronal(1, [1], F)
	datos = [[1],[0]]
	esperados = [[1],[0]]
	print('EPOCHS', rn.entrenar_set(datos, esperados, LEARNING_RATE, EPOCHS, TOLERANCIA))
	print(rn.procesar([1]))
	print(rn.procesar([0]))
	print('PRUEBA COMPUERTA NOT')
	rn = RedNeuronal(1, [1], F)
	datos = [[1],[0]]
	esperados = [[0],[1]]
	print('EPOCHS', rn.entrenar_set(datos, esperados, LEARNING_RATE, EPOCHS, TOLERANCIA))
	print(rn.procesar([1]))
	print(rn.procesar([0]))
	print('PRUEBA COMPUERTA AND')
	rn = RedNeuronal(2, [1], F)
	datos = [[1,1],[1,0],[0,1],[0,0]]
	esperados = [[1],[0],[0],[0]]
	print('EPOCHS', rn.entrenar_set(datos, esperados, LEARNING_RATE, EPOCHS, TOLERANCIA))
	print(rn.procesar([1,1]))
	print(rn.procesar([1,0]))
	print(rn.procesar([0,1]))
	print(rn.procesar([0,0]))
	print('PRUEBA COMPUERTA OR')
	rn = RedNeuronal(2, [1], F)
	datos = [[1,1],[1,0],[0,1],[0,0]]
	esperados = [[1],[1],[1],[0]]
	print('EPOCHS', rn.entrenar_set(datos, esperados, LEARNING_RATE, EPOCHS, TOLERANCIA))
	print(rn.procesar([1,1]))
	print(rn.procesar([1,0]))
	print(rn.procesar([0,1]))
	print(rn.procesar([0,0]))
	print('PRUEBA COMPUERTA XAND')
	rn = RedNeuronal(2, [2,1], F)
	datos = [[1,1],[1,0],[0,1],[0,0]]
	esperados = [[1],[0],[0],[1]]
	print('EPOCHS', rn.entrenar_set(datos, esperados, LEARNING_RATE, EPOCHS, TOLERANCIA))
	print(rn.procesar([1,1]))
	print(rn.procesar([1,0]))
	print(rn.procesar([0,1]))
	print(rn.procesar([0,0]))
	print('PRUEBA COMPUERTA XOR')
	rn = RedNeuronal(2, [2,1], F)
	datos = [[1,1],[1,0],[0,1],[0,0]]
	esperados = [[0],[1],[1],[0]]
	print('EPOCHS', rn.entrenar_set(datos, esperados, LEARNING_RATE, EPOCHS, TOLERANCIA))
	print(rn.procesar([1,1]))
	print(rn.procesar([1,0]))
	print(rn.procesar([0,1]))
	print(rn.procesar([0,0]))
	
def preprocesar_salidas(salidas):
	nuevas_salidas = []
	for valor in salidas:
		nuevas_salidas.append([valor/10])
	return nuevas_salidas

def preprocesar_entrada(datos_de_entrada):
	nuevos_datos = []
	for valor in datos_de_entrada:
		nuevos_datos.append(valor/255)
	return nuevos_datos
	
def preprocesar_entradas(set_de_entradas):
	nuevas_entradas = []
	for entradas in set_de_entradas:
		nuevas_entradas.append(preprocesar_entrada(entradas))
	return nuevas_entradas
	
def mnist_test():
	LEARNING_RATE = 1
	EPOCHS = 2
	TOLERANCIA = 0.05
	F = Sigmoide()
	mndata = MNIST('samples')
	
	print("CARGANDO DATOS DE ENTRENAMIENTO")
	images, labels = mndata.load_training()
	print("PRE-PROCESANDO SALIDAS")
	labels = preprocesar_salidas(labels)
	print("PRE-PROCESANDO ENTRADAS")
	images = preprocesar_entradas(images)
	print("GENERANDO RED NEURONAL")
	rn = RedNeuronal(784, [196,10,1], F)
	#rn = RedNeuronal(784, [784,392,196,98,49,25,10,1], F)
	print("ENTRENANDO")
	start = time.time()
	print('EPOCHS', rn.entrenar_set(images, labels, LEARNING_RATE, EPOCHS, TOLERANCIA))
	end = time.time()
	print('TIEMPO TOTAL:',end - start)
	print("CARGANDO DATOS DE PRUEBA")
	images, labels = mndata.load_testing()
	print("INICIANDO PRUEBA")
	while input("CONTINUAR? (Y/N):") != "N":
		index = random.randrange(0, len(images))
		imagen = preprocesar_entrada(images[index])
		r = rn.procesar(imagen)
		print(mndata.display(images[index]))
		print("Esperado", labels[index])
		print("Resultado:", round(r[0] * 10,0))
		

def imprimir_grafico(rn, paso, cant_decimales):
	#w, h = 512, 512
	#data = np.zeros((h, w, 3), dtype=np.uint8)
	#data[256, 256] = [255, 0, 0]
	#img = Image.fromarray(data, 'RGB')
	#img.save('my.png')
	#img.show()
	x = 0
	y = 0
	dim = int(1/paso) + 1
	#DEFINO EL TAMAÑO DEL PAPEL
	w, h = dim, dim
	data = np.zeros((h, w, 3), dtype=np.uint8)
	for i in range(dim):
		x = 0
		for j in range(dim):
			r = rn.procesar([x,y])[0]
			pixel = [int(255 * r), 0, 0]
			data[i,j] = pixel
			x = x + paso
		y = y + paso
	img = Image.fromarray(data, 'RGB')
	img.transpose(Image.FLIP_TOP_BOTTOM).show()
		
def pruebas_neuronales():
	LEARNING_RATE = 0.5
	EPOCHS = 10000
	TOLERANCIA = 0.1
	F = Sigmoide()
	print('PRUEBA COMPUERTA AND')
	rn = RedNeuronal(2, [1], F)
	datos = [[1,1],[1,0],[0,1],[0,0]]
	esperados = [[1],[0],[0],[0]]
	print('EPOCHS', rn.entrenar_set(datos, esperados, LEARNING_RATE, EPOCHS, TOLERANCIA))
	imprimir_grafico(rn, 0.005, 2)
	print('PRUEBA COMPUERTA OR')
	rn = RedNeuronal(2, [1], F)
	datos = [[1,1],[1,0],[0,1],[0,0]]
	esperados = [[1],[1],[1],[0]]
	print('EPOCHS', rn.entrenar_set(datos, esperados, LEARNING_RATE, EPOCHS, TOLERANCIA))
	imprimir_grafico(rn, 0.005, 2)
	print('PRUEBA COMPUERTA XAND')
	rn = RedNeuronal(2, [2,1], F)
	datos = [[1,1],[1,0],[0,1],[0,0]]
	esperados = [[1],[0],[0],[1]]
	print('EPOCHS', rn.entrenar_set(datos, esperados, LEARNING_RATE, EPOCHS, TOLERANCIA))
	imprimir_grafico(rn, 0.005, 2)
	print('PRUEBA COMPUERTA XOR')
	rn = RedNeuronal(2, [2,1], F)
	datos = [[1,1],[1,0],[0,1],[0,0]]
	esperados = [[0],[1],[1],[0]]
	print('EPOCHS', rn.entrenar_set(datos, esperados, LEARNING_RATE, EPOCHS, TOLERANCIA))
	imprimir_grafico(rn, 0.005, 2)
	print('PRUEBA CIRCULO')
	rn = RedNeuronal(2, [4,1], F)
	datos = [[0,0.5],[0.25,0.25],[0.25,0.75],[0.5,0],[0.5,1],[0.75,0.25],[0.75,0.75],[1,0.5],[0.5,0.5]]
	esperados = [[1],[1],[1],[1],[1],[1],[1],[1],[0]]
	print('EPOCHS', rn.entrenar_set(datos, esperados, LEARNING_RATE, EPOCHS, TOLERANCIA))
	imprimir_grafico(rn, 0.005, 2)
	print(rn.procesar([0,0.5]))
	print(rn.procesar([0.25,0.25]))
	print(rn.procesar([0.25,0.75]))
	print(rn.procesar([0.5,0]))
	print(rn.procesar([0.5,1]))
	print(rn.procesar([0.75,0.25]))
	print(rn.procesar([0.75,0.75]))
	print(rn.procesar([1,0.5]))
	print(rn.procesar([0.5,0.5]))
	
def main():
	#pruebas_neuronales()
	mnist_test()
	#test()
	
main()
