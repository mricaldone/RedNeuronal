from mnist import MNIST
from RedNeuronal import *
import os
import time
import random

NOMBRE_ARCHIVO = 'numeros'
MNDATA = MNIST('samples')
F = Sigmoide()
N_ENTRADAS = 784
CAPAS = [196,10,1]
	
def si_no_prompt(texto):
	opc_si = ['s','S']
	opc_no = ['n','N']
	while True:
		opcion = input(texto + ' (S/N): ')
		if opcion in opc_si:
			return True
		if opcion in opc_no:
			return False

def existen_datos(nombre_archivo):
	if(os.path.exists(nombre_archivo + '.dat')):
		return si_no_prompt('YA EXISTE UNA RED NEURONAL ENTRENADA. ¿DESEA UTILIZARLA?')
	return False

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

def crear_red(rn):
	LEARNING_RATE = 1
	EPOCHS = 2
	TOLERANCIA = 0.05
	print("CARGANDO DATOS DE ENTRENAMIENTO")
	images, labels = MNDATA.load_training()
	print("PRE-PROCESANDO SALIDAS")
	labels = preprocesar_salidas(labels)
	print("PRE-PROCESANDO ENTRADAS")
	images = preprocesar_entradas(images)
	print("ENTRENANDO")
	start = time.time()
	print('EPOCHS', rn.entrenar_set(images, labels, LEARNING_RATE, EPOCHS, TOLERANCIA))
	end = time.time()
	print('TIEMPO TOTAL:',end - start)
	print('GUARDANDO DATOS DE LA RED')
	rn.guardar(NOMBRE_ARCHIVO)

def cargar_red(rn):
	print('CARGANDO DATOS DE LA RED')
	rn.cargar(NOMBRE_ARCHIVO)
	
def generar_red():
	print("GENERANDO RED NEURONAL")
	return RedNeuronal(N_ENTRADAS, CAPAS, F)

def cargar_prueba():
	print("CARGANDO DATOS DE PRUEBA")
	return MNDATA.load_testing()

def mostrar_prueba(images, labels, rn):
	print("INICIANDO PRUEBA")
	while si_no_prompt("¿CONTINUAR?"):
		index = random.randrange(0, len(images))
		imagen = preprocesar_entrada(images[index])
		r = rn.procesar(imagen)
		print(MNDATA.display(images[index]))
		val_esperado = labels[index]
		val_obtenido = round(r[0] * 10,0)
		print("Esperado:", val_esperado)
		print("Resultado:", val_obtenido)
		if val_esperado != val_obtenido:
			print('Fallo')
		else:
			print('Correcto')

def main():
	os.system('clear')
	rn = generar_red()
	if existen_datos(NOMBRE_ARCHIVO):
		cargar_red(rn)
	else:
		crear_red(rn)
	images, labels = cargar_prueba()
	mostrar_prueba(images, labels, rn)
	
main()
