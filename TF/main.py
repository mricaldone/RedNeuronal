import tensorflow as tf
import tflearn
from mnist import MNIST
import time
import random
	
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

def crear_modelo(LEARNING_RATE):
	tf.reset_default_graph()
	red = tflearn.input_data([None, 784])
	red = tflearn.fully_connected(red, 196, activation='sigmoid')
	red = tflearn.fully_connected(red, 10, activation='sigmoid')
	red = tflearn.fully_connected(red, 1, activation='sigmoid')
	red = tflearn.regression(red, optimizer='sgd', learning_rate=LEARNING_RATE, loss='mean_square')
	modelo = tflearn.DNN(red)
	return modelo

def mnist_test():
	LEARNING_RATE = 1
	EPOCHS = 5
	TOLERANCIA = 0.05
	mndata = MNIST('samples')
	
	print("CARGANDO DATOS DE ENTRENAMIENTO")
	images, labels = mndata.load_training()
	print("PRE-PROCESANDO SALIDAS")
	labels = preprocesar_salidas(labels)
	print("PRE-PROCESANDO ENTRADAS")
	images = preprocesar_entradas(images)
	print("GENERANDO RED NEURONAL")
	rn = crear_modelo(LEARNING_RATE)
	print("ENTRENANDO")
	start = time.time()
	rn.fit(images, labels, validation_set=0.1, show_metric=True, batch_size=1, n_epoch=EPOCHS)
	end = time.time()
	print('TIEMPO TOTAL:',end - start)
	print("CARGANDO DATOS DE PRUEBA")
	images, labels = mndata.load_testing()
	print("INICIANDO PRUEBA")
	while input("CONTINUAR? (Y/N):") != "N":
		index = random.randrange(0, len(images))
		imagen = preprocesar_entrada(images[index])
		r = rn.predict([imagen])
		print(mndata.display(images[index]))
		print("Esperado", labels[index])
		print("Resultado:", round(r[0][0] * 10,0))

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
	
def main():
	mnist_test()
	
main()
