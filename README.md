# RedNeuronal

Modelo de red neuronal.

## Información previa

Este paquete contiene dos versiones de una red neuronal, una orientada a objetos que se encuentra dentro de la carpeta 'OO' y otra matricial que se encuentra en la raiz del proyecto. La version matricial es mucho mas rapida ya que utiliza varios nucleos del procesador para realizar las operaciones, esto es gracias a la libreria numpy que debe ser previamente instalada utilizando:
```
pip3 install numpy
```
Por otro lado la versión orientada a objetos no requiere numpy. Sin embargo es mucho mas lenta.

## Instalación (Versión Matricial)

Instalar la libreria numpy. Colocar la carpeta RedNeuronal en la raiz del proyecto. Finalmente, importar al proyecto utilizando:
```
from RedNeuronal.RedNeuronal import *
```

## Instalación (Versión O.O.)

Colocar la carpeta RedNeuronal en la raiz del proyecto. Finalmente, importar al proyecto utilizando:
```
from RedNeuronal.OO.RedNeuronal import *
```

## Modo de uso

### Crear estructura de la red

La red neuronal consiste en un conjunto de capas de neuronas en donde cada neurona de una capa se conecta con todas las neuronas de la siguiente. Para esto necesitamos pasarle la informacion respecto de la cantidad de entradas y la estructura al constructor. Ademas se puede definir la funcion de activación de manera optativa. La funcion de activación por defecto es la sigmoidea. El prototipo es el siguiente:
```
RedNeuronal(cant_entradas, estructura, f_activacion = Sigmoide())
```
Donde cant_entradas es un numero entero que posee la cantidad de entradas de la red y estructura es una lista donde cada elemento es un entero que indica la cantidad de neuronas por capa. Adicionalmente se puede definir una funcion de activacion del archivo Funciones.py.

Un ejemplo de uso podría ser:
```
RedNeuronal(2500, [2500,70,2])
```
En este tipo particular la cantidad de entradas es 2500, la cantidad de neuronas de la primer capa es 2500, para la segunda 70 y finalmente para la tercera 2. Además la funcion de activación es la sigmoide.
Cabe destacar que la red tiene tantas salidas como neuronas en la ultima capa. Para el ejemplo, la cantidad de salidas es dos.

### Entrenar red

El entrenamiento de la red consiste en indicarle a la red la salida deseada para una entrada determinada.
Para entrenar la red se utiliza el metodo:
```
entrenar(entradas, valores_esperados, learning_rate = 0.01):
```
Dónde 'entradas' es una lista en la cual cada elemento contiene un valor para cada entrada de la red; 'valores_esperados' es una lista donde cada elemento es la salida deseada; y 'learning_rate' es la velocidad de entrenamiento.
Por ejemplo:
```
rn = RedNeuronal(4,[4,2])
rn.entrenar([0.5, 0.7, 1, 0.5],[1, 0])
```
El método entrenar no devuelve ningun valor. Lo que logra es modificar las veriables internas de la red para que a partir de las entradas dadas el resultado sea el mas aproximado.

### Procesar entradas

Una vez entrenada la red es necesario verificar que el entrenamiento fue el adecuado. Esto lo hacemos mediante el metodo procesar.
```
procesar(entradas)
```
Donde 'entradas' es una lista en la cual cada elemento representa un valor de entrada de la red. Este metodo devuelve una lista donde cada elemento es el valor de salida de la red. Recuerde que la lista de salida tendrá tantos elementos como neuronas en la ultima capa de la red.
Debe tener en cuenta que procesar las entradas en una red sin entrenamiento devolverá un resultado totalmente aleatorio.

### Otras funciones de activacion

## Buenas practicas
Las entradas deben estar normalizadas.
Es recomendable utilizar la misma cantidad de neuronas en la primer capa que de entradas a la red.
Los datasets deben ser entrenados de manera completa. Cada vez que se recorre un dataset se llama epoca. Una baja cantidad de epocas provoca underfitting y una alta cantidad de epocas puede provocar overfitting.
Un learning rate muy alto puede provocar que nunca se alcanze el resultado adecuado. Un learning rate muy bajo puede provocar la necesidad de utilizar mas epocas.
Es importante que los datasets posean la mayor cantidad de datos variados para que la generalizacion sea mayor.
Para probar que la red no esta memorizando en lugar de aprender, no se deben utilizar los mismos datos de prueba que de entrenamiento.
