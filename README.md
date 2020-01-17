# RED NEURONAL

Éste es un modelo (TDA) muy básico (aunque no menos potente) de una red neuronal. 

## INFORMACIÓN PREVIA

Éste paquete contiene dos versiones de una red neuronal, una orientada a objetos que se encuentra dentro de la carpeta 'OO' y otra matricial que se encuentra en la raíz del proyecto. La versión matricial es mucho más rápida ya que utiliza varios núcleos del procesador para realizar las operaciones, esto es gracias a la librería numpy que debe ser previamente instalada. Por otro lado la versión orientada a objetos no requiere numpy y el código es más legible. Sin embargo es mucho más lenta.

## INSTALACIÓN (VERSIÓN MATRICIAL)

1. Instalar la librería numpy.
```
pip3 install numpy
```
1. Colocar la carpeta RedNeuronal en la raíz del proyecto.
1. Finalmente, importar al proyecto utilizando:
```
from RedNeuronal.RedNeuronal import *
```

## INSTALACIÓN (VERSIÓN O.O.)

1. Colocar la carpeta RedNeuronal en la raíz del proyecto.
1. Importar al proyecto utilizando:
```
from RedNeuronal.OO.RedNeuronal import *
```

## MODO DE USO

### Crear estructura de la red

La red neuronal consiste en un conjunto de capas de neuronas en donde cada neurona de una capa se conecta con todas las neuronas de la siguiente. Para esto necesitamos pasarle la información respecto de la cantidad de entradas y la estructura al constructor. Además se puede definir la función de activación de manera optativa. La función de activación por defecto es la sigmoidea. El prototipo es el siguiente:
```
RedNeuronal(cant_entradas, estructura, f_activacion = Sigmoide())
```
Dónde *cant_entradas* es un número entero que posee la cantidad de entradas de la red y *estructura* es una lista donde cada elemento es un entero que indica la cantidad de neuronas por capa. Adicionalmente se puede definir en *f_activacion* una función de activación del archivo Funciones.py.

Un ejemplo de uso podría ser:
```
RedNeuronal(2500, [2500,70,2])
```
En este tipo particular la cantidad de entradas es 2500, la cantidad de neuronas de la primer capa es 2500, para la segunda 70 y finalmente para la tercera 2. Además la función de activación es la sigmoide.
Cabe destacar que la red tiene tantas salidas como neuronas en la última capa. Para el ejemplo, la cantidad de salidas es dos.

### Entrenar red

El entrenamiento de la red consiste en indicarle a la red la salida deseada para una entrada determinada.
Para entrenar la red se utiliza el método:
```
entrenar(entradas, valores_esperados, learning_rate = 0.01):
```
Dónde *entradas* es una lista en la cual cada elemento contiene un valor para cada entrada de la red; *valores_esperados* es una lista donde cada elemento es la salida deseada; y *learning_rate* es la velocidad de entrenamiento.
Por ejemplo:
```
rn = RedNeuronal(4,[4,2])
rn.entrenar([0.5, 0.7, 1, 0.5],[1, 0])
```
El método *entrenar* no devuelve ningún valor. Lo que logra es modificar las variables internas de la red para que a partir de las entradas dadas el resultado sea el más aproximado.

### Procesar entradas

Una vez entrenada la red es necesario verificar que el entrenamiento fue el adecuado. Esto lo hacemos mediante el método procesar.
```
procesar(entradas)
```
Donde *entradas* es una lista en la cual cada elemento representa un valor de entrada de la red. Este método devuelve una lista donde cada elemento es el valor de salida de la red. Recuerde que la lista de salida tendrá tantos elementos como neuronas en la última capa de la red.
Ejemplo:
```
rn = RedNeuronal(4,[4,2])
#AQUI DEBERIAN IR LAS FUNCIONES DE ENTRENAMIENTO
resultado = rn.procesar([0.4, 0.2, 0.2, 0.5])
print(resultado)
```
Debe tener en cuenta que procesar las entradas en una red sín entrenamiento devolverá un resultado totalmente aleatorio.

### Guardar y Cargar
Es posible guardar el estado de la red utilizando el método guardar.
```
guardar(nombre_archivo)
```
Dónde 'nombre_archivo' es el nombre del archivo donde se almacenarán los datos.
Para cargar estos datos se utiliza el metodo cargar, que es análogo al anterior.
```
cargar(nombre_archivo)
```
Ejemplo:
```
rn1 = RedNeuronal(4,[4,2])
#AQUI DEBERIAN IR LAS FUNCIONES DE ENTRENAMIENTO
rn1.guardar('datos.rn')
rn2 = RedNeuronal(4,[4,2])
resultado = rn2.procesar([1,1,1,1])
print(resultado)
```
Es importante que la estructura de la red que carga los datos sea la misma que la de la red que los graba.

### Otras funciones de activación
Los siguientes son los prototipos de las funciones de activación disponibles en el archivo Funciones.py:
```
Sigmoide()
Tanh()
Relu()
```
Tenga en cuenta que todas las capas utilizan la misma función de activación que fue definida en el constructor de la red. Además tenga en cuenta que la imagen de la salida dependerá de la función de activación utilizada. Por ejemplo para la sigmoide la imagen es [0;1], para tanh [-1;1] y para relu [0;inf].

## BUENAS PRÁCTICAS
* Las entradas deben estar normalizadas. Es recomendable pre-procesar las entradas según la función de activación utilizada.
* Las salidas utilizadas durante el entrenamiento deben estar dentro del codominio de la función de activación. Por lo tanto deben ser pre-procesadas. Adicionalmente, las salidas de la red deben ser post-procesadas de manera inversa al pre-procesamiento del entrenamiento.
* Es recomendable utilizar la misma cantidad de neuronas en la primer capa que de entradas a la red.
* Los datasets deben ser entrenados de manera completa. Cada vez que se recorre un dataset se llama época. Una baja cantidad de épocas provoca underfitting y una alta cantidad de épocas puede provocar overfitting.
* Un learning rate muy alto puede provocar que nunca se alcance el resultado adecuado. Un learning rate muy bajo puede provocar la necesidad de utilizar más epocas.
* Es importante que los datasets posean la mayor cantidad de datos variados para que la red aprenda a generalizar.
* Para probar que la red no está memorizando en lugar de generalizar, no se deben utilizar los mismos datos de prueba que de entrenamiento. Es recomendable utilizar el 80% de los datos disponibles para entrenamiento y el 20% para realizar pruebas.
