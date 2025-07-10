[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/Lj3YlzJp)
# Proyecto Final 2025-1: AI Neural Network
## **CS2013 Programación III** · Informe Final

### **Descripción**

> Ejemplo: Implementación de una red neuronal multicapa en C++ para clasificación de dígitos manuscritos.

### Contenidos

1. [Datos generales](#datos-generales)
2. [Requisitos e instalación](#requisitos-e-instalación)
3. [Investigación teórica](#1-investigación-teórica)
4. [Diseño e implementación](#2-diseño-e-implementación)
5. [Ejecución](#3-ejecución)
6. [Análisis del rendimiento](#4-análisis-del-rendimiento)
7. [Trabajo en equipo](#5-trabajo-en-equipo)
8. [Conclusiones](#6-conclusiones)
9. [Bibliografía](#7-bibliografía)
10. [Licencia](#licencia)
---

### Datos generales

* **Tema**: Redes Neuronales en AI
* **Grupo**: `group_3_custom_name`
* **Integrantes**:

  * Matos Copello Rayhan Derek – 202410377 (Responsable de investigación teórica)
  * Tamara Ureta, Anyeli Azumi – 202410590 (Responsable de investigación teórica)
  * Alvarado León, Adriana Celeste – 209900002 (Diseño e implementación)
  * Mattos Gutierrez, Angel Daniel – 202420199 (Implementación del modelo)
  * Aquino Castro Farid Jack – 202410569 (Análisis y Rendimiento)
  * Portugal Vilca Julio Cesar – 202410487 (Documentación y demo)
> **Nota**: El informe completo se encuentra adjunto en el repositorio del proyecto.
---

### Requisitos e instalación

1. **Compilador**: GCC 11 o superior, Clang 19 o superior
2. **Dependencias**:\
  **Generales:**
   * CMake 3.12+
   * OpenCV 4.6.0+
   * OpenMP (Incluido en GCC, instalación adicional en Clang)
   * FFmpeg 7.0+
     
   **Linux:**
   * Xlib (libX11) 
   * libpipewire 0.3+
3. **Instalación**:

   ```bash
   git clone https://github.com/CS1103/projecto-final-conciencia.git
   cd proyecto-final
   mkdir build && cd build
   cmake ..
   make
   ```

Al compilar, encontrarás con el ejecutable principal (popn_ai) y dos carpetas adicionales.
- (generator/popn_generator) Generador de imágenes
- (tests/batch_evaluator) Test de datos de prueba
- (tests/image_evaluator) Test de reconocimiento de imágenes
- (tests/gameplay_evaluator) Test de gameplay (requiere una ROM de Pop'n Music 10 y el emulador PCSX2)

#### Configuracion del emulador PCSX2 (gameplay_evaluator)
En el apartado gráfico, configurar la sección de Renderizado (Rendering), para cambiar la resolución nativa del emulador a 3x Nativo (~1080px/FHD). Adicionalmente, en el apartado de controladores, cambiar el de PlayStation 2 por un control de Pop'n Music y configurarlo de la siguiente manera:
```
  F   G   H   J
C   V   B   N   M
```
Puedes guiarte a partir del orden que se muestra en el menu de el emulador. De lo contrario, no va a funcionar el test de gameplay.

---

### 1. Investigación teórica

**Objetivo General**:

* Desarrollar una red neuronal que pueda predecir los 512 posibles patrones basados en las imágenes del videojuego Pop’n Music.

**Objetivos específicos**

* Crear las clases Tensor, IOptimizer, ILayer, Neural Network,  y ILoss. 
* Crear sus clases hijas ReLU, Dense, MSELoss, BCELoss, CrossEntropyLoss, SGD,  Adam. 
* Crear imágenes con diferentes patrones para el entrenamiento de la red. 
* Obtener los pesos calculados para realizar los test y calcular la precisión del modelo.

## Marco Teórico

### 1.1 Redes Neuronales Artificiales

Las redes neuronales artificiales (RNA) son simulaciones informáticas basadas en el funcionamiento del cerebro humano. Estas redes se componen de un grupo de nodos conocidos como neuronas, que se agrupan en niveles y se vinculan entre ellos mediante enlaces conocidos como pesos sinápticos que pueden ser modificados. Cada neurona acoge señales de entrada, las maneja a través de una función de activación y produce una salida que puede ser enviada a otras neuronas en niveles sucesivos [1].

Una red neuronal común se segmenta en tres componentes básicos. La capa de entrada acoge la información en bruto que será procesada, como pueden ser imágenes o señales. Las capas encubiertas llevan a cabo cálculos y transformaciones no lineales que facilitan a la red el aprendizaje de representaciones intermedias y patrones complicados. Finalmente, la capa de salida proporciona la predicción definitiva, la cual puede ser una clasificación, una probabilidad o un volumen determinado [2].

El proceso de aprendizaje en una red neuronal implica la modificación de los pesos de sus conexiones con el fin de reducir la discrepancia entre la salida producida y el valor verdadero anticipado. Este proceso se lleva a cabo a través de algoritmos de optimización, siendo el algoritmo de retropropagación del error (backpropagation) el más empleado. Este calcula la manera en que cada peso debe ser modificado para disminuir el error total [3]. Estas modificaciones están dirigidas por una función de pérdida, que evalúa el grado de error en cada época. Mediante este ciclo de capacitación, la red incrementa su habilidad para realizar pronósticos exactos.

Cuando las redes neuronales aumentan su número de capas ocultas, se convierten en redes profundas, lo que permite que aprendan representaciones jerárquicas de los datos. Las redes profundas han demostrado un desempeño sobresaliente en tareas complejas como el reconocimiento de imágenes, la traducción automática y el análisis de voz [1].

#### 1.1.1 Investigación teórica: fundamentos y arquitecturas

El estudio de las redes neuronales artificiales tiene sus raíces en la década de 1940 con el modelo de McCulloch y Pitts, pero fue recién en los años 80 y 90 que se consolidaron algoritmos como el perceptrón multicapa (MLP) y el backpropagation. Con el avance del hardware y el crecimiento de datasets, se introdujeron arquitecturas más profundas y especializadas.

Entre las arquitecturas más representativas se encuentran:

- **MLP (Multi-Layer Perceptron)**: compuesta por capas totalmente conectadas. Es la base del modelo usado en este proyecto y se caracteriza por transformar entradas vectoriales mediante pesos, biases y funciones de activación.
- **CNN (Convolutional Neural Network)**: especializada en el análisis de imágenes. Utiliza filtros para detectar patrones locales y jerárquicos.
- **RNN (Recurrent Neural Network)**: diseñada para datos secuenciales, como texto o audio. Posee conexiones recurrentes que permiten conservar memoria del estado anterior.

Para el entrenamiento de estas redes se utiliza el algoritmo de retropropagación, propuesto en su forma moderna por Rumelhart, Hinton y Williams (1986), que calcula el gradiente de la función de pérdida respecto a cada peso mediante la regla de la cadena.

Además, se requiere un algoritmo de optimización que ajuste los pesos. En este proyecto se usaron dos: **SGD** y **Adam**, seleccionados por su bajo costo computacional y capacidad de adaptación a diferentes tasas de aprendizaje, respectivamente.

Esta base teórica permitió estructurar el desarrollo modular del sistema en C++, reproduciendo comportamientos esenciales del aprendizaje profundo mediante clases y estructuras propias inspiradas en bibliotecas como **PyTorch** y **TensorFlow**.

---

### 1.2 Visión por Computadora

La visión computacional es una disciplina de la inteligencia artificial que permite a los ordenadores analizar y procesar imágenes o vídeos con el objetivo de recopilar datos relevantes del entorno [4]. Esta disciplina tiene como objetivo emular la capacidad humana para entender lo que se percibe, pero a través de algoritmos y esquemas matemáticos.

En este proyecto, se utiliza la visión computacional para examinar imágenes producidas a partir de un juego de ritmo japonés. El sistema tiene que reconocer patrones visuales que simbolizan distintas combinaciones de notas musicales que se descienden por carriles determinados, replicando la experiencia de juegos como *Pop'n Music* o *Beatmania IIDX*.

Para facilitar la interpretación de las imágenes, se utilizan las siguientes técnicas de preprocesamiento:

- **Conversión a escala de grises**: Este procedimiento elimina la información de color y mantiene únicamente los niveles de intensidad de cada píxel. Según Goodfellow, Bengio y Courville [3], esto reduce significativamente la complejidad computacional, ya que la red neuronal trabaja con menos información redundante, centrándose únicamente en los patrones estructurales.

- **Reducción de tamaño**: Consiste en disminuir la resolución de las imágenes para optimizar tanto el almacenamiento como la velocidad de procesamiento. Este enfoque es especialmente útil cuando se trabaja con grandes volúmenes de datos, permitiendo entrenar modelos más rápidamente sin sacrificar demasiado la calidad de las características relevantes.

- **Normalización**: Hace referencia a incrementar los valores de los píxeles dentro del intervalo [0, 1]. Este procedimiento es esencial para consolidar el proceso de entrenamiento, puesto que contribuye a que las funciones de activación y los algoritmos de optimización funcionen de forma más eficaz [3].

---

### 1.3 Generación de Datos Sintéticos

La generación de datos sintéticos consiste en la creación de ejemplos artificiales mediante algoritmos o modelos matemáticos, en lugar de recolectarlos directamente del entorno real [5]. Este enfoque es especialmente útil cuando los datos reales son escasos, costosos de adquirir, o presentan restricciones legales y éticas relacionadas con la privacidad [6].

En el contexto de este proyecto, se diseñó un generador en C++ utilizando OpenCV, capaz de sintetizar imágenes de un juego de ritmo japonés con diferentes combinaciones de notas. Las principales ventajas de este método son:

- **Escalabilidad y costo reducido**: Una vez desarrollado el entorno sintético, es posible generar grandes volúmenes de datos de forma automática y económica [7].

- **Etiquetado automático y exacto**: Cada imagen generada se acompaña de etiquetas precisas, lo cual evita errores comunes en el etiquetado manual.

- **Control de variabilidad**: Se puede simular ruido visual, diferentes niveles de dificultad y combinaciones específicas, garantizando un dataset robusto frente a condiciones reales [6].

Investigaciones como la realizada por Lu et al. [6] destacan cómo los datos artificiales facilitan la superación de los desafíos de calidad y privacidad. Además, Bauer et al. [7] indican que las herramientas fundamentadas en redes neuronales, como **GANs**, **modelos de difusión** y **transformers**, permiten el control de la generación de datos en contextos complejos.

---

### 1.4 Funciones de Activación

Las funciones de activación introducen la no linealidad necesaria para que una red neuronal aprenda relaciones complejas entre los datos. Sin estas funciones, la red se comportaría como una combinación lineal, lo cual limitaría drásticamente su capacidad de modelado.

- **ReLU (Rectified Linear Unit)**: definida como 𝑓(𝑥) = max(0, 𝑥), fue introducida originalmente por Kunihiko Fukushima en 1969 [9]. Su uso modernizado en redes profundas se consolidó en 2011 por su capacidad para evitar el problema del gradiente desvanecido, acelerar el aprendizaje y generar activaciones esparsas [10], [11].

- **Softmax**: transforma un vector de valores reales en una distribución de probabilidad cuya suma es 1. Está relacionada con la distribución de Boltzmann en mecánica estadística [12], formalizada en el aprendizaje automático para clasificación multiclase [13]. Es clave en la capa de salida y permite aplicar la entropía cruzada como función de pérdida [3].

---

### 1.5 Algoritmos de Optimización

Los algoritmos de optimización son esenciales en el entrenamiento de redes neuronales artificiales, ya que permiten ajustar los pesos de la red con el objetivo de minimizar la función de pérdida [15].

- **Descenso del Gradiente Estocástico (SGD)**: destaca por su eficiencia computacional al trabajar con mini batches. Sin embargo, puede quedar atrapado en mínimos locales o generar oscilaciones [16].

- **Adam (Adaptive Moment Estimation)**: propuesto por Kingma y Ba [17], combina momentum y RMSProp para ajustar tasas de aprendizaje por parámetro. Su fortaleza radica en manejar datos complejos y lograr convergencia rápida y estable.

**Comparación**: Aunque Adam suele ofrecer resultados iniciales superiores, estudios como el de Wilson et al. [18] sugieren que en ciertos casos SGD puede generar soluciones más generalizables a largo plazo.

---

### 1.6 Sobreajuste y Técnicas de Regularización

Durante el entrenamiento de una red neuronal, es común que el modelo se ajuste demasiado a los datos de entrenamiento, lo que reduce su capacidad para generalizar ante datos nuevos. Este fenómeno se conoce como **sobreajuste (overfitting)**.

En el presente proyecto, este riesgo implica que el modelo podría funcionar bien solo con imágenes iguales a las entrenadas, fallando ante leves variaciones. Por ello, se emplean técnicas de **regularización** para evitarlo y asegurar que el modelo reconozca patrones nuevos con precisión durante la inferencia.

## Desarrollo de componentes claves:

Para diseñar e implementar la red neuronal se incluyeron las siguientes clases para poder simular todo su comportamiento, estas son: Tensor, Activation, Layer, Dense, Loss, Optimización y la clase Neural_Network donde se alojarán todas nuestras funciones. A continuación se detalla su inclusión.

En primer lugar, se implementó la clase Tensor.h para manejar toda la información que ingresa a la red neuronal, en estos se incluyen los datos los cuales uno entrenará y otro con los valores que desea predecir. El motivo por el cual se escogieron los tensores es para poder trabajar con grandes volúmenes de información de forma eficiente, debido a que estos se componen principalmente de datos binarios. Los modelos aumentan en complejidad y tamaño, los puntos de control distribuidos se convierten en un componente crucial del proceso de entrenamiento. No obstante estos suelen generar importantes demandas de almacenamiento. Para abordar este desafío, la comprensión surge como una solución natural. Dado que los puntos de control se componen principalmente de datos binarios (tensores)[23]. Es por ello que para desarrollar esta clase tomamos como referencia la biblioteca Pytorch.

En esta biblioteca trabajamos simulando los distintos métodos de operaciones con matrices como por ejemplo la función de transposición, multiplicación escalar, tensor broadcasting, lo cual permitió operar entre tensores de distintas dimensiones respetando reglas similares a las de bibliotecas como Numpy. 

En el marco del proyecto, esto fue crucial para poder representar correctamente tanto las imágenes de entrada generadas a partir de combinaciones de notas del videojuego Pop’n Music, como también los pesos y salidas de cada capa de la red neuronal. Esta implementación ayuda durante el entrenamiento del modelo, por ejemplo en la aplicación aplicar gradientes y operaciones de retropropagación de manera controlada y precisa. 

Las clases layers son las capas que conectadas entre sí, hacen posible que funcione el proceso de entrenamiento. Hay tres partes donde se presentan estas capas: Una capa de entrada, donde se ingresa los valores iniciales del entrenamiento; las capas ocultas, donde se realiza todo el proceso de cálculo y ponderación; por último la capa de salida con los valores de destino que se busca en el entrenamiento. Durante todas estas capas se busca que cada valor inicial que ingrese se procese dentro de las capas ocultas con ponderaciones aleatorias que se van actualizando gradualmente para acercarlas a la capa de destino. Este proceso sigue su curso hasta un punto en que varias capas coincidan con la salida; es decir, tengan valores esperados [29]. La generación de las capas tienen dos métodos como forward, utilizado para generar un resultado en cada capa acorde al valor anterior que tiene y backward para obtener los valores de los errores en la gradiente en cada capa respecto a los pesos de cada capa. Esto para hacer el proceso de "back propagation". La utilidad de ello es para la actualización de los pesos y continuar con el modelo del entrenamiento.

En tercer lugar se incluyó la clase Dense para conectar cada una de nuestras capas, también conocida como capa completamente conectada. Su función principal es realizar una transformación lineal de la entrada mediante multiplicación de matrices y suma de un bias, antes de aplicar la función de activación. En esta capa cada nodo está conectado a todos los nodos de la capa anterior [28]. Este tipo de capa, es utilizada para que en cada uno de los datos de salida, su cálculo sea dado como resultado de la suma de todos los nodos de las capas anteriores por sus respectivos pesos más un sesgo. De esta manera en cada parte del entrenamiento los pesos modificados influirán en cada de uno de los resultados. La función de activación «f» envuelve el producto escalar entre la entrada de la capa y su matriz de ponderaciones. Tenga en cuenta que las columnas de la matriz de ponderaciones tendrían valores diferentes y se optimizarían durante el entrenamiento del modelo [25].
En tercer lugar, la clase Dense implementa una capa totalmente conectada, uno de los elementos fundamentales en redes neuronales artificiales.

Dentro del proyecto, esta clase se encarga de transformar los datos que representan los patrones visuales del videojuego Pop’s Music en representaciones intermedias que pueden ser interpretadas por las siguientes capas. Es decir, permite que la red entienda las combinaciones de notas a través del aprendizaje de pesos y sesgos. De este modo, la clase contiene los siguientes elementos clave: 

- `W` y `b`: representan los pesos y sesgos de la capa. Son inicializados con funciones externas y actualizados durante el entrenamiento. 
- `dW` y `db`: almacenan las gradientes de los pesos y sesgos, calculados durante la retropropagación. 
- `forward()`: recibe una entrada `x`, calcula `xWᵗ + b` y devuelve el resultado. Esta operación transforma la entrada para que sea procesada por la siguiente capa.
- `backward()`: calcula los gradientes (`dW`, `db`) usando la derivada de la pérdida con respecto a la salida de esta capa (`dZ`). Luego, devuelve el gradiente respecto a la entrada para continuar la retropropagación.
- `update_params()`: utiliza un optimizador (`SGD`, `Adam`, etc.) para ajustar los pesos `W` y sesgos `b` en función de los gradientes.

De esta manera, la red neuronal puede aprender una representación abstracta de las imágenes de entrada y adaptarse a patrones complejos en los datos generados por el entorno de Pop’n Music. 

En cuarto lugar, las funciones de activación son fundamentales para introducir no linealidad en el modelo, lo que permite que la red aprenda patrones complejos y represente relaciones no triviales en los datos.

En el marco del proyecto, estas funciones permiten que la red reaccione de manera diferenciada según las combinaciones de notas detectadas. La activación correcta en cada capa mejora significativamente la capacidad de clasificación del modelo. Para este proceso, se usa cuatro clases:

- **ReLU (Rectified Linear Unit)**: evalúa cada dato de entrada y lo deja igual si es mayor que 0, o lo convierte en 0 si es negativo. Es ideal para capas ocultas debido a su eficiencia computacional y porque ayuda a mitigar el problema del gradiente desvanecido, un fenómeno que ocurre cuando los gradientes utilizados para actualizar los pesos se vuelven tan pequeños que impiden el aprendizaje efectivo en redes profundas [28]. Este problema puede detener por completo el entrenamiento, especialmente en modelos con muchas capas. El uso de ReLU permite que los gradientes se mantengan estables, acelerando el proceso de aprendizaje y evitando que se “apague” la red.

- **Sigmoid**: transforma cada valor en una probabilidad entre 0 o 1, lo cual es útil para tareas de clasificación binaria o como activación en capas de salida con decisiones dicotómicas.

- **Softmax**: convierte los valores de salida en una distribución de probabilidad sobre múltiples clases. Es utilizada al final del modelo, donde cada patrón visual puede pertenecer a una clase distinta.

Cada una de estas funciones de activación hereda de `ILayer<T>` e implementa los métodos `forward()` y `backward()`


En quinto lugar, la clase Neural Network actúa como el núcleo de aprendizaje profundo, ya que permite organizar, entrenar, evaluar y guardar toda la arquitectura construida con capas. En el marco del proyecto, esta clase coordina el aprendizaje de patrones visuales complejos, reconociendo combinaciones específicas de notas que definen el input del jugador. A continuación se presentan las funcionalidades que se requieren para este aprendizaje: 

- `add_layer()`: permite añadir nuevas capas de manera secuencial al modelo. Esto permite construir arquitecturas flexibles que combinan transformaciones lineales (`Dense`) y no lineales (`ReLU`, `Softmax`), necesarias para el aprendizaje jerárquico de los datos.
- `predict()`: realiza la propagación hacia adelante (*forward propagation*) de una entrada a través de todas las capas del modelo, produciendo una predicción final. En el contexto del juego, esto se traduce en una decisión del modelo respecto a qué clase pertenece una determinada imagen del patrón musical.
- `train()`: entrena la red usando un algoritmo de retropropagación con descenso del gradiente. En cada época:
  - Divide los datos en mini-lotes (`batch_size`).
  - Propaga cada mini-lote con `predict()`.
  - Calcula el error usando una función de pérdida (`LossType`).
  - Retropropaga el gradiente con `backward()` y actualiza los parámetros con un optimizador (`OptimizerType`, por defecto `SGD`).
  - Mide la pérdida y la precisión (*accuracy*) tras cada época.
- `save()` y `load()`: permiten guardar y restaurar el estado del modelo entrenado. Esto resulta fundamental para no tener que reentrenar el modelo desde cero al reiniciar el juego, y poder cargar una red ya entrenada que reconozca los patrones.

Gracias a esta clase, el videojuego puede aprender a partir de los datos de entrenamiento y generalizar el reconocimiento de nuevas combinaciones de notas. Esto permite construir una experiencia interactiva más robusta, donde el modelo comprende patrones sin necesidad de ser programado explícitamente para cada combinación. 

En quinto lugar la clase Loss define las funciones de pérdida utilizadas durante el entrenamiento de la red neuronal. Estas funciones son importantes debido a que permiten calcular la diferencia entre las predicciones generadas por el modelo y los valores esperados. En el contexto del presente proyecto, que busca entrenar una red neuronal capaz de reconocer patrones visuales del videojuego Pop’n Music, la presente clase cumple un rol fundamental en el aprendizaje automático. Así, este módulo define las funciones de pérdida que permiten cuantificar que tan acertada es la predicción del modelo al clasificar una imagen generada con combinaciones de notas respecto a su etiqueta real. 
Durante el entrenamiento de la red neuronal, al mostrarle una imagen que representa una secuencia del juego, se compara la salida predicha con la etiqueta correcta. La función de pérdida mide esta diferencia, devolviendo un valor numérico que indica cuán lejos estuvo la predicción del resultado esperado. En este caso, la clase incluye tres implementaciones especializadas: MSELoss, útil para tareas de regresión; BCELoss, aplicada cuando se trata de decisiones binarias; CrossEntropyLoss, más utilizada debido a que el modelo realiza una clasificación multiclase entre los diferentes tipos de combinaciones posibles en las imágenes del juego. 
De este modo, cada una de estas funciones de pérdida implementa dos métodos clave: loss(), que devuelve el valor de error, y loss_gradient(), que permite retropropagar dicho error para ajustar los pesos de la red. Gracias a esta retroalimentación, el modelo mejora su precisión en la clasificación de nuevos patrones generados a partir del entorno visual de Pop’n Music. 

En sexto lugar, la clase Optimización se utiliza para implementar los algoritmos de optimización que se encargan de actualizar los pesos de la red neuronal durante el proceso de entrenamiento. Para el presente proyecto, estos algoritmos permiten que la red aprenda a clasificar correctamente los patrones al minimizar el valor de la función pérdida. El archivo define dos optimizadores fundamentales: 
- `SGD` (Stochastic Gradient Descent): se trata de un optimizador clásico que ajusta cada peso restando una fracción proporcional a su gradiente. Su simplicidad lo hace rápido, pero puede tener dificultades en converger cuando el espacio de parámetros es irregular o tiene muchos mínimos locales. 
- `Adam` (Adaptive Moment Estimation): es un optimizador más sofisticado que combina el enfoque de momento y adaptación de tasa de aprendizaje. Almacena promedios móviles de los gradientes (primer momento m) y de sus cuadrados (segundo momento v) y los corrige en cada paso de actualización.  Esto permite una convergencia más rápida y estable, especialmente útil en entornos como el nuestro donde los datos visuales pueden ser ruidosos o variados.

Ambas clases implementan el método update(), que recibe los parámetros actuales y sus respectivos gradientes, y luego los actualiza con base en la lógica del optimizador correspondiente. 

En séptimo lugar se implementó un archivo denominado “interfaces” que define las interfaces principales que sirven como base para el diseño modular de toda la red neuronal implementada. Estas interfaces permiten separar responsabilidades entre capas, funciones de pérdida y optimizadores, facilitando que cada componente sea reutilizable e intercambiable en distintos contextos. 
La interfaz ILayer<T> es implementada por todas las capas del modelo, incluyendo las densas (Dense) y las de activación (ReLU, Sigmoid, Softmax). Esta interfaz obliga a definir dos funciones principales:
- `forward()`: se encarga de propagar los datos de entrada hacia adelante, capa por capa.
- `backward()`: se utiliza en la retropropagación del error para ajustar los pesos en función de los gradientes.
- `update_params()`: método opcional que permite actualizar los parámetros entrenables como pesos y sesgos cuando corresponde.
Por otro lado, la interfaz ILoss<T, DIMS> define el comportamiento de las funciones de pérdida como CrossEntropyLoss. Estas funciones permiten medir cuán lejos están las predicciones del modelo respecto a las etiquetas reales. Esta interfaz exige definir el método loss(), que calcula el error de predicción, y loss_gradient(), que genera el gradiente necesario para que la red aprenda durante el entrenamiento.
Finalmente, la interfaz IOptimizer<T> es implementada por optimizadores como SGD o Adam, y se encarga de actualizar los pesos y sesgos del modelo según los gradientes calculados. En nuestro caso, permite que la red neuronal aprenda a clasificar correctamente los patrones visuales asociados a combinaciones musicales, ajustando los parámetros después de cada retropropagación.

Por último, la clase NeuralNetwork representa el modelo principal de red neuronal que orquesta el flujo completo de datos a través de todas las capas definidas y permite entrenar, predecir y guardar/cargar el modelo. Así, actúa como el núcleo funcional del sistema inteligente. Sus principales funcionales se divide en las siguientes partes: 
- `add_layer`: permite ir agregando las capas (como Dense, ReLU, Softmax, etc.) de forma modular. Así, se puede diseñar la arquitectura del modelo de manera flexible y personalizada para nuestro juego.
- `predict`: ejecuta una propagación hacia adelante (forward) pasando la entrada por todas las capas agregadas. Esto es usado tanto durante el entrenamiento como durante la etapa de inferencia, cuando se desea predecir la clase correspondiente a una nueva entrada musical.
- `train`: realiza el proceso de entrenamiento mediante:
  - La generación de batches con los datos de entrada y etiquetas.
  - El cálculo de la predicción usando predict.
  - La evaluación del error con una función de pérdida (como CrossEntropyLoss).
  - La retropropagación del gradiente con backward
La actualización de pesos con el optimizador (SGD por defecto).
Este entrenamiento se realiza por varias épocas, lo que permite que el modelo aprenda progresivamente a clasificar correctamente las combinaciones musicales.
- `save/load`: implementa funciones para guardar y cargar los pesos del modelo entrenado. Así se evita tener que entrenarlo cada vez, y se puede reutilizar en futuras sesiones del juego.

Es así como gracias a esta clase,todo el sistema de entrenamiento y evaluación está encapsulado en un solo objeto, lo que permite entrenar y reutilizar modelos completos de manera eficiente. En el contexto del proyecto, esta clase facilita el reconocimiento preciso de patrones musicales, mejorando progresivamente su capacidad predictiva con cada época de entrenamiento.


### 2. Diseño e implementación

#### 2.1 Arquitectura de la solución

* **Patrones de diseño**: ejemplo: Factory para capas, Strategy para optimizadores.
* **Estructura de carpetas (ejemplo)**:

  ```
  proyecto-final/
  ├── generator/	              #Generador de imágenes de entrada tipo juego rítmico
  |   ├── assets/              #Recursos visuales: pop-kuns y plantillas
  |   |	├── kuns/
  |   |	|   ├── blue.png
  |   |	|   ├── green.png
  |   |	|   ├── red.png
  |   |	|   ├── white.png
  |   |	|   └── yellow.png
  |   |	├── halo.png
  |   |	├── halo_smaller.png
  |   |	├── empty_template.png
  |   |	├── empty_template_smaller.png
  |   |   ├── template.png
  |   |   └── template_smaller.png
  |   └── main.cpp             #Código para la generación visual
  ├── include/                 #Headers del proyecto
  │   ├── utec/
  │   │   ├── algebra/         #Implementación del tensor personalizado
  │   │   │   └── tensor.h
  │   │   ├── nn/              #Módulos de red neuronal
  │   │   │   ├── neural_network.h
  │   │   │   ├── nn_activation.h
  │   │   │   ├── nn_dense.h
  │   │   │   ├── nn_interfaces.h
  │   │   │   ├── nn_loss.h
  │   │   │   └── nn_optimizer.h
  ├── src/                     #Entrada principal del programa entrenable
  │   └── main.cpp
  ├── tests/                   #Evaluadores por lote, juego y por imagen
  │   ├── batch_evaluator/
  │   │   └── src/main.cpp
  │   ├── gameplay_evaluator/
  │   │   └── src/main.cpp
  │   └── image_evaluator/
  └── │   └── src/main.cpp
  ```

#### 2.2 Manual de uso y casos de prueba

* **Cómo ejecutar**: `./build/neural_net_demo input.csv output.csv`
* **Casos de prueba**:

  * Test unitario de capa densa.
  * Test de función de activación ReLU.
  * Test de convergencia en dataset de ejemplo.

> *Personalizar rutas, comandos y casos reales.*

---

### 3. Ejecución

> **Demo de ejemplo**: Video/demo alojado en `docs/demo.mp4`.
> Pasos:
>
> 1. Preparar datos de entrenamiento (formato CSV).
> 2. Ejecutar comando de entrenamiento.
> 3. Evaluar resultados con script de validación.

---

### 4. Análisis del rendimiento

* **Métricas de ejemplo**:

  * Iteraciones: 150 épocas.
  * Tiempo total de entrenamiento: 1h03m15s.
  * Precisión final: 92.5%.
* **Ventajas/Desventajas**:
  * * Código efectivo para el entrenamiento.
  * * Uso adecuado de la función de perdida.
  * * Uso de hilos que facilitan los calculos para el entreamiento.

  * – Sin optimizaciones numéricas.
* **Mejoras futuras**:

  * Incorporar el uso de BLAS (Basic Linear Algebra Subprograms) para operaciones matriciales, como la multiplicación de matrices en capas densas.
  * Incorporar algoritmos de optimización más avanzados, como Gradiente Descendente con Momentum. La razón de esto es que el optimizador actual (SGD) actualiza los pesos solo con base en el gradiente actual, lo que puede generar oscilaciones en regiones de la función de pérdida con mucha curvatura o valles largos. Y el uso de momentum agrega una fracción del gradiente anterior al gradiente actual, permitiendo avanzar más rápido en direcciones consistentes y amortiguar oscilaciones.

---

## 5. Trabajo en equipo

| **Tarea**                    | **Miembro**                             | **Rol**                         |
|-----------------------------|-----------------------------------------|---------------------------------|
| Investigación teórica       | Matos Copello Rayhan Derek              | Documentar bases teóricas       |
| Investigación teórica       | Tamara Ureta, Anyeli Azumi              | Documentar bases teóricas       |
| Diseño e implementación     | Alvarado León, Adriana Celeste          | UML y esquemas de clases        |
| Implementación del modelo   | Mattos Gutierrez, Angel Daniel          | Código C++ de la NN             |
| Análisis y rendimiento      | Aquino Castro Farid Jack                | Generación de métricas          |
| Documentación y demo        | Portugal Vilca Julio Cesar              | Tutorial y video demo           |

---

### 6. Conclusiones

* **Logros:**  Se llevó a cabo con éxito una red neuronal desde el inicio en C++, incluyendo elementos como tensores, funciones de activación, pérdida y optimización, y se comprobó su rendimiento en un dataset sintético creado automáticamente.

* **Resultados:**  El modelo logró una exactitud que superó el 99% en validaciones por lotes y demostró habilidad para generalizar al implementarse en situaciones reales en el videojuego Pop’n Music.

 * **Aprendizaje:**  Se profundizó en aspectos fundamentales del aprendizaje profundo como la retropropagación, la normalización de entradas, la inicialización de pesos y la modificación de hiperparámetros en ambientes de nivel bajo sin frameworks externos.

 * **Sugerencias:**  Para futuras mejoras, se recomienda expandirse a datasets más complejos y diversos, incorporar entrenamiento con GPU (CUDA) para disminuir los tiempos computacionales, y aplicar métodos adicionales como la normalización y normalización en grupo.

---

## 7. Referencias Bibliográficas

[1] J. Schmidhuber, "Deep Learning in Neural Networks: An Overview," Neural Networks, vol. 61, pp. 85–117, 2015.  
[2] R. Zhang, W. Li, and T. Mo, "Review of Deep Learning," arXiv preprint arXiv:1804.01653, 2018.  
[3] I. Goodfellow, Y. Bengio, and A. Courville, *Deep Learning*. MIT Press, 2016.  
[4] R. Szeliski, *Computer Vision: Algorithms and Applications*. Springer, 2010.  
[5] The Royal Society and The Alan Turing Institute, "Synthetic Data: What, Why and How?," The Royal Society, 2021.  
[6] Y. Lu, M. Shen, H. Wang, et al., "Machine Learning for Synthetic Data Generation: A Review," arXiv preprint arXiv:2305.15799, 2023.  
[7] A. Bauer, S. Trapp, M. Stenger, et al., "Comprehensive Exploration of Synthetic Data Generation: A Survey," arXiv preprint arXiv:2401.03067, 2024.  
[8] M. Giuffrè and D. L. Shung, "Harnessing the Power of Synthetic Data in Healthcare: Innovation, Application, and Privacy," *npj Digital Medicine*, vol. 6, no. 186, 2023. doi: 10.1038/s41746-023-00927-3.  
[9] K. Fukushima, "Visual feature extraction by a multilayered network of analog threshold elements," *IEEE Trans. Systems Science and Cybernetics*, vol. 5, no. 4, pp. 322–333, 1969.  
[10] K. Jarrett, K. Kavukcuoglu, M. A. Ranzato, and Y. LeCun, "What is the Best Multi-Stage Architecture for Object Recognition?," in *2009 IEEE 12th International Conference on Computer Vision*, 2009, pp. 2146–2153. doi: 10.1109/ICCV.2009.5459469.  
[11] X. Glorot, A. Bordes, and Y. Bengio, "Deep sparse rectifier neural networks," in *Proc. 14th Int. Conf. Artif. Intell. Statist. (AISTATS)*, 2011, pp. 315–323.  
[12] L. Boltzmann, "Studien über das Gleichgewicht der lebendigen Kraft zwischen bewegten materiellen Punkten," *Wiener Berichte*, vol. 58, pp. 517–560, 1868.  
[13] J. W. Gibbs, *Elementary Principles in Statistical Mechanics*. Yale University Press, 1902.  
[14] V. Shatravin, D. Shashev, and S. Shidlovskiy, "Implementation of the SoftMax Activation for Reconfigurable Neural Network Hardware Accelerators," *Applied Sciences*, vol. 13, no. 23, 2023. doi: 10.3390/app132312784.  
[15] S. Ruder, "An overview of gradient descent optimization algorithms," arXiv preprint arXiv:1609.04747, 2017.  
[16] L. Bottou, "Large-Scale Machine Learning with Stochastic Gradient Descent," in *Proceedings of COMPSTAT'2010*, Springer, 2010, pp. 177–186. doi: 10.1007/978-3-7908-2604-3_16.  
[17] D. P. Kingma and J. Ba, "Adam: A method for stochastic optimization," arXiv preprint arXiv:1412.6980, 2014.  
[18] A. C. Wilson, R. Roelofs, M. Stern, N. Srebro, and B. Recht, "The marginal value of adaptive gradient methods in machine learning," in *Advances in Neural Information Processing Systems (NeurIPS)*, 2017.  
[19] J. Brownlee, “Rectified Linear Activation Function for Deep Learning Neural Networks,” *Machine Learning Mastery*, 2019. [Online]. Available: https://machinelearningmastery.com/rectified-linear-activation-function-for-deep-learning-neural-networks/  
[20] DataCamp, “What is ReLU? The Rectified Linear Unit Activation Function,” *DataCamp Blog*, 2023. [Online]. Available: https://www.datacamp.com/blog/rectified-linear-unit-relu  
[21] OldGameShelf, “Pop’n Music GB (GBC),” *OldGameShelf.com*, [Online]. Available: https://oldgameshelf.com/es/games/gbc/pop%27n-music-gb-gbc-6719  
[22] Deepchecks, “Rectified Linear Unit (ReLU),” *Deepchecks Glossary*, 2023. [Online]. Available: https://www.deepchecks.com/glossary/rectified-linear-unit-relu/  
[23] PyTorch, “Reducing storage footprint and bandwidth usage for distributed checkpoints with PyTorch DCP,” *PyTorch Blog*, 2023. [Online]. Available: https://pytorch.org/blog/reducing-storage-footprint-and-bandwidth-usage-for-distributed-checkpoints-with-pytorch-dcp/  
[24] Analytics Vidhya, “Introduction to Softmax for Neural Network,” *Analytics Vidhya*, 2021. [Online]. Available: https://www.analyticsvidhya.com/blog/2021/04/introduction-to-softmax-for-neural-network/  
[25] Built In, “What Is a Fully Connected Layer?,” *builtin.com*, [Online]. Available: https://builtin.com/machine-learning/fully-connected-layer  
[26] Nerdjock, “Deep Learning Course Lesson 5: Forward and Backward Propagation,” *Medium*, 2020. [Online]. Available: https://medium.com/@nerdjock/deep-learning-course-lesson-5-forward-and-backward-propagation-ec8e4e6a8b92  
[27] KeepCoding, “Forward y Back Propagation en Deep Learning,” *KeepCoding.io*, [Online]. Available: https://keepcoding.io/blog/forward-back-propagation-deep-learning/  
[28] Universidad de Guadalajara, “Capas de una red neuronal,” *CUCSur UDGVirtual*, 2024. [Online]. Available: http://cucsur.udgvirtual.udg.mx/oa/2024/RedesNeu/capas.html
[29] IBM, “Neural network model,” IBM Documentation, [Online]. Available: https://www.ibm.com/docs/en/spss-modeler/saas?topic=networks-neural-model. [Accessed: Jul. 10, 2025].

### Licencia

Este proyecto usa la licencia **MIT**. Ver [LICENSE](LICENSE) para detalles.

---
