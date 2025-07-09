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
  * Tamara Ureta, Anyeli Azumi – 200900000 (Responsable de investigación teórica)
  * Alvarado León, Adriana Celeste – 209900002 (Diseño e implementación)
  * Mattos Gutierrez, Angel Daniel – 202420199 (Implementación del modelo)
  * Aquino Castro Farid Jack – 202410569 (Análisis y Rendimiento)
  * Portugal Vilca Julio Cesar – 202410487 (Documentación y demo)

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

### Desarrollo de componentes claves:

Para poder diseñar la red neuronal se incluyeron las siguientes clases para poder simular todo el comportamiento de la red neuronal: Tensor, Activation, Layer, Dense, Loss, Optimización y, por último, la clase Neural_Network donde se alojarán todas nuestras funciones. A continuación se detalla su inclusión.

En primer lugar, se implementó la clase Tensor para manejar toda la información que ingresa a la red neuronal, es decir los datos con los que uno se entrena y se predice. El motivo por el cual se escogieron los tensores es para poder trabajar con grandes volúmenes de información de forma eficiente, los cuales se componen principalmente de datos binarios. Es por ello que para desarrollar esta clase tomamos como referencia la biblioteca Pytorch. A medida que los modelos aumentan en complejidad y tamaño, los puntos de control distribuidos se convierten en un componente crucial del proceso de entrenamiento. No obstante estos suelen generar importantes demandas de almacenamiento. Para abordar este desafío, la comprensión surge como una solución natural. Dado que los puntos de control se componen principalmente de datos binarios (tensores). (Pytorch, 2025).

Trabajamos simulando los distintos métodos de operaciones con matrices como por ejemplo la función de transposición, multiplicación escalar, tensor broadcasting, lo cual permitió operar entre tensores de distintas dimensiones respetando reglas similares a las de bibliotecas como Numpy o PyTorch. 

En el marco del proyecto, esto fue crucial para poder representar correctamente tanto las imágenes de entrada generadas a partir de combinaciones de notas del videojuego Pop’n Music, como también los pesos y salidas de cada capa de la red neuronal. Gracias a esta implementación de tensores visuales durante el entrenamiento del modelo, así como aplicar gradientes y operaciones de retropropagación de manera controlada y precisa. 

En segundo lugar en Activation, se agrupan las funciones de activación utilizadas por las distintas capas de la red neuronal durante el entrenamiento y la inferencia. Las funciones de activación son fundamentales para introducir no linealidad en el modelo, lo que permite que la red aprenda patrones complejos y represente relaciones no triviales en los datos. 

En el marco del proyecto, estas funciones permiten que la red reaccione de manera diferenciada según las combinaciones de notas detectadas. La activación correcta en cada capa mejora significativamente la capacidad de clasificación del modelo. Para este proceso, se usa cuatro clases: 

- **ReLU (Rectified Linear Unit)**: evalúa cada dato de entrada y lo deja igual si es mayor que 0, o lo convierte en 0 si es negativo. Es ideal para capas ocultas debido a su eficiencia computacional y porque ayuda a mitigar el problema del gradiente desvanecido, un fenómeno que ocurre cuando los gradientes utilizados para actualizar los pesos se vuelven tan pequeños que impiden el aprendizaje efectivo en redes profundas (Canales, 2025). Este problema puede detener por completo el entrenamiento, especialmente en modelos con muchas capas. El uso de ReLU permite que los gradientes se mantengan estables, acelerando el proceso de aprendizaje y evitando que se “apague” la red.

- **Sigmoid**: transforma cada valor en una probabilidad entre 0 o 1, lo cual es útil para tareas de clasificación binaria o como activación en capas de salida con decisiones dicotómicas. 

- **Softmax**: convierte los valores de salida en una distribución de probabilidad sobre múltiples clases. Es utilizada al final del modelo, donde cada patrón visual puede pertenecer a una clase distinta. 

Cada una de estas funciones de activación hereda de `ILayer<T>` e implementa los métodos `forward()` y `backward()`, que permite aplicar las transformaciones durante la propagación directa y calcular los gradientes respectivos durante la retropropagación. 

En tercer lugar, la clase Dense implementa una capa totalmente conectada, uno de los elementos fundamentales en redes neuronales artificiales. Su función principal es realizar una transformación lineal de la entrada mediante multiplicación de matrices y suma de un bias, antes de aplicar la función de activación. 

Dentro del proyecto, esta clase se encarga de transformar los datos que representan los patrones visuales del videojuego Pop’s Music en representaciones intermedias que pueden ser interpretadas por las siguientes capas. Es decir, permite que la red entienda las combinaciones de notas a través del aprendizaje de pesos y sesgos. De este modo, la clase contiene los siguientes elementos clave: 

- `W` y `b`: representan los pesos y sesgos de la capa. Son inicializados con funciones externas y actualizados durante el entrenamiento. 
- `dW` y `db`: almacenan las gradientes de los pesos y sesgos, calculados durante la retropropagación. 
- `forward()`: recibe una entrada `x`, calcula `xWᵗ + b` y devuelve el resultado. Esta operación transforma la entrada para que sea procesada por la siguiente capa.
- `backward()`: calcula los gradientes (`dW`, `db`) usando la derivada de la pérdida con respecto a la salida de esta capa (`dZ`). Luego, devuelve el gradiente respecto a la entrada para continuar la retropropagación.
- `update_params()`: utiliza un optimizador (`SGD`, `Adam`, etc.) para ajustar los pesos `W` y sesgos `b` en función de los gradientes.

De esta manera, la red neuronal puede aprender una representación abstracta de las imágenes de entrada y adaptarse a patrones complejos en los datos generados por el entorno de Pop’n Music. 

En cuarto lugar, la clase Neural Network actúa como el núcleo de aprendizaje profundo, ya que permite organizar, entrenar, evaluar y guardar toda la arquitectura construida con capas. En el marco del proyecto, esta clase coordina el aprendizaje de patrones visuales complejos, reconociendo combinaciones específicas de notas que definen el input del jugador. A continuación se presentan las funcionalidades que se requieren para este aprendizaje: 

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

  * * Código ligero y dependencias mínimas.
  * – Sin paralelización, rendimiento limitado.
* **Mejoras futuras**:

  * Uso de BLAS para multiplicaciones (Justificación).
  * Paralelizar entrenamiento por lotes (Justificación).

---

### 5. Trabajo en equipo

| Tarea                     | Miembro  | Rol                       |
| ------------------------- | -------- | ------------------------- |
| Investigación teórica     | Alumno A | Documentar bases teóricas |
| Diseño de la arquitectura | Alumno B | UML y esquemas de clases  |
| Implementación del modelo | Alumno C | Código C++ de la NN       |
| Pruebas y benchmarking    | Alumno D | Generación de métricas    |
| Documentación y demo      | Alumno E | Tutorial y video demo     |

> *Actualizar con tareas y nombres reales.*

---

### 6. Conclusiones

* **Logros**: Implementar NN desde cero, validar en dataset de ejemplo.
* **Evaluación**: Calidad y rendimiento adecuados para propósito académico.
* **Aprendizajes**: Profundización en backpropagation y optimización.
* **Recomendaciones**: Escalar a datasets más grandes y optimizar memoria.

---

### 7. Bibliografía

1. J. Schmidhuber, *Deep Learning in Neural Networks: An Overview*, Neural Networks, vol. 61, pp. 85–117, 2015.

2. R. Zhang, W. Li, and T. Mo, *Review of Deep Learning*, arXiv preprint arXiv:1804.01653, 2018. [https://arxiv.org/abs/1804.01653](https://arxiv.org/abs/1804.01653)

3. I. Goodfellow, Y. Bengio, and A. Courville, *Deep Learning*. MIT Press, 2016.

4. R. Szeliski, *Computer Vision: Algorithms and Applications*. Springer, 2010.

5. The Royal Society and The Alan Turing Institute, *Synthetic Data: What, Why and How?*. The Royal Society, 2021.

6. Y. Lu, M. Shen, H. Wang, et al., *Machine Learning for Synthetic Data Generation: A Review*, arXiv preprint arXiv:2305.15799, 2023. [https://arxiv.org/abs/2305.15799](https://arxiv.org/abs/2305.15799)

7. A. Bauer, S. Trapp, M. Stenger, et al., *Comprehensive Exploration of Synthetic Data Generation: A Survey*, arXiv preprint arXiv:2401.03067, 2024. [https://arxiv.org/abs/2401.03067](https://arxiv.org/abs/2401.03067)

8. M. Giuffrè and D. L. Shung, *Harnessing the Power of Synthetic Data in Healthcare: Innovation, Application, and Privacy*, npj Digital Medicine, vol. 6, no. 186, 2023. [https://doi.org/10.1038/s41746-023-00927-3](https://doi.org/10.1038/s41746-023-00927-3)

9. K. Fukushima, *Visual feature extraction by a multilayered network of analog threshold elements*, IEEE Trans. Systems Science and Cybernetics, vol. 5, no. 4, pp. 322–333, 1969.

10. K. Jarrett, K. Kavukcuoglu, M. A. Ranzato, and Y. LeCun, *What is the Best Multi-Stage Architecture for Object Recognition?*, in 2009 IEEE 12th International Conference on Computer Vision, 2009, pp. 2146–2153. [https://doi.org/10.1109/ICCV.2009.5459469](https://doi.org/10.1109/ICCV.2009.5459469)

11. X. Glorot, A. Bordes, and Y. Bengio, *Deep sparse rectifier neural networks*, in Proc. 14th Int. Conf. Artif. Intell. Statist. (AISTATS), 2011, pp. 315–323.

12. L. Boltzmann, *Studien über das Gleichgewicht der lebendigen Kraft zwischen bewegten materiellen Punkten*, Wiener Berichte, vol. 58, pp. 517–560, 1868.

13. J. W. Gibbs, *Elementary Principles in Statistical Mechanics*. Yale University Press, 1902.

14. V. Shatravin, D. Shashev, and S. Shidlovskiy, *Implementation of the SoftMax Activation for Reconfigurable Neural Network Hardware Accelerators*, Applied Sciences, vol. 13, no. 23, 2023. [https://doi.org/10.3390/app132312784](https://doi.org/10.3390/app132312784)

15. S. Ruder, *An overview of gradient descent optimization algorithms*, arXiv preprint arXiv:1609.04747, 2017. [https://arxiv.org/abs/1609.04747](https://arxiv.org/abs/1609.04747)

16. L. Bottou, *Large-Scale Machine Learning with Stochastic Gradient Descent*, in Proceedings of COMPSTAT'2010, Springer, 2010, pp. 177–186. [https://doi.org/10.1007/978-3-7908-2604-3_16](https://doi.org/10.1007/978-3-7908-2604-3_16)

17. D. P. Kingma and J. Ba, *Adam: A method for stochastic optimization*, arXiv preprint arXiv:1412.6980, 2014. [https://arxiv.org/abs/1412.6980](https://arxiv.org/abs/1412.6980)

18. A. C. Wilson, R. Roelofs, M. Stern, N. Srebro, and B. Recht, *The marginal value of adaptive gradient methods in machine learning*, in Advances in Neural Information Processing Systems (NeurIPS), 2017.
---

### Licencia

Este proyecto usa la licencia **MIT**. Ver [LICENSE](LICENSE) para detalles.

---
