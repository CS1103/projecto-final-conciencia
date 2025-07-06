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
  * Alvarado León, Adriana Celeste B – 209900002 (Diseño e implementación)
  * Mattos Gutierrez, Angel Daniel C – 209900003 (Implementación del modelo)
  * Aquino Castro Farid Jack D – 202410569 (Análisis y Rendimiento)
  * Portugal Vilca Julio Cesar E – 209900005 (Documentación y demo)

> *Nota: Reemplazar nombres y roles reales.*

---

### Requisitos e instalación

1. **Compilador**: GCC 11 o superior
2. **Dependencias**:

   * CMake 3.18+
   * Eigen 3.4
   * \[Otra librería opcional]
3. **Instalación**:

   ```bash
   git clone https://github.com/EJEMPLO/proyecto-final.git
   cd proyecto-final
   mkdir build && cd build
   cmake ..
   make
   ```
---

### 1. Investigación teórica

* **Objetivo**: Explorar fundamentos y arquitecturas de redes neuronales.

Para poder diseñar la red neuronal se incluyeron las siguientes clases para poder simular todo el comportamiento de la red neuronal: Tensor, Layer, Dense, ReLU, Loss y por último la clase Neural_Network donde se alojarán todas nuestras funciones. A continuación se detallará el porqué de su inclusión.

En primer lugar, se implementó la clase Tensor para manejar toda la información que ingresa a la red neuronal, es decir los datos con los que uno se entrena y se predice. El motivo por el cual se escogieron los tensores es para poder trabajar con grandes volúmenes de información de forma eficiente, los cuales se componen principalmente de datos binarios. Es por ello que para desarrollar esta clase tomamos como referencia la biblioteca Pytorch. A medida que los modelos aumentan en complejidad y tamaño, los puntos de control distribuidos se convierten en un componente crucial del proceso de entrenamiento. No obstante estos suelen generar importantes demandas de almacenamiento. Para abordar este desafío, la comprensión surge como una solución natural. Dado que los puntos de control se componen principalmente de datos binarios (tensores). (Pytorch, 2025).
Además como trabajamos con tensores, es esencial el desarrollo de métodos de operaciones con matrices como por ejemplo la función de transposición, multiplicación escalar, tensor broadcasting, sumas y resta, métodos de acceso. Cada uno de ellos escenciales para realizar el cálculo de los pesos y valores de salida en cada capa de la red neuronal, al igual que sus respectivos sesgos.

Las clases layers son las capas que conectadas entre sí, hacen posible que funcione el proceso de entrenamiento. Hay tres partes donde se presentan estas capas: Una capa de entrada, donde se ingresa los valores iniciales del entrenamiento; las capas ocultas, donde se realiza todo el proceso de cálculo y ponderación; por último la capa de salida con los valores de destino que se busca en el entrenamiento. En el entrenamiento se busca que cada valor inicial que ingrese, se procese dentro de las capas ocultas con ponderaciones aleatorias que se van actualizando gradualmente para acercarlas a la capa de destino. Este proceso sigue su curso hasta un punto en que varias capas coincidan con la salida; es decir, tengan valores esperados (IBM, 2021).
La generación de las capas tienen dos métodos como forward, utilizado para generar un resultado en cada capa acorde al valor anterior que tiene y backward para obtener los valores de los errores en la gradiente en cada capa respecto a los pesos de cada capa. Esto para hacer el proceso de "back propagation". La utilidad de ello es para la actualización de los pesos y continuar con el modelo del entrenamiento.

Se implementó la clase ReLU como función de activación para el proyecto, su objetivo es evaluar cada dato de entrada y comparar si este dato es mayor o igual a 0 o menor. Si es mayor a 0, la función deja tal cual el valor de entrada. Caso contrario, si el valor es un número negativo, lo convierte a 0. Este método es uno de los más utilizados hoy en día para formar redes neuronales más profundas debido a que reduce en gran medida el back propagation. Asimismo, ayuda a mitigar el problema del gradiente evanescente, ocasionado porque la gradiente usada para actualizar nuestros valores en cada época tiene un valor muy pequeño, el cual retrasa el entrenamiento y puede llegar a detenerlo (Canales, 2025).
En el proyecto al usar esta versión ReLU donde los valores negativos ya no existen se pueden activar algunas neuronas en el proceso de entrenamiento las cuales ayudan a un cálculo mucho más eficiente. Esto resulta aún más útil cuando en el proyecto consideramos los métodos que usamos como Sigmoid y Softmax, una para convertir el valor de entrada en uno dentro del rango de 0 y 1 y la otra para el cálculo de las probabilidades de que los valores insertados pertenezcan a una clase u otra, donde el problema del desvanecimiente es más común. 
La inclusión de este tipo de función nos permite la generación de una mejor capa densa para la realización del entrenamiento pues se puede escalar a múltiples capas sin hacer sobrecarga computacional.



---

### 2. Diseño e implementación

#### 2.1 Arquitectura de la solución

* **Patrones de diseño**: ejemplo: Factory para capas, Strategy para optimizadores.
* **Estructura de carpetas (ejemplo)**:

  ```
  proyecto-final/
  ├── src/
  │   ├── layers/
  │   ├── optimizers/
  │   └── main.cpp
  ├── tests/
  └── docs/
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

  * Iteraciones: 1000 épocas.
  * Tiempo total de entrenamiento: 2m30s.
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

> *Actualizar con bibliografia utilizada, al menos 4 referencias bibliograficas y usando formato IEEE de referencias bibliograficas.*

---

### Licencia

Este proyecto usa la licencia **MIT**. Ver [LICENSE](LICENSE) para detalles.

---
