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
  * Portugal Vilca Julio Cesar E – 202410487 (Documentación y demo)

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

> *Ejemplo de repositorio y comandos, ajustar según proyecto.*

---

### 1. Investigación teórica

* **Objetivo**: Explorar fundamentos y arquitecturas de redes neuronales.
* **Contenido de ejemplo**:

  1. Historia y evolución de las NNs.
  2. Principales arquitecturas: MLP, CNN, RNN.
  3. Algoritmos de entrenamiento: backpropagation, optimizadores.

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

[1] J. Schmidhuber, "Deep Learning in Neural Networks: An Overview," Neural Networks, vol. 61, pp. 85–117, 2015.
[2] R. Zhang, W. Li, and T. Mo, "Review of Deep Learning," arXiv preprint arXiv:1804.01653, 2018.
[3] I. Goodfellow, Y. Bengio, and A. Courville, Deep Learning. MIT Press, 2016.
[4] R. Szeliski, Computer Vision: Algorithms and Applications. Springer, 2010.
[5] The Royal Society and The Alan Turing Institute, "Synthetic Data: What, Why and How?," The Royal Society, 2021.
[6] Y. Lu, M. Shen, H. Wang, et al., "Machine Learning for Synthetic Data Generation: A Review," arXiv preprint arXiv:2305.15799, 2023.
[7] A. Bauer, S. Trapp, M. Stenger, et al., "Comprehensive Exploration of Synthetic Data Generation: A Survey," arXiv preprint arXiv:2401.03067, 2024.
[8] M. Giuffrè and D. L. Shung, "Harnessing the Power of Synthetic Data in Healthcare: Innovation, Application, and Privacy," npj Digital Medicine, vol. 6, no. 186, 2023. doi: 10.1038/s41746-023-00927-3.
[9] K. Fukushima, "Visual feature extraction by a multilayered network of analog threshold elements," IEEE Trans. Systems Science and Cybernetics, vol. 5, no. 4, pp. 322–333, 1969.
[10] K. Jarrett, K. Kavukcuoglu, M. A. Ranzato, and Y. LeCun, "What is the Best Multi-Stage Architecture for Object Recognition?," in 2009 IEEE 12th International Conference on Computer Vision, 2009, pp. 2146–2153. doi: 10.1109/ICCV.2009.5459469.
[11] X. Glorot, A. Bordes, and Y. Bengio, "Deep sparse rectifier neural networks," in Proc. 14th Int. Conf. Artif. Intell. Statist. (AISTATS), 2011, pp. 315–323.
[12] L. Boltzmann, "Studien über das Gleichgewicht der lebendigen Kraft zwischen bewegten materiellen Punkten," Wiener Berichte, vol. 58, pp. 517–560, 1868.
[13] J. W. Gibbs, Elementary Principles in Statistical Mechanics. Yale University Press, 1902.
[14] V. Shatravin, D. Shashev, and S. Shidlovskiy, "Implementation of the SoftMax Activation for Reconfigurable Neural Network Hardware Accelerators," Applied Sciences, vol. 13, no. 23, 2023. doi: 10.3390/app132312784.
[15] S. Ruder, "An overview of gradient descent optimization algorithms," arXiv preprint arXiv:1609.04747, 2017.
[16] L. Bottou, "Large-Scale Machine Learning with Stochastic Gradient Descent," in Proceedings of COMPSTAT'2010, Springer, 2010, pp. 177–186. doi: 10.1007/978-3-7908-2604-3_16.
[17] D. P. Kingma and J. Ba, "Adam: A method for stochastic optimization," arXiv preprint arXiv:1412.6980, 2014.
[18] A. C. Wilson, R. Roelofs, M. Stern, N. Srebro, and B. Recht, "The marginal value of adaptive gradient methods in machine learning," in Advances in Neural Information Processing Systems (NeurIPS), 2017.


---

### Licencia

Este proyecto usa la licencia **MIT**. Ver [LICENSE](LICENSE) para detalles.

---
