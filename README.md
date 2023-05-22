This repository contains files that allow us to calibrate the MODFLOW model, specifically, estimate groundwater parameters.

# IDEAS GENERALES PAPER - ESTRUCTURA

## Title
* Distributed Bayesian Optimization and Concolutional Layer to Estimate Groundwater Parameters

## Keywords
* Distributed Bayesian Optimization
* BestBetween(Distributed Particle Swarm Optimization ó Distributed Differential Evolution)
* Convolutional Neural Networks (¿?)
* Aquifer hydraulic parameters
* Simulation - Optimization model

## Introducción
* Problema a resolver¨--> Objetivo y vacio en el conocimiento.

    ### Objetivo principal
    * Estimar los parámetros hidráulicos de un acuífero utilizando modelación inversa con bayesian optimization y convolutional layers junto a un modelo integrado de recursos hídricos.
    ### Objetivo específco
    * Optimizar la calibración de un modelo subterráneo utilizando DBO-CL
    * Calibrar paramétros hidráulicos (variables de decisión): 
      * Conductividad hidráulica (Kx y Kz) --> Considerando Ky = Kx
      * Rendimiento específico (Sy) --> Considerando que Almacenamiento específico (Ss) = Sy / 100
      * Supuesto: No hay variación de los parámentros en el tiempo. Limitación del programa.
    * Definir la función objetivo --> Encontrar el set de parámetros óptimos que reduzca el error entre los valores observados y simulados
      * Niveles de agua subterránea en pozos DGA. --> (Euclidean norm (Lingireddy, 1997), Sum of Squared Difference (SSD). Sum of the Root Mean Squared Error (SRMSE) (Patel et al., 2022))
      * No deberían haber celdas aisladas con unicos valores, es decir, si se escoge un valor debe tener una celda adyacente como mínimo con el mismo valor.
      * Establecer el rango de valores en los que se moverán las variables de decisión (Conductividad hidráulica, rendimiento y almacenamiento específico). Estos valores se establecen en función de estudios geológicos en la zona (citar carta geológica de la zona, guía DGA, 2019).
      * MODFLOW Cell Head no crezca en el tiempo.
      * MODFLOW Cell Head no tenga caídas abruptas en primeros años.
      * MODFLOW Cell Head no tenga caídas mayores a más de 90 metros.
    * Metodología aplicada a un acuífero real --> Ligua - Petorca, Chile central.

## Data and  Methodology
### Integrated Managment Water Model --> Modelo WEAP - MODFLOW.
* [Explicación del modelo - Mencionar que este está siendo desarrollado en el proyecto de Ligua Petorca (¿?)]
* Ecuación que resuelve MODFLOW.
* Imagen del modelo

### Modelo de optamización  

#### Convolución
* Máscara por zona de PH

#### Study area (Podría ir después de explicar Convolución)
* Características semi áridas de las cuencas.
