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
        * ¿Conviene tener a Kz, considerando que se debe restringir a Kz = Kx/10 ó Kz = Kx?, ¿o es preferible delimitar zonas según DGA, 2019?
      * Rendimiento específico (Sy) --> Considerando que Almacenamiento específico (Ss) = Sy / 100
      * Conductancia del río (Cr)
      * Conductancia de elementros DRN (Cd) 
      * Supuesto: No hay variación de los parámentros en el tiempo. Limitación del programa.
    * Definir la función objetivo --> Encontrar el set de parámetros óptimos que reduzca el error entre los valores observados y simulados
        f(K, Sy, Cr, Cd) --> minE{Recarga distribuida, Recarga desde río, Afloramiento, Pozos de observación, Caudales, Recarga lateral (¿?)}
        Subject to:
            Rango de variables por zonas
            Celdas con crecimientos o caídas abruptas.
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

### Convolución
* Máscara por zona de PH

### Operar WEAP - MODFLOW
La calibración se hará por zonas según geología superficial, los valores iniciales serán los presentados en la Tabla 3-3 (DGA, 2019).

1. Generar shapes de 0 y 1s donde se delimiten las zonas de las PH. 
2. Ver como cambiar los valores de conductancia en el río y celdas DRN. También necesitarían shapes de 0 y 1s para no extender la zona.
3. Las variables que forman parte de E:
    WEAP: Pozos de Observación, celdas como comportamientos anómalos.
    MODFLOW: Desde balance --> recarga distribuida, recarga desde el río, afloramientos (En base a investigación DGA, 2019).
4. Periodo de simulación 1980 - 2020 --> Resultados del 85 en adelante.
5. Pasar de valor por zona, a evaluar la anisotropía.
    
### Study area (Podría ir después de explicar Convolución)
* Características semi áridas de las cuencas.
¿Ver en papers que toman casos reales como hacen la comparación de los valores de los PH? - Puede ser si caen dentro del rango de valores de la calibración de la DGA, 2019.

### DUDAS!!
* ¿Cómo se haría la comparación de pozos observados vs simulados, si las fechas de simulación no son las mmismas? ¿Es valido completar los datos de manera lineal?

### Falta hacer en el modelo:
* Agregar streamflow gauges.
* Sacar SHAPES por zonas geológicas rectangulares.
* Cómo unir matriz final después de operación

* Decidir si los valores a acercarnos son los de la DGA. (LEER BIEN SI SUS VALORES SON ACEPTABLES. AUNQUE LA DGA HA COMENTADO QUE LA CALIBRACION LES PARECE BIEN)
* Cuánto demora el modelo en ejecutarse para el periodo de 1980 - 2020. Probar en máquinas de 2, 3 o 4 GB RAM.
