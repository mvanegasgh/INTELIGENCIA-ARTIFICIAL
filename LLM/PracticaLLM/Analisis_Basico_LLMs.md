# Análisis del laboratorio: introducción a los LLM

Curso: Inteligencia Artificial, UNINTEP
Autor: Milton Vanegas Delgado

## Objetivos

El código muestra, con tres ejemplos en PyTorch, por qué existen los modelos de lenguaje actuales y cómo se hacen más eficientes. Primero muestra el problema de las redes recurrentes (RNN), luego la solución con atención (Transformer) y por último dos técnicas para ahorrar recursos (FlashAttention y LoRA).

## Ejemplo 1: desvanecimiento de memoria en RNN

Una RNN lee el texto palabra por palabra y guarda lo aprendido en un estado oculto, que hace de memoria. Al entrenar, el error se propaga hacia atrás por todos los pasos y en cada uno el gradiente se multiplica por valores menores que 1. Con secuencias largas se vuelve cada vez más pequeño, hasta casi desaparecer. Es el **desvanecimiento del gradiente**: las primeras palabras casi no influyen en el aprendizaje y la red «olvida» el inicio de las oraciones largas.

El código crea una RNN con una secuencia de 100 pasos, calcula una pérdida que depende solo del último paso y compara el gradiente en el token 0, el 50 y el 99. Al ejecutarlo, el token 99 tuvo un gradiente de 1,38 y los otros dos aparecieron como 0.00000000 (valores tan pequeños que el formato de impresión los redondea a cero). Eso confirma que la información del inicio casi no llega.

## Ejemplo 2: atención en paralelo (Transformer)

La **auto-atención** resuelve el problema anterior. Cada palabra se compara con todas las demás de la secuencia y recibe más información de las que son relevantes para entender su significado. Por ejemplo, en «el banco del río» y «el banco de arena», el contexto indica qué significa cada «banco». La versión **multicabeza** repite este proceso varias veces en paralelo, y cada cabeza puede fijarse en un tipo distinto de relación entre palabras.

La clase `MultiHeadAttention` calcula tres vectores por palabra (consulta, clave y valor), los compara con una multiplicación de matrices, aplica softmax para obtener los pesos de atención y combina los valores. No usa ningún bucle sobre la secuencia: procesa todo a la vez, y eso aprovecha muy bien las GPU. Al ejecutarlo, un lote de 32 textos de 512 palabras entró y salió con la misma forma, `[32, 512, 768]`.

## Ejemplo 3: eficiencia

**FlashAttention.** La atención gasta mucha memoria porque crea una matriz que crece con el cuadrado de la longitud del texto. FlashAttention calcula lo mismo (el resultado es exacto), pero por bloques pequeños que caben en la memoria rápida de la GPU, así que es más veloz y gasta menos memoria. El código solo llama a la función de PyTorch que lo implementa. En mi equipo se ejecutó en CPU, por lo que confirma que la función existe pero no muestra el beneficio real, que requiere una GPU.

**LoRA.** Ajustar un modelo grande para una tarea nueva obliga a entrenar todos sus parámetros, algo muy costoso. LoRA congela el modelo original y entrena solo dos matrices pequeñas que se suman a él, y con eso se logra una adaptación parecida. El código crea la clase `LoRALinear` y compara los parámetros de una capa de 4096 × 4096: 16.781.312 con ajuste completo frente a 65.536 con LoRA (rango 8). Es una reducción del 99,6 % en los parámetros que hay que entrenar, lo que permite ajustar modelos grandes con pocos recursos.

## Conclusión

El laboratorio resume la evolución de los LLM: las RNN olvidaban información en textos largos, la atención lo resolvió procesando todo en paralelo, y FlashAttention y LoRA reducen el costo de usar y adaptar esos modelos.
