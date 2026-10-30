# Taller de redes neuronales

Taller de Inteligencia Artificial (UNINTEP, docente Oscar F. Pulgarin). Tiene dos partes: un análisis escrito sobre qué arquitectura conviene para predecir el caudal del río Pescador, y una implementación de Multi-Head Attention en PyTorch, hecha desde cero y sin usar `torch.nn.MultiheadAttention`.

## Qué hay en esta carpeta

- `taller_redes_neuronales.md`: la guía y el enunciado del taller.
- `Flujo_Caudal_Rio_Pescador.md`: respuesta al punto 1. Elijo una GRU para modelar el caudal (lecturas cada 10 minutos durante 3 años) y explico por qué dejo por fuera la LSTM y el Transformer, mirando los datos, el costo de cómputo y la latencia.
- `Multihead_Atention.py`: respuesta al punto 2. La atención de cabezas múltiples está armada solo con `nn.Linear`, multiplicación de matrices y softmax, siguiendo las secciones 3.2 y 3.3 de la guía.
- `resultados.txt`: lo que imprime el script al ejecutarlo con un tensor sintético.
- `pyproject.toml` y `uv.lock`: dependencias del entorno de `uv`.

## Cómo correrlo

```bash
uv run Multihead_Atention.py
```

El script genera un tensor aleatorio de forma `[4, 10, 64]`, lo pasa por el módulo y comprueba que la salida conserve esa misma forma. Si todo va bien, se ve así:

```
Entrada: [4, 10, 64]
Salida:  [4, 10, 64]
Dimensiones conservadas: True
```

## Una aclaración

El análisis del punto 1 se apoya en la guía del curso y en el material de clase (entre otros, el caso de Chicago). El tamaño de la ventana y los tiempos de inferencia no los medí; habría que confirmarlos con datos reales del sensor.
