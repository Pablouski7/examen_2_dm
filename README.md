
### Predicción de Cancelación de Clientes

Este proyecto corresponde al examen final de Minería de Datos (USFQ, mayo 2025). Utiliza datos de Interconnect para predecir el abandono de clientes (`churn`) y proponer acciones de retención, como descuentos personalizados.

#### Instrucciones

1. Clona este repositorio.
2. Asegúrate de tener Python instalado
3. Instala las dependencias con:

```bash
pip install -r requirements.txt
```

4. Ejecuta el notebook:

```bash
jupyter notebook notebooks/herrera_pablo_churn.ipynb
```

#### Estructura del proyecto

```
data/           # Archivos CSV originales (no incluidos en el repo)
notebooks/      # Notebook principal del proyecto
models/         # Modelos entrenados (opcional)
requirements.txt
README.md
```

#### Modelos evaluados

* Logistic Regression
* Random Forest
* XGBoost (mejor desempeño)

Incluye técnicas de manejo de desbalance, validación estratificada, interpretación de variables por ganancia y plan de despliegue.

