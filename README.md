# 🧠 DVT GP Ultra — Entrenamiento y Validación de Ensambles GP (Versión Final)

### 🔬 Descripción General

`train_gp.py` implementa un **pipeline científico completamente automatizado** para el entrenamiento y validación de modelos de *Gaussian Process Regression* aplicados al **potencial inflacionario** dentro del marco de la **Teoría del Vacío Dinámico (DVT)**.

Esta versión final fusionada incorpora:

- ⚙️ **Paralelización avanzada** con `joblib`  
- 🧮 **Optimización automática** de hiperparámetros mediante `Optuna`  
- 🧠 **Validaciones extremas** (multi-escala, simbólicas, adversariales y de extrapolación)  
- 📊 **Visualización científica HQ (1200 DPI)** con estilo `scienceplots`  
- 🧾 **Reporte PDF reproducible y citable**  
- 💻 **Control completo por CLI** (Command Line Interface)

---

## ⚙️ Arquitectura General

El sistema sigue la filosofía **Code → Solve → Learn → Validate → Report**, estructurada modularmente:

| Módulo / Función | Propósito |
|------------------|------------|
| `generate_training_data()` | Genera datos sintéticos del potencial escalar \( V(\phi) = \frac{1}{2}m^2\phi^2 \) con ruido y outliers. |
| `BayesianEnsembleGP` | Núcleo del sistema. Ensamble de modelos GP (RBF, Matern, RQ, Polynomial o personalizados). Combina modelos por verosimilitud marginal. |
| `_safe_kernel_eval()` | Evalúa expresiones de kernel de forma segura (`"ConstantKernel()*RBF()+WhiteKernel()"`). |
| `multi_scale_validation()` | Evalúa convergencia del modelo con distintos tamaños de datos (200, 1000, 5000). |
| `symbolic_validation()` | Compara derivadas numéricas del GP contra derivadas analíticas \( dV/d\phi \) y \( d^2V/d\phi^2 \). |
| `adversarial_validation()` | Evalúa robustez frente a datos ruidosos o fuera de distribución. |
| `extreme_cross_validation()` | Realiza validaciones cruzadas 5×2 con métricas de estabilidad. |
| `extrapolation_test()` | Prueba extrapolación fuera del rango de entrenamiento. |
| `compute_comprehensive_metrics()` | Calcula R², MSE, RMSE, MAE, cobertura IC95 %, y NLL. |
| `create_hq_plots()` | Genera gráficos científicos en PNG y PDF (reconstrucción, residuales, Q-Q, etc). |
| `build_comprehensive_pdf_report()` | Compila automáticamente un reporte PDF con todas las tablas, figuras y conclusiones. |
| `train_pipeline()` | Orquesta el flujo completo: generación, entrenamiento, validación, graficado y reporte. |
| `main()` | Interfaz CLI (`argparse`). Permite ejecutar todo desde terminal. |

---

## 🧩 Requisitos

Instala las dependencias requeridas (Python ≥ 3.9):

```bash
pip install numpy scipy matplotlib seaborn optuna sympy scikit-learn joblib reportlab rich scienceplots
````

Opcional (recomendado para entorno científico):

```bash
pip install tqdm jupyter
```

---

## 🚀 Ejecución CLI

Ejemplo básico:

```bash
python train_gp.py --n-points 2000 --phi-min -5 --phi-max 5 --m-phi 0.5 --noise 0.001 --plot-results --n-jobs 4 --extreme-validation
```

### Argumentos Principales

| Parámetro                | Descripción                                       | Valor por defecto                         |
| ------------------------ | ------------------------------------------------- | ----------------------------------------- |
| `--n-points`             | Nº de puntos sintéticos generados.                | `1000`                                    |
| `--phi-min`, `--phi-max` | Rango del campo escalar φ.                        | `-5`, `5`                                 |
| `--m-phi`                | Masa efectiva del campo.                          | `0.5`                                     |
| `--noise`                | Nivel de ruido gaussiano.                         | `1e-3`                                    |
| `--extend-range-factor`  | Factor de extensión del rango para extrapolación. | `0.2`                                     |
| `--kernel-types`         | Tipos de kernel base (RBF, Matern, etc.).         | `["matern", "rbf", "rational_quadratic"]` |
| `--kernel-expr`          | Expresión de kernel personalizada.                | `None`                                    |
| `--optimize`             | Activa optimización con Optuna.                   | `False`                                   |
| `--n-optuna-trials`      | Nº de pruebas de Optuna por modelo.               | `25`                                      |
| `--n-restarts`           | Reinicios del optimizador GP.                     | `10`                                      |
| `--extreme-validation`   | Activa validaciones multi-escala y adversariales. | `False`                                   |
| `--plot-results`         | Genera gráficos y reporte PDF.                    | `False`                                   |
| `--outdir`               | Carpeta de salida.                                | `dvt_ultra_results`                       |
| `--n-jobs`               | Nº de núcleos CPU.                                | `1`                                       |
| `--verbose`              | Muestra logs detallados.                          | `False`                                   |

---

## 💻 Ejemplo Avanzado

```bash
python train_gp.py \
  --n-points 5000 \
  --phi-min -8 --phi-max 8 \
  --m-phi 0.65 \
  --noise 0.0005 \
  --extend-range-factor 0.3 \
  --kernel-types matern rbf rational_quadratic \
  --optimize --n-optuna-trials 30 \
  --extreme-validation --plot-results \
  --n-jobs 6 --verbose
```

---

## 📊 Resultados Generados

Al finalizar, se crean automáticamente los siguientes archivos en `output/` o `dvt_ultra_results/`:

| Tipo                        | Archivo                                                           | Descripción                                          |
| --------------------------- | ----------------------------------------------------------------- | ---------------------------------------------------- |
| 🧠 **Modelo entrenado**     | `model_YYYYMMDD_HHMMSS.pkl`                                       | Objeto `BayesianEnsembleGP` serializado.             |
| 📈 **Gráficos científicos** | `potential_reconstruction_*.png/pdf`, `curvature_*.png/pdf`, etc. | Visualizaciones HQ (1200 DPI).                       |
| 📄 **Reporte PDF**          | `dvt_gp_report_YYYYMMDD_HHMMSS.pdf`                               | Informe completo con métricas, figuras y análisis.   |
| 🧾 **Log de ejecución**     | `dvt_gp_ultra.log`                                                | Registro detallado del entrenamiento y validaciones. |

---

## 📘 Ejemplo de Métricas Finales

* **Reconstrucción:** RMSE < 0.01 en región central.
* **Validación simbólica:** ( d^2V/d\phi^2 \approx m_\phi^2 ) dentro de ±3 %.
* **Robustez adversarial:** R² > 0.8 con 10 % de outliers.
* **Extrapolación:** Cobertura IC95 % ≈ 0.94.
* **Reporte PDF:** Auto-generado, con tablas y figuras integradas.

---

## 🧩 Comando CLI — Integración Completa con DVT

Cuando se integra en el *pipeline* modular DVT, la simulación bayesiana completa se ejecuta con:

```bash
python -m genesis_modular.run_pipeline \
  --gp "genesis_modular/dvt/dvt_ultra_results/model_20250909_165051.pkl" \
  --walkers 150 --steps 15000 --pool 6 --thin 10 \
  --outdir "results/dvt_run_paper_con_gp"
```

### Desglose del Comando CLI

| Fragmento                                | Propósito                            | Valor Ejemplo                                                       |
| :--------------------------------------- | :----------------------------------- | :------------------------------------------------------------------ |
| `python -m genesis_modular.run_pipeline` | Ejecuta el módulo principal del DVT. | `run_pipeline`                                                      |
| `--gp`                                   | Ruta al modelo GP entrenado.         | `"genesis_modular/dvt/dvt_ultra_results/model_20250909_165051.pkl"` |
| `--walkers`                              | Nº de cadenas MCMC.                  | `150`                                                               |
| `--steps`                                | Iteraciones por cadena.              | `15000`                                                             |
| `--pool`                                 | Núcleos de CPU.                      | `6`                                                                 |
| `--thin`                                 | Factor de adelgazamiento.            | `10`                                                                |
| `--outdir`                               | Carpeta de salida.                   | `"results/dvt_run_paper_con_gp"`                                    |

**En resumen:** este comando lanza la simulación completa de inferencia bayesiana del DVT usando el modelo GP entrenado por `train_gp.py`.

---

## 🧾 Notas Técnicas

* Usa los estilos gráficos `science` y `ieee` para coherencia visual con artículos científicos.
* Los gráficos `.pdf` se generan listos para insertarse con `\includegraphics{}` en LaTeX.
* Compatible con Linux, macOS y WSL2.
* Cada ejecución es reproducible (semillas fijas, logs detallados).

---

## 👨‍🔬 Autoría y Propósito

**Desarrollado por:** *Juan Galaz*
**Proyecto:** *Dynamic Vacuum Toolkit (DVT) – Geometría Causal-Informacional (GCI)*

Enfocado en la reconstrucción falsable del potencial inflacionario usando modelos probabilísticos y metodologías reproducibles de nivel científico.

---

## 📜 Licencia

Este proyecto se distribuye bajo licencia **MIT**, fomentando la ciencia abierta y el uso libre para investigación académica.
