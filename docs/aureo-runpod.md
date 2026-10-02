# FASE 4 — Chatterbox Multilingual en Runpod GPU

Infraestructura preparada para **Chatterbox Multilingual 0.1.7**, mediante el
adaptador AUREO existente. Este Cloud de desarrollo no tiene GPU: no se han
descargado pesos ni ejecutado inferencia real. Los resultados de pruebas con SDK
simulado verifican infraestructura, no calidad de voz o rendimiento GPU.

## Arquitectura y requisitos

- `aureo/gpu/`: un proceso CUDA persistente, IPC JSON por stdin/stdout, caché
  acotada de Voice DNA, exportación WAV, métricas y gateways Pod/Serverless.
- `deploy/aureo/`: instalador, dos requirements con locks completos y hashes,
  contenedor sin modelos y plantilla de variables.
- El gateway usa Runpod 1.12.0/FastAPI; el hijo GPU usa Chatterbox 0.1.7,
  Torch/Torchaudio 2.6.0 (CUDA 12.4 en los wheels Linux). Son entornos distintos:
  Runpod necesita `tomlkit>=0.15.1`, mientras Gradio de Chatterbox exige `<0.14`.
- Objetivo reproducible: Linux x86_64, Python 3.11. El contenedor fija Python
  3.11.16; `uv` está fijado a 0.12.19. Los locks se generan para esta plataforma.
- No se modifican OmniVoice, motores heredados, rutas API existentes, SQLite,
  usuarios, interfaz, administración, versión ni locks de la aplicación.

**GPU recomendada:** NVIDIA RTX 4090 de 24 GB para el primer Pod; L4 de 24 GB
también es una opción conservadora. En Serverless selecciona una clase NVIDIA
de 24 GB disponible en la región del volumen. **VRAM mínima operativa configurada:
8 GiB**; es un suelo de este despliegue, no un mínimo oficial comprobado del
modelo. No garantiza textos largos. Se recomienda 24 GB hasta medir la carga real.
El worker valida CUDA y memoria total del dispositivo, sin fallback automático a
CPU. Usa un host con driver compatible con CUDA 12.4 y al menos 16 GB RAM.

Reserva **50 GB de almacenamiento persistente** para empezar: pesos, cachés,
referencias y resultados; las dependencias del contenedor añaden varios GB.

## Variables exactas

| Variable | Pod | Serverless / significado |
| --- | --- | --- |
| `AUREO_STORAGE_DIR` | `/workspace/aureo` | `/runpod-volume/aureo` |
| `AUREO_MODEL_DIR` | `/workspace/aureo/models/chatterbox` | `/runpod-volume/aureo/models/chatterbox` |
| `AUREO_ENV_DIR` | `/workspace/aureo-env` | `/opt/aureo-env` dentro de la imagen |
| `AUREO_GATEWAY_ENV_DIR` | `/workspace/aureo-env-gateway` | `/opt/aureo-env-gateway` |
| `AUREO_WORKER_PYTHON` | `/workspace/aureo-env/bin/python` | `/opt/aureo-env/bin/python` |
| `AUREO_DEVICE` | `cuda:0` | Un dispositivo CUDA explícito |
| `AUREO_MIN_VRAM_GIB` | `8` | Suelo validado antes de cargar |
| `AUREO_REFERENCE_CACHE_ENTRIES` | `3` | Entradas GPU LRU por contenido + energía; máximo 16 |
| `AUREO_JOB_TIMEOUT_SECONDS` | `600` | Timeout de inicio/operación IPC; mata y recoge el hijo |
| `AUREO_MAX_TEXT_CHARS` | `2000` | Límite de entrada; comenzar con menos de 250 caracteres |
| `AUREO_API_TOKEN` | Token privado aleatorio, ≥16 caracteres | Solo Pod HTTP; Runpod autentica sus jobs con su API key |
| `AUREO_MODEL_REVISION` | Commit HF real de 40 caracteres | Solo para aprovisionar; no usar `main` |
| `HF_TOKEN` | Opcional para descargar del repositorio oficial | No es necesario durante inferencia offline |

El hijo configura `HF_HOME=$AUREO_STORAGE_DIR/hf`, `HF_HUB_CACHE` dentro de este,
`HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1` y telemetría HF deshabilitada. También
bloquea conexiones Python IPv4/IPv6 en carga/síntesis. El SDK Runpod permanece en
el padre, por lo que polling y heartbeats siguen funcionando. Esto no es un
sandbox para código nativo no confiable. La única compatibilidad importada del
backend es su marca de audio, con datos aislados en `watermark-compat/`.

## 1. Primer despliegue: Pod GPU

1. Crea un **Pod NVIDIA RTX 4090 24 GB**, sin spot/interrupciones para la primera
   medición. Selecciona una región con volumen de red de 50 GB. El volumen Pod
   se monta en `/workspace`; no guardes pesos en disco efímero del contenedor.
2. Usa una imagen Linux con Python 3.11, soporte venv, `nvidia-smi`, `git`,
   `ffmpeg`, `libsndfile1`, `libgomp1` y certificados TLS. Si usas la imagen propia
   construida abajo, el checkout está en `/opt/VoiceStudioZam`, los venvs están
   en `/opt/aureo-env` y `/opt/aureo-env-gateway` y el comando del Pod debe ser
   `pod`: no necesitas clonar ni instalar de nuevo; ajusta las variables de
   intérpretes a esos paths, manteniendo almacenamiento en `/workspace/aureo`.
3. Abre el puerto HTTP **8000** en Runpod para la API. Usa el proxy HTTPS de
   Runpod `https://POD_ID-8000.proxy.runpod.net`; no es una URL creada aquí.
4. Clona **este repositorio** en el Pod, desde una revisión de FASE 4 concreta:

```bash
cd /workspace
git clone https://github.com/negociosdjzam-cpu/VoiceStudioZam.git VoiceStudioZam
cd VoiceStudioZam
git checkout fase-4-aureo-gpu-real
# Para repetir exactamente, sustituye la rama por el hash publicado de FASE 4.
export AUREO_STORAGE_DIR=/workspace/aureo
export AUREO_MODEL_DIR=/workspace/aureo/models/chatterbox
export AUREO_ENV_DIR=/workspace/aureo-env
export AUREO_GATEWAY_ENV_DIR=/workspace/aureo-env-gateway
export AUREO_WORKER_PYTHON=/workspace/aureo-env/bin/python
export AUREO_DEVICE=cuda:0
bash deploy/aureo/install.sh
```

El instalador aborta antes de instalar si no detecta NVIDIA. Instala únicamente
dependencias con verificación de hashes, nunca pesos. Se puede repetir sobre sus
dos venvs; rechaza `.venv` de la aplicación y directorios existentes que no sean
venvs. `--dry-run` muestra el plan; `--build` permite instalar dependencias sin
GPU al construir la imagen, pero sigue sin descargar modelos.

## 2. Aprovisionar solo este motor en el Pod GPU

Elige un commit concreto de
[ResembleAI/chatterbox](https://huggingface.co/ResembleAI/chatterbox/commits/main)
que contenga los cinco archivos siguientes. Guarda ese hash en
`AUREO_MODEL_REVISION`; no inventar ni usar una referencia móvil.

```bash
# AUREO_MODEL_REVISION debe contener el commit real elegido en Hugging Face.
"$AUREO_WORKER_PYTHON" -m aureo.gpu.provision --revision "$AUREO_MODEL_REVISION"
```

Descarga únicamente `ve.pt`, `t3_mtl23ls_v2.safetensors`, `s3gen.pt`,
`grapheme_mtl_merged_expanded_v1.json` y `Cangjie5_TC.json`. Registra repo,
revisión, SDK y SHA-256 en `aureo-model.json`. Un intento interrumpido puede
reanudarse con la misma revisión; otra revisión requiere otro directorio.
La carga verifica todos los hashes y usa `from_local`, nunca `from_pretrained`.
El diccionario Cangjie se resuelve desde su archivo local ya verificado.

**Sin GPU el aprovisionador aborta antes de importar Hub o descargar pesos.**
Esta fase no ejecutó ese comando contra Hugging Face.

## 3. Modelo precargado y prueba HTTP

Configura `AUREO_API_TOKEN` como secreto del Pod. Ejecuta:

```bash
bash deploy/aureo/start.sh pod
```

El gateway inicia un hijo y carga el modelo una sola vez **antes de aceptar
tráfico**. Un solo proceso Uvicorn y un hijo por GPU; no usar `--workers 2`
porque duplicaría modelo/VRAM. `/healthz` indica proceso HTTP vivo; `/readyz`
solo devuelve 200 si el hijo precargado sigue vivo. Referencias, pruebas,
benchmark y descargas requieren `Authorization: Bearer $AUREO_API_TOKEN`.

Para una referencia mono limpia de 6–10 s (admite PCM WAV mono/estéreo de 3–30 s):

```bash
python3.11 - <<'PY'
import base64,json
from pathlib import Path
payload = {"text":"Esta es una prueba real de AUREO en español.",
           "language":"es", "preset":"NATURAL", "seeds":[42,43,44],
           "reference_wav_base64":base64.b64encode(Path('/workspace/reference.wav').read_bytes()).decode()}
Path('/workspace/aureo-request.json').write_text(json.dumps(payload),encoding='utf-8')
PY
curl --fail-with-body http://127.0.0.1:8000/v1/aureo/test \
  -H "Authorization: Bearer $AUREO_API_TOKEN" -H 'Content-Type: application/json' \
  --data-binary @/workspace/aureo-request.json > /workspace/aureo-response.json
```

Esta llamada guarda **3 WAV** con seeds distintos y `metrics.json`/`report.md`
en `$AUREO_STORAGE_DIR/results/RUN_ID/`. El JSON entrega `run_id`, hash de la
referencia y nombres WAV, sin texto original ni rutas privadas. Para otra prueba
reutiliza `voice_id` igual a `reference_sha256`, omitiendo el base64 de referencia.
`POST /v1/aureo/reference` también permite registrar una referencia sin generar.

Descarga cada archivo por `GET /v1/aureo/runs/RUN_ID/FILENAME` con el mismo bearer.
`include_audio:true` añade WAV base64 a la respuesta, con límite conjunto de
8 MiB de base64. Para lotes grandes usa los archivos persistentes y descarga por
archivo. Las referencias están identificadas por SHA-256 e integridad verificada;
no se aceptan URLs, rutas arbitrarias ni pickle de embeddings desde la API.

## 4. Benchmark automático de tres estilos y tres seeds

`POST /v1/aureo/benchmark` usa el mismo payload sin `preset`: ejecuta
**NATURAL, ENERGÉTICO y FIDELIDAD × seeds 42/43/44 = 9 WAV**. `ENERGETICO` se admite
como alias. También puede ejecutarse como una sola orden local:

```bash
"$AUREO_GATEWAY_ENV_DIR/bin/python" -m aureo.gpu benchmark \
  --text "Esta es una prueba real de AUREO en español." \
  --reference /workspace/reference.wav --language es --seeds 42 43 44
```

El modelo se conserva durante las nueve tomas. La CLI termina al finalizar y
libera la GPU; para medir entre requests usa el servidor persistente.
`warmup` carga y comprueba el modelo, pero sale después: **no deja un proceso
caliente**. El servidor Pod o `runpod` sí mantiene el hijo y los pesos residentes.

Los presets aplican los controles compatibles de FASE 3 y registran omisiones.
Chatterbox admite seed, variation, energía/exaggeration y fidelity como fuerza
CFG; **fidelity no mide similitud de voz**. Este SDK no admite control nativo de
speed, expresión libre ni streaming real; opciones explícitas incompatibles
devuelven error. Cada WAV conserva Perth nativo y pasa además por el chokepoint
existente `mark_synthetic`. AudioSeal adicional es fail-open/offline y el JSON
registra si realmente se aplicó; no se descarga su modelo de forma automática.

## 5. Runpod Serverless con el mismo volumen

1. Desde este checkout construye una imagen propia; no usa el workflow GHCR
   heredado que apunta al namespace upstream:

```bash
docker build -f deploy/aureo/Dockerfile -t REGISTRO/USUARIO/aureo-gpu:FASE4 .
# Publicar en TU registro cuando corresponda; esta fase no publicó imágenes.
```

2. Publica la imagen en tu registro y configura en Runpod su digest concreto.
   Crea un endpoint **Queue-based**, imagen propia, comando por defecto
   `serverless`; el SDK expone `/run`, `/runsync` y `/status`. No abras 8000 en
   modo Serverless: el padre se comunica con Runpod mediante su SDK.
3. Adjunta **el mismo Network Volume** aprovisionado en el Pod, en su región.
   En Serverless se monta en `/runpod-volume`, de modo que los mismos archivos
   del Pod `/workspace/aureo` se encuentran en `/runpod-volume/aureo`.
4. Configura las variables de la columna Serverless de arriba. El volumen
   conserva pesos, referencias y WAV cuando el worker escala a cero; la caché
   de Voice DNA preparada vive solo en la RAM/VRAM del proceso, no en el volumen.
5. Empieza con **máximo 1 worker**, concurrencia de generación 1 y timeout de
   ejecución 600 s. Para bajo costo usa mínimo 0 e idle timeout 60 s; para
   mantenerlo precargado usa mínimo 1 / Active worker y asume su costo continuo.
   FlashBoot/model caching pueden ayudar, pero no eliminan toda inicialización.
6. Envía el payload como `{"input":{...}}` con `operation:"synthesize"` o
   `operation:"benchmark"`; el handler reutiliza el mismo hijo entre jobs.

```bash
curl --fail-with-body "https://api.runpod.ai/v2/$RUNPOD_ENDPOINT_ID/run" \
  -H "Authorization: Bearer $RUNPOD_API_KEY" -H 'Content-Type: application/json' \
  --data-binary @/workspace/aureo-runpod-job.json
# Consultar https://api.runpod.ai/v2/ENDPOINT_ID/status/JOB_ID
```

Crea `aureo-runpod-job.json` envolviendo la solicitud anterior en `input` y
añadiendo `operation`. Usa `/run` para el primer arranque/benchmark; el timeout
síncrono de `/runsync` no garantiza esperar un cold start largo. Runpod API key
permanece en el cliente que llama al endpoint, no en código ni en la imagen.
Para recuperar un WAV guardado envía otro job con
`{"input":{"operation":"fetch","run_id":"RUN_ID","filename":"NATURAL-seed42.wav"}}`.
Recibe `content_base64`. También puedes leer los resultados en un Pod adjunto
al volumen. Verifica los límites de payload/respuesta actuales del proveedor.

## Cold start, warm worker y métricas

Cold start incluye imagen, proceso, imports, verificación de SHA-256, carga y
sincronización CUDA; su tiempo GPU se guarda una vez en `model_startup` junto a
hardware, VRAM total, memoria muestreada y revisión del modelo. No hay cifras
reales todavía. Precargar significa ejecutar el servidor hasta `/readyz` 200,
no solamente tener archivos en disco.

En warm worker no se vuelve a cargar el modelo. La referencia preparada usa una
LRU GPU por hash del WAV + energía: tres seeds reutilizan el mismo Voice DNA;
distintos estilos/energías tienen condicionamiento separado. Cada nueva voz
puede evictar la anterior. La semilla no cambia la clave y la preparación tiene
su propio scope RNG fijo; cache hit/miss no debe cambiar la variante de un seed.
Un worker eliminado, timeout o reinicio pierde esa caché y vuelve a cargar.

Las filas registran carga warm (0), primer PCM, generación, exportación, total,
duración, RTF, RAM RSS muestreada, VRAM **asignada por PyTorch** muestreada,
cache hit/preparación, seed y errores. El primer audio es buffered y coincide
con la terminación nativa, incluyendo su Perth; no es una métrica de streaming.
El total de fila incluye marcar/guardar WAV; `batch_seconds` mide el lote.
`gateway_seconds`/`queue_wait_seconds` se devuelven en HTTP/IPC y no se escriben
en el JSON interno del hijo. Cola de Runpod, transferencia y startup de imagen
requieren las mediciones de Runpod: no están incluidas en TTFA nativo.
Los picos muestreados pueden omitir picos breves; no equivalen al consumo total
reportado por `nvidia-smi`. El límite del hijo mata/recoge el proceso completo;
una solicitud posterior puede iniciar otro. Errores de una toma se guardan y
las otras continúan; no se publican WAV parcialmente escritos de esa toma.

## Costo estimado y límites de esta fase

Para presupuestar un **Pod de 24 GB**, reserva de forma orientativa
**US$0.40–0.80/h**, más almacenamiento; es un presupuesto estimado, no una tarifa
Runpod verificada en vivo. La página dinámica de precios no fue accesible desde
este Cloud. Comprueba región/GPU/Secure o Community Cloud antes de contratar.

Serverless se cobra por segundo desde que el worker inicia hasta que se detiene,
incluyendo carga, ejecución e idle timeout. **Ejemplo aritmético, no tarifa:**
si la consola muestra US$0.0002/s, una hora facturada cuesta US$0.72 y 10 minutos
US$0.12. Mantener un worker 24 h costaría US$17.28/día con esa tarifa supuesta.
No asumir descuentos Active: la documentación actual remite a ventas.
El Network Volume cuesta US$0.07/GB/mes en el primer TB según la documentación
oficial consultada: 50 GB ≈ US$3.50/mes. El costo final depende de tarifas vigentes.

Fuentes oficiales consultadas (documentación obtenida del repositorio oficial
`runpod/docs`; no se creó ningún recurso de pago):
[precios](https://docs.runpod.io/serverless/pricing),
[volúmenes](https://docs.runpod.io/storage/network-volumes),
[handlers](https://docs.runpod.io/serverless/workers/handler-functions),
[requests](https://docs.runpod.io/serverless/endpoints/send-requests),
[tarifas actuales](https://www.runpod.io/pricing).

La validación local cubre SDK simulado, procesos reales, HTTP/IPC, caché,
presets/seeds, WAV/JSON, marca, timeouts y el rechazo sin CUDA. Queda pendiente
medir Chatterbox real en GPU, escuchar las variantes y comprobar límites bajo
carga antes de elegir FAST/PRO/ULTRA. No se ha hecho merge ni desplegado Runpod.

La construcción local de la imagen se intentó, pero Docker Hub respondió
**HTTP 429 Too Many Requests** al resolver la imagen Python base. La imagen
completa aún no está validada. No se saltaron hashes ni verificación TLS para
continuar; este bloqueo del registro requiere repetir la construcción cuando
la descarga esté disponible, antes de publicar/desplegar.

Comprobaciones realizadas en esta fase: **173 pruebas AUREO**, suite Python
completa **9.649 passed, 31 skipped, 8 xfailed, 1 xpassed** y **472 pruebas backend**.
Se instaló y repitió el instalador `--build` en dos venvs aislados, con hashes y
compatibilidad de dependencias aprobados; se importaron Chatterbox 0.1.7 con
Torch 2.6.0+cu124 y Runpod 1.12.0 con FastAPI 0.142.2. Un servidor HTTP real con
esos intérpretes completó nueve tomas de SDK simulado y descargó un WAV.
Wheel/sdist contienen los 44 módulos AUREO y conservan OmniVoice. Ninguna de
estas comprobaciones cargó pesos TTS ni demuestra rendimiento/calidad GPU.
