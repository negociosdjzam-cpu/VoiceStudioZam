# AUREO VOICE STUDIO — FASE 1

Primer rebranding visual de la aplicación VoiceStudio, realizado en el repositorio
Cloud `/workspace/VoiceStudioZam`, sin archivos de una computadora local.

## Cambios de identidad

- Nombre mostrado: **AUREO VOICE STUDIO** en la interfaz y en los 21 idiomas.
- Títulos de pestaña/ventana, recuperación, menús nativos y panel Acerca de.
- Encabezados, etiquetas, textos de configuración/ayuda y marca en ejemplos visibles.
- Símbolo existente en dorado: SVG, PNG, favicon y recursos ICO/ICNS de escritorio.
- Título de documentación API y marca en mensajes públicos de error.
- Etiqueta de la barra lateral ajustada para que el nombre no se corte.

## Compatibilidad preservada

No se cambiaron IDs internos, nombres de paquetes, versiones, lockfiles,
preload/IPC, esquema SQLite, migraciones, rutas API, directorios OmniVoice,
rutas de datos, contratos TTSBackend ni implementación/selección de motores.

`app.setName` conserva `VoiceStudio`, porque Electron utiliza ese nombre para
las rutas existentes de datos y caché. El nombre visible se aplica por separado
a ventanas, menús y Acerca de; los roles nativos mantienen sus acciones y el
idioma del sistema operativo.

El empaquetador, nombres de ejecutables/instaladores, appId y updater mantienen
sus valores anteriores. Los enlaces y feeds de actualización apuntan a sus
destinos originales. No se generaron ni publicaron nuevos instaladores.

Los nombres de motores (incluido `VoiceStudio (subprocess)` y OmniVoice),
URLs, correos de contacto, cuentas sociales y atribuciones originales permanecen.
Los nombres de perfiles ya almacenados, incluida la voz de demostración, son
datos del usuario: no se renombraron. Las capturas y muestras históricas del
proyecto original tampoco se editaron. LICENSE y los créditos se conservaron.

No se descargaron pesos de modelos de IA ni se ejecutó síntesis/transcripción.
El asistente inicial de la aplicación sigue disponible; su indicador de
finalización se simuló solamente en contextos temporales de navegador para
comprobar las pantallas de desarrollo.

## Inventario exacto

Los siguientes archivos contienen cambios de esta fase. Se conserva su ubicación
y nombre técnico; no se renombraron carpetas ni componentes.

### Traducciones (42 archivos)

Se modificaron únicamente valores de marca, manteniendo claves, interpolaciones,
URLs, correos, variables, headers y referencias de motores. En cada una de las
siguientes carpetas se modificaron estos 21 archivos:

- `electron/src/renderer/src/i18n/locales/`
- `electron/src/shared/i18n/locales/`

`ar.json`, `de.json`, `en.json`, `es.json`, `fr.json`, `hi.json`, `id.json`,
`it.json`, `ja.json`, `ko.json`, `nl.json`, `pl.json`, `pt.json`, `ru.json`,
`sv.json`, `th.json`, `tr.json`, `uk.json`, `vi.json`, `zh-CN.json`, `zh-TW.json`.

### Otros archivos

| Archivo | Cambio |
| --- | --- |
| [README.md](../README.md) | Nombre visible y documentación; atribución y enlaces originales conservados. |
| [backend/main.py](../backend/main.py) | Título de documentación API y textos públicos de error; rutas sin cambios. |
| [docs/logo.png](../docs/logo.png) | Símbolo existente en dorado; formatos, rutas y tamaños principales conservados. |
| [docs/logo.svg](../docs/logo.svg) | Símbolo existente en dorado; formatos, rutas y tamaños principales conservados. |
| [docs/rebranding-phase1.md](../docs/rebranding-phase1.md) | Informe de alcance, compatibilidad, inventario y validación. |
| [electron/README.md](../electron/README.md) | Nombre visible y documentación; atribución y enlaces originales conservados. |
| [electron/build/Info.plist](../electron/build/Info.plist) | Textos visibles de permisos de micrófono/cámara. |
| [electron/build/icons/32x32.png](../electron/build/icons/32x32.png) | Símbolo existente en dorado; formatos, rutas y tamaños principales conservados. |
| [electron/build/icons/icon.icns](../electron/build/icons/icon.icns) | Símbolo existente en dorado; formatos, rutas y tamaños principales conservados. |
| [electron/build/icons/icon.ico](../electron/build/icons/icon.ico) | Símbolo existente en dorado; formatos, rutas y tamaños principales conservados. |
| [electron/build/icons/icon.png](../electron/build/icons/icon.png) | Símbolo existente en dorado; formatos, rutas y tamaños principales conservados. |
| [electron/public/favicon.svg](../electron/public/favicon.svg) | Símbolo existente en dorado; formatos, rutas y tamaños principales conservados. |
| [electron/src/main/app-identity.test.ts](../electron/src/main/app-identity.test.ts) | Expectativas de marca actualizadas; aserciones funcionales conservadas. |
| [electron/src/main/app-identity.ts](../electron/src/main/app-identity.ts) | Nombre de menús/Acerca de; roles nativos localizados; nombre interno de almacenamiento conservado. |
| [electron/src/main/blank-window-guard.ts](../electron/src/main/blank-window-guard.ts) | Título de la página de recuperación. |
| [electron/src/main/index.ts](../electron/src/main/index.ts) | Título inicial de ventana y tooltip de bandeja. |
| [electron/src/renderer/index.html](../electron/src/renderer/index.html) | Título de pestaña/ventana. |
| [electron/src/renderer/src/components/app-shell/sponsor-inquiry.tsx](../electron/src/renderer/src/components/app-shell/sponsor-inquiry.tsx) | Nombre visible en el asunto de la consulta; destinatario conservado. |
| [electron/src/renderer/src/components/app-shell/workspace-sidebar.tsx](../electron/src/renderer/src/components/app-shell/workspace-sidebar.tsx) | Etiqueta de marca completa, con ajuste de línea para evitar recorte. |
| [electron/src/renderer/src/components/backend-gate.test.tsx](../electron/src/renderer/src/components/backend-gate.test.tsx) | Expectativas de marca actualizadas; aserciones funcionales conservadas. |
| [electron/src/renderer/src/components/report-bug.tsx](../electron/src/renderer/src/components/report-bug.tsx) | Nombre de la aplicación en el contexto visible del informe. |
| [electron/src/renderer/src/features/home/home-page.test.tsx](../electron/src/renderer/src/features/home/home-page.test.tsx) | Expectativas de marca actualizadas; aserciones funcionales conservadas. |
| [electron/src/renderer/src/features/integrations/integration-detail-page.test.tsx](../electron/src/renderer/src/features/integrations/integration-detail-page.test.tsx) | Expectativas de marca actualizadas; aserciones funcionales conservadas. |
| [electron/src/renderer/src/features/integrations/openai-agents-setup.ts](../electron/src/renderer/src/features/integrations/openai-agents-setup.ts) | Texto de ejemplo/comentarios mostrados; contratos y rutas sin cambios. |
| [electron/src/renderer/src/features/integrations/setup-registry.ts](../electron/src/renderer/src/features/integrations/setup-registry.ts) | Texto de ejemplo/comentarios mostrados; contratos y rutas sin cambios. |
| [electron/src/renderer/src/features/settings/gpu-acceleration.test.tsx](../electron/src/renderer/src/features/settings/gpu-acceleration.test.tsx) | Expectativas de marca actualizadas; aserciones funcionales conservadas. |
| [electron/src/renderer/src/lib/themes/t3-palettes.ts](../electron/src/renderer/src/lib/themes/t3-palettes.ts) | Nombre mostrado de la paleta; ID y colores de la paleta conservados. |
| [electron/src/renderer/src/lib/themes/tauri-palette.ts](../electron/src/renderer/src/lib/themes/tauri-palette.ts) | Nombre mostrado de la paleta; ID y colores de la paleta conservados. |
| [electron/src/shared/api/client.test.ts](../electron/src/shared/api/client.test.ts) | Expectativas de marca actualizadas; aserciones funcionales conservadas. |
| [electron/src/shared/components/AnalyticsConsentBanner.jsx](../electron/src/shared/components/AnalyticsConsentBanner.jsx) | Wordmark o textos de marca de respaldo; lógica sin cambios. |
| [electron/src/shared/components/AnalyticsConsentCard.jsx](../electron/src/shared/components/AnalyticsConsentCard.jsx) | Wordmark o textos de marca de respaldo; lógica sin cambios. |
| [electron/src/shared/components/BootstrapSplash.jsx](../electron/src/shared/components/BootstrapSplash.jsx) | Wordmark o textos de marca de respaldo; lógica sin cambios. |
| [electron/src/shared/components/FirstRunSetup.jsx](../electron/src/shared/components/FirstRunSetup.jsx) | Wordmark o textos de marca de respaldo; lógica sin cambios. |
| [electron/src/shared/components/Header.jsx](../electron/src/shared/components/Header.jsx) | Wordmark o textos de marca de respaldo; lógica sin cambios. |
| [electron/src/shared/components/HfTokenCard.jsx](../electron/src/shared/components/HfTokenCard.jsx) | Wordmark o textos de marca de respaldo; lógica sin cambios. |
| [electron/src/shared/components/LogsFooter.jsx](../electron/src/shared/components/LogsFooter.jsx) | Wordmark o textos de marca de respaldo; lógica sin cambios. |
| [electron/src/shared/components/UpdateToast.jsx](../electron/src/shared/components/UpdateToast.jsx) | Wordmark o textos de marca de respaldo; lógica sin cambios. |
| [electron/src/shared/components/brand/VoiceStudioMark.jsx](../electron/src/shared/components/brand/VoiceStudioMark.jsx) | Color dorado del símbolo; nombre técnico del componente conservado. |
| [electron/src/shared/components/donate/GoalBar.jsx](../electron/src/shared/components/donate/GoalBar.jsx) | Wordmark o textos de marca de respaldo; lógica sin cambios. |
| [electron/src/shared/components/settings/AboutTab.jsx](../electron/src/shared/components/settings/AboutTab.jsx) | Wordmark o textos de marca de respaldo; lógica sin cambios. |
| [electron/src/shared/components/settings/AboutTab.test.jsx](../electron/src/shared/components/settings/AboutTab.test.jsx) | Expectativas de marca actualizadas; aserciones funcionales conservadas. |
| [electron/src/shared/components/settings/AnalyticsOptIn.jsx](../electron/src/shared/components/settings/AnalyticsOptIn.jsx) | Wordmark o textos de marca de respaldo; lógica sin cambios. |
| [electron/src/shared/components/settings/ApiKeysPanel.jsx](../electron/src/shared/components/settings/ApiKeysPanel.jsx) | Wordmark o textos de marca de respaldo; lógica sin cambios. |
| [electron/src/shared/components/settings/GenerateBudgetPanel.jsx](../electron/src/shared/components/settings/GenerateBudgetPanel.jsx) | Wordmark o textos de marca de respaldo; lógica sin cambios. |
| [electron/src/shared/components/settings/HFMirrorPanel.jsx](../electron/src/shared/components/settings/HFMirrorPanel.jsx) | Wordmark o textos de marca de respaldo; lógica sin cambios. |
| [electron/src/shared/components/settings/JoinWorkerPanel.jsx](../electron/src/shared/components/settings/JoinWorkerPanel.jsx) | Wordmark o textos de marca de respaldo; lógica sin cambios. |
| [electron/src/shared/components/settings/MCPBindingsPanel.jsx](../electron/src/shared/components/settings/MCPBindingsPanel.jsx) | Wordmark o textos de marca de respaldo; lógica sin cambios. |
| [electron/src/shared/components/settings/OpenApiPanel.jsx](../electron/src/shared/components/settings/OpenApiPanel.jsx) | Wordmark o textos de marca de respaldo; lógica sin cambios. |
| [electron/src/shared/components/settings/OpenApiPanel.test.jsx](../electron/src/shared/components/settings/OpenApiPanel.test.jsx) | Expectativas de marca actualizadas; aserciones funcionales conservadas. |
| [electron/src/shared/components/settings/ResetPanel.jsx](../electron/src/shared/components/settings/ResetPanel.jsx) | Wordmark o textos de marca de respaldo; lógica sin cambios. |
| [electron/src/shared/components/settings/ResetPanel.test.jsx](../electron/src/shared/components/settings/ResetPanel.test.jsx) | Expectativas de marca actualizadas; aserciones funcionales conservadas. |
| [electron/src/shared/components/settings/StorageTab.jsx](../electron/src/shared/components/settings/StorageTab.jsx) | Wordmark o textos de marca de respaldo; lógica sin cambios. |
| [electron/src/shared/components/settings/StorageUsagePanel.jsx](../electron/src/shared/components/settings/StorageUsagePanel.jsx) | Wordmark o textos de marca de respaldo; lógica sin cambios. |
| [electron/src/shared/components/settings/UninstallPanel.jsx](../electron/src/shared/components/settings/UninstallPanel.jsx) | Wordmark o textos de marca de respaldo; lógica sin cambios. |
| [electron/src/shared/components/settings/UsageTab.jsx](../electron/src/shared/components/settings/UsageTab.jsx) | Wordmark o textos de marca de respaldo; lógica sin cambios. |
| [electron/src/shared/components/settings/WorkersPanel.jsx](../electron/src/shared/components/settings/WorkersPanel.jsx) | Wordmark o textos de marca de respaldo; lógica sin cambios. |
| [electron/src/shared/components/settings/WorkersPanel.test.jsx](../electron/src/shared/components/settings/WorkersPanel.test.jsx) | Expectativas de marca actualizadas; aserciones funcionales conservadas. |
| [electron/src/shared/components/settings/settingsCategories.jsx](../electron/src/shared/components/settings/settingsCategories.jsx) | Wordmark o textos de marca de respaldo; lógica sin cambios. |
| [electron/src/shared/components/settings/settingsCategories.test.jsx](../electron/src/shared/components/settings/settingsCategories.test.jsx) | Expectativas de marca actualizadas; aserciones funcionales conservadas. |
| [electron/src/shared/test/HeaderNavStyle.test.jsx](../electron/src/shared/test/HeaderNavStyle.test.jsx) | Expectativas de marca actualizadas; aserciones funcionales conservadas. |
| [electron/src/shared/test/LanguageSwitchPrompt.test.jsx](../electron/src/shared/test/LanguageSwitchPrompt.test.jsx) | Expectativas de marca actualizadas; aserciones funcionales conservadas. |
| [electron/src/shared/test/LogsFooterDonatePopover.test.jsx](../electron/src/shared/test/LogsFooterDonatePopover.test.jsx) | Expectativas de marca actualizadas; aserciones funcionales conservadas. |
| [electron/src/shared/test/SetupWizardConsent.test.jsx](../electron/src/shared/test/SetupWizardConsent.test.jsx) | Expectativas de marca actualizadas; aserciones funcionales conservadas. |
| [electron/src/shared/test/SupportPageSections.test.jsx](../electron/src/shared/test/SupportPageSections.test.jsx) | Expectativas de marca actualizadas; aserciones funcionales conservadas. |
| [electron/src/shared/test/SupportPageSponsors.test.jsx](../electron/src/shared/test/SupportPageSponsors.test.jsx) | Expectativas de marca actualizadas; aserciones funcionales conservadas. |
| [electron/src/shared/test/client.crash.test.ts](../electron/src/shared/test/client.crash.test.ts) | Expectativas de marca actualizadas; aserciones funcionales conservadas. |
| [electron/src/shared/test/client.restart.test.ts](../electron/src/shared/test/client.restart.test.ts) | Expectativas de marca actualizadas; aserciones funcionales conservadas. |
| [electron/src/shared/test/client.startFailure.test.ts](../electron/src/shared/test/client.startFailure.test.ts) | Expectativas de marca actualizadas; aserciones funcionales conservadas. |
| [electron/src/shared/test/client.unreachable.test.ts](../electron/src/shared/test/client.unreachable.test.ts) | Expectativas de marca actualizadas; aserciones funcionales conservadas. |
| [electron/tests/locale-encoding.mjs](../electron/tests/locale-encoding.mjs) | Expectativas de marca actualizadas; aserciones funcionales conservadas. |
| [electron/tests/packaging-contract.mjs](../electron/tests/packaging-contract.mjs) | Expectativas de marca actualizadas; aserciones funcionales conservadas. |
| [tests/test_voicestudio_branding.py](../tests/test_voicestudio_branding.py) | Expectativas de marca actualizadas; aserciones funcionales conservadas. |

## Validación reproducible

En el repositorio Cloud, activar primero
`source /workspace/.cloud-tools/voicestudio-env.sh`.
Para las peticiones Node del entorno, usar `NODE_USE_ENV_PROXY=1` y
`NODE_USE_SYSTEM_CA=1`; el instalador de Electron usa
`electron_config_cache=/workspace/.cache/electron`. Se instaló el binario
Electron ya fijado por el lockfile, con sus checksums originales, sin cambiar
dependencias ni desactivar verificación TLS.

- `bun run typecheck`
- `bun run build`
- `bun run build:web`
- `node electron/tests/packaging-contract.mjs`
- Desde `electron/`: `./node_modules/.bin/vp test --run --maxWorkers=2`.
- Desde `electron/`: `./node_modules/.bin/vp test --run --maxWorkers=4 --config vite.shared.config.ts`.
- Pytest seleccionado: `tests/test_voicestudio_branding.py`,
  `tests/test_locale_parity.py`, `tests/test_app_version.py`, `tests/smoke/`
  y `tests/test_health_liveness_2490.py`. También se comprobaron
  `tests/test_untrusted_mount_1957.py`, `tests/test_context_free_hints_1943.py`,
  `tests/test_port_in_use_exit.py`, `tests/test_response_safety.py`,
  `tests/test_host_memory_failure_2462.py` y `tests/test_input_too_short_1826.py`.
  Ejecutar con `uv run --no-sync pytest`,
  `HF_HUB_OFFLINE=1`, una caché HF vacía y directorios de datos/env temporales.
- Chromium del entorno: título, marca visible completa, edición de guion,
  página de apariencia y título OpenAPI a través del proxy real.
- Auditoría del diff: claves y tokens técnicos de los 42 JSON preservados;
  motores, migraciones, configuración de datos, paquetes y updater sin cambios.

Las pruebas nativas en macOS/Windows y el empaquetado de instaladores no se
realizaron en esta máquina Linux. La síntesis requiere modelos y no se validó.

### Resultados

- Suite principal de interfaz/Electron: **1.254 pruebas aprobadas**, 233 archivos.
- Suite compartida: **2.946 pruebas aprobadas**, 328 archivos.
- Selección Python de marca, traducciones, versión y smoke: **551 aprobadas**.
- Selección Python de seguridad/respuestas/errores de la API: **90 aprobadas**.
- Tipos, compilaciones web/Electron y contrato de empaquetado: aprobados.
- Tras el ajuste final de menús y etiqueta de barra lateral: 6 pruebas
  específicas aprobadas y nueva comprobación visual del nombre completo.
- Auditoría de tokens/archivos protegidos y `git diff --check`: aprobados.
- La caché configurada contiene cero archivos de pesos de modelos.

La validación se realizó en Cloud Linux. Los avisos de jsdom sobre canvas/media,
los avisos de tamaño de chunks y las advertencias de traducciones/deprecación de
Starlette no produjeron fallos. Las expectativas del nombre anterior se
actualizaron; no se deshabilitaron pruebas ni aserciones.
