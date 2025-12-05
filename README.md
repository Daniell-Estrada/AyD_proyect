
# AyD Proyect

Bienvenido a AyD Proyect, una plataforma para el análisis y documentación de algoritmos.

## Estructura del Proyecto

- **backend/**: API y lógica de negocio en Python.
  - `main.py`: Punto de entrada.
  - `app/`: Módulos de aplicación, dominio, infraestructura, presentación y utilidades.
  - `tests/`: Pruebas unitarias y de integración.
- **frontend/**: Interfaz web en React + TypeScript.
  - `src/`: Componentes, páginas, servicios y utilidades.

## Instalación Rápida

### Backend
1. Ve a la carpeta `backend`.
2. Instala dependencias:
	```zsh
	pip install -r requirements.txt
	```
3. Ejecuta el servidor:
	```zsh
	python main.py
	```

### Configuración de entorno (backend)
Edita el archivo `.env` en `backend/` para definir tus claves y variables:
```env
# LLM API Keys
GITHUB_TOKEN=...
ANTHROPIC_API_KEY=...
OPENAI_API_KEY=...
GEMINI_API_KEY=...
MISTRAL_API_KEY=...

# MongoDB Configuration
MONGODB_URI=...
MONGODB_DB_NAME=AyD

# API Configuration
APP_ENV=development
API_DEBUG=true
SECRET_KEY=...
API_HOST=0.0.0.0
API_PORT=5000

# CORS Configuration
CORS_ORIGINS=[...]

# LLM Configuration
PRIMARY_LLM_PROVIDER=...
PRIMARY_LLM_MODEL=...
FALLBACK_LLM_PROVIDER=...
FALLBACK_LLM_MODEL=...
MAX_TOKENS=4096
TEMPERATURE=0.1

# Application Settings
LOG_LEVEL=INFO
MAX_SESSION_DURATION=3600
ENABLE_HITL=true

# Redis Configuration
REDIS_URL=...
REDIS_NAMESPACE=ayd_app
```

### Frontend
1. Ve a la carpeta `frontend`.
2. Instala dependencias:
	```zsh
	npm install
	```
3. Ejecuta la app:
	```zsh
	npm run dev
	```

### Configuración de entorno (frontend)
Edita el archivo `.env` en `frontend/` para definir la conexión:
```env
VITE_API_BASE_URL=http://localhost:5000
VITE_WS_URL=ws://localhost:5000

VITE_ENABLE_HITL=true
VITE_ENABLE_COST_TRACKING=true
```

## Características
- Análisis de algoritmos
- Documentación automática
- Interfaz interactiva y moderna
- Integración backend/frontend

## Contribuir
¡Las contribuciones son bienvenidas! Por favor, abre un issue o envía un pull request.

## Licencia
Este proyecto está bajo la licencia MIT.
