# Docker Setup para Mortality AMI Predictor

Este directorio contiene la configuración de Docker para ejecutar la aplicación Mortality AMI Predictor en contenedores.

## 📋 Requisitos Previos

- **Docker Desktop** (Windows/Mac) o **Docker Engine** (Linux)
  - Windows/Mac: [Descargar Docker Desktop](https://www.docker.com/products/docker-desktop)
  - Linux: [Instalar Docker Engine](https://docs.docker.com/engine/install/)
- **Docker Compose** (incluido en Docker Desktop, puede requerir instalación separada en Linux)

## 🚀 Inicio Rápido


```bash
# Dar permisos de ejecución a los scripts
cd scripts
chmod +x *.sh

# Ejecutar la aplicación
./run-app.sh

# Ejecutar en modo desarrollo (con Jupyter y MLflow)
./run-dev.sh

# Detener la aplicación
./stop-app.sh

# Reconstruir la imagen
./rebuild.sh
```

## 📦 Servicios Disponibles

### Modo Producción (por defecto)
- **Dashboard Streamlit**: http://localhost:8501
  - Interfaz principal de la aplicación

### Modo Desarrollo (con `--profile dev`)
- **Dashboard Streamlit**: http://localhost:8501
- **Jupyter Lab**: http://localhost:8888
  - Para desarrollo y análisis de datos
- **MLflow UI**: http://localhost:5000
  - Para tracking de experimentos

## 🛠️ Uso Manual con Docker Compose

### Construir la imagen

```bash
cd docker
docker-compose build
```

### Iniciar la aplicación (producción)

```bash
docker-compose up -d
```

### Iniciar con servicios de desarrollo

```bash
docker-compose --profile dev up -d
```

### Ver logs

```bash
# Todos los servicios
docker-compose logs -f

# Solo la aplicación principal
docker-compose logs -f app

# Solo Jupyter
docker-compose logs -f jupyter
```

### Detener los servicios

```bash
# Detener y eliminar contenedores
docker-compose down

# Detener y eliminar contenedores + volúmenes
docker-compose down -v
```

### Reconstruir desde cero

```bash
docker-compose down
docker-compose build --no-cache
docker-compose up -d
```

## 📂 Estructura de Archivos

```
docker/
├── Dockerfile              # Imagen principal para la aplicación
├── Dockerfile.jupyter      # Imagen para Jupyter Lab
├── docker-compose.yml      # Configuración de servicios
└── .dockerignore          # Archivos excluidos del build

scripts/
├── run-app.bat/.sh        # Iniciar aplicación (Windows/Linux-Mac)
├── run-dev.bat/.sh        # Iniciar modo desarrollo
├── stop-app.bat/.sh       # Detener aplicación
└── rebuild.bat/.sh        # Reconstruir imagen
```

## 💾 Volúmenes y Persistencia

Los siguientes directorios se montan como volúmenes para persistir datos:

- `DATA/` → Datos de entrada (solo lectura)
- `processed/` → Datos procesados
- `models/` → Modelos entrenados
- `mlruns/` → Experimentos de MLflow
- `logs/` → Logs de la aplicación

## 🔧 Personalización

### Cambiar el puerto del Dashboard

Editar `docker-compose.yml`:

```yaml
services:
  app:
    ports:
      - "8080:8501"  # Cambiar 8080 por el puerto deseado
```

### Agregar variables de entorno

Editar `docker-compose.yml`:

```yaml
services:
  app:
    environment:
      - MI_VARIABLE=valor
      - OTRA_VARIABLE=otro_valor
```

### Usar archivo de variables de entorno

Crear archivo `.env` en el directorio `docker/`:

```env
STREAMLIT_PORT=8501
JUPYTER_PORT=8888
MLFLOW_PORT=5000
```

Y referenciar en `docker-compose.yml`:

```yaml
services:
  app:
    env_file:
      - .env
```

## 🐛 Troubleshooting

### Error: "Cannot connect to Docker daemon"

**Solución**: Asegúrate de que Docker Desktop esté corriendo (Windows/Mac) o que el servicio de Docker esté activo (Linux):

```bash
# Linux
sudo systemctl start docker

# Verificar estado
docker info
```

### Error: "Port already in use"

**Solución**: Otro servicio está usando el puerto. Detén el servicio o cambia el puerto en `docker-compose.yml`.

```bash
# Ver qué está usando el puerto 8501
# Windows
netstat -ano | findstr :8501

# Linux/Mac
lsof -i :8501
```

### La aplicación no se inicia

**Solución**: Ver los logs para diagnosticar:

```bash
docker-compose logs app
```

### Reconstruir completamente

Si hay problemas persistentes:

```bash
# Detener todo
docker-compose down -v

# Limpiar imágenes
docker system prune -a

# Reconstruir
docker-compose build --no-cache
docker-compose up -d
```

## 📚 Recursos Adicionales

- [Documentación de Docker](https://docs.docker.com/)
- [Documentación de Docker Compose](https://docs.docker.com/compose/)
- [Documentación de Streamlit](https://docs.streamlit.io/)
- [Best Practices para Dockerfile](https://docs.docker.com/develop/develop-images/dockerfile_best-practices/)

## 🔒 Seguridad

⚠️ **Nota**: Esta configuración es para desarrollo/demostración. Para producción, considera:

- Usar secretos de Docker para credenciales
- Configurar HTTPS con certificados
- Implementar autenticación en Streamlit
- Usar redes Docker para aislar servicios
- Escanear imágenes para vulnerabilidades
- No exponer Jupyter sin autenticación

## 📝 Notas

- La imagen de Docker se optimiza para tamaño usando Python slim
- Los datos se montan como volúmenes para evitar reconstruir la imagen
- El modo desarrollo incluye herramientas adicionales para análisis
- Los logs se pueden ver en tiempo real con `docker-compose logs -f`
