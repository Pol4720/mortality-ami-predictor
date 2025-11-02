# Dockerfile para Mortality AMI Predictor
FROM python:3.11-slim

# Establecer variables de entorno
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Crear directorio de trabajo
WORKDIR /app

# Instalar dependencias del sistema
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copiar archivos de requisitos
COPY requirements.txt .

# Instalar dependencias de Python
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Copiar el código de la aplicación
COPY src/ ./src/
COPY dashboard/ ./dashboard/
COPY notebooks/ ./notebooks/
COPY config.py .
COPY __init__.py .

# Crear directorios necesarios
RUN mkdir -p processed/models processed/plots processed/cleaned_datasets \
    models mlruns logs

# Copiar y dar permisos al script de healthcheck
COPY docker/healthcheck.sh /healthcheck.sh
RUN chmod +x /healthcheck.sh

# Healthcheck
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD /healthcheck.sh

# Exponer puerto para Streamlit
EXPOSE 8501

# Comando por defecto: ejecutar dashboard
CMD ["streamlit", "run", "dashboard/Dashboard.py", "--server.port=8501", "--server.address=0.0.0.0"]
