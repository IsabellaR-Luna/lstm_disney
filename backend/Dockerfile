FROM python:3.12

WORKDIR /app

RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiar TUDO de uma vez (exceto o que está no .dockerignore)
COPY . .

# Criar diretórios se não existirem
RUN mkdir -p /app/models /app/data /app/logs

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8000/health')"

CMD ["uvicorn", "app.app:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]

# Debug: listar o que foi copiado
RUN echo "=== Conteúdo de /app ===" && ls -la /app/
RUN echo "=== Conteúdo de /app/models ===" && ls -la /app/models/ || echo "Pasta models não existe!"
RUN echo "=== Conteúdo de /app/data ===" && ls -la /app/data/ || echo "Pasta data não existe!"