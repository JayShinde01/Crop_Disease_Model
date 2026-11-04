
FROM python:3.10.13-slim

# Prevent Python from creating cache files
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libgomp1 \
    libjpeg62-turbo \
    libpng16-16 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Create upload directory
RUN mkdir -p static/uploads

# Render provides PORT
ENV PORT=10000

# Expose application port
EXPOSE 10000

# Start Flask using Gunicorn
CMD ["sh", "-c", "gunicorn app:app --workers 1 --threads 1 --timeout 180 --bind 0.0.0.0:${PORT}"]
