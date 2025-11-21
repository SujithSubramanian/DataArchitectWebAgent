# Data Architect Web Agent Dockerfile
# Use official Python image
FROM python:3.13-slim

# Set working directory
WORKDIR /app

# Copy project files
COPY . /app

# Install dependencies
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Expose Flask default port
EXPOSE 5000

# Optionally copy .env.example for reference (do not use real .env in image)
COPY .env.example /app/.env.example

# Set environment variables (for production, use --env-file or secrets)
ENV PYTHONUNBUFFERED=1

# Start the Flask app
CMD ["python", "app.py"]
