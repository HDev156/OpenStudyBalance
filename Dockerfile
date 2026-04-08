FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . /app

# Expose port 7860 for Hugging Face Spaces
EXPOSE 7860

# Default command for container startup
CMD ["python3", "api.py"]
