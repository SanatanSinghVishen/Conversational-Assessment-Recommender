FROM python:3.11-slim

WORKDIR /app

# Copy dependencies first to leverage Docker cache
COPY requirements.txt .

# Install dependencies (We keep the CPU-only PyTorch config in requirements.txt to save image size)
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application files
COPY . .

# Hugging Face Spaces routes traffic to port 7860 by default
EXPOSE 7860

# Start the FastAPI server
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]
