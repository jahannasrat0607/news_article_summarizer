# Use an official Python runtime as the base image
FROM python:3.10

# Set the working directory
WORKDIR /app

# Copy project files into the container
COPY . /app/

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose port
EXPOSE 7860

# Run the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]
