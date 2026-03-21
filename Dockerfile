FROM python:3.10-slim

# Install required Python libraries
RUN pip install --no-cache-dir sympy

WORKDIR /app