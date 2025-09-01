# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the working directory
COPY requirements.txt ./

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir --upgrade -r requirements.txt

# Copy the rest of the application code into the working directory
COPY . .

# Expose the port that the app will run on
ENV PORT 8080
EXPOSE ${PORT}

# Run the Uvicorn server, binding to all interfaces and the specified port
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8080"]
