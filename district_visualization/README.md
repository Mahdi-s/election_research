# 🏙️ District Visualization Application

## 🚀 Getting Started

This guide will help you set up and run the District Visualization Flask application, either locally or using Docker.

## 🛠️ Prerequisites

- Python 3.7+
- pip
- Docker (optional, for containerized deployment)

## 🏃‍♂️ Running Locally


1. Create a virtual environment:

```
python -m venv venv
source venv/bin/activate # On Windows, use venv\Scripts\activate
```

3. Install dependencies:

```
pip install -r requirements.txt
```


4. Set environment variables:

```
export FLASK_APP=app.py
export FLASK_ENV=development
```

5. Run the Flask application:
```
flask run
```

The application should now be running at `http://localhost:5000`.

## 🐳 Running with Docker

1. Build the Docker image:
```
docker build -t district-visualization .
```

2. Run the Docker container:
```
docker run -p 5000:5000 district-visualization
```


The application should now be accessible at `http://localhost:5000`.

## 🔧 Configuration

- Modify `config.py` to adjust application settings.
- Environment variables can be used to override configuration values.

## 📚 API Documentation

Coming Soon!

