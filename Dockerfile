# Containerizes the Streamlit demo of the preprocessing pipeline.
#
# Build:  docker build -t vitaldb-pipeline .
# Run:    docker run -p 8501:8501 vitaldb-pipeline
# Then open http://localhost:8501

FROM python:3.11-slim

WORKDIR /app

# Install dependencies first (better layer caching: only re-installs when
# requirements.txt actually changes, not on every code edit)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the project
COPY . .

EXPOSE 8501

HEALTHCHECK CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health')" || exit 1

ENTRYPOINT ["streamlit", "run", "streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
