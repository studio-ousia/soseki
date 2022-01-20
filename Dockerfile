FROM python:3.8-slim-buster

# Expose the port for streamlit
EXPOSE 8501

RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates \
        && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Install requirements
RUN pip install --no-cache-dir \
        faiss-cpu \
        lmdb \
        numpy \
        onnxruntime \
        tqdm \
        transformers \
        streamlit

# Download transformers models in advance
ARG TRANSFORMERS_BASE_MODEL_NAME="bert-base-uncased"
RUN python -c "from transformers import AutoTokenizer; AutoTokenizer.from_pretrained('${TRANSFORMERS_BASE_MODEL_NAME}')"
ENV TRANSFORMERS_OFFLINE=1

# Copy files to the image
WORKDIR /app
COPY soseki/ soseki
COPY .streamlit/ .streamlit
COPY demo.py .
