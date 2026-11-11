# DocuAsk - RAG assistant over FCC consumer guides.
# Works locally (docker build/run) and as a Hugging Face Docker Space.
FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Hugging Face Spaces run containers as user id 1000; use the same user locally.
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    HF_HOME=/home/user/.cache/huggingface \
    PYTHONPATH=/home/user/app/src \
    ANONYMIZED_TELEMETRY=False \
    TOKENIZERS_PARALLELISM=false
WORKDIR /home/user/app

# 1) Dependencies first, so code changes don't reinstall them (Docker layer cache).
COPY --chown=user requirements-docker.txt .
RUN pip install --user torch==2.15.0 --index-url https://download.pytorch.org/whl/cpu \
 && pip install --user -r requirements-docker.txt

# 2) Code, config and the raw documents (public-domain FCC guides, committed in the repo).
COPY --chown=user config.yaml pyproject.toml ./
COPY --chown=user src ./src
COPY --chown=user scripts ./scripts
COPY --chown=user app ./app
COPY --chown=user data/sources.csv ./data/sources.csv
COPY --chown=user data/raw ./data/raw

# 3) Build chunks + vector index at image build time: downloads the embedding model
#    once, so the container starts fast and needs no network except the LLM API.
RUN python scripts/ingest.py && python scripts/build_index.py --quiet

ENV HF_HUB_OFFLINE=1 \
    GRADIO_SERVER_NAME=0.0.0.0 \
    GRADIO_SERVER_PORT=7860
EXPOSE 7860

# GROQ_API_KEY must be provided at runtime (Space secret, or --env-file .env locally).
CMD ["python", "app/app.py"]
