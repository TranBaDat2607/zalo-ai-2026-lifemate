# Final LifeMate solution. Runs fully offline: every model is baked into the image.
#
# Build from the repository root, with the exported models in ./models (src/lifemate/export_models.py):
#   docker build -f docker/Dockerfile -t lifemate:final .
# Run the way the organisers do, with the test split mounted at /data and the result collected from /result:
#   docker run --rm --gpus all --network none -v /path/to/private_test:/data -v "$PWD/result:/result" lifemate:final
# The default command is `bash /code/predict.sh`, so `docker run ... lifemate:final bash /code/predict.sh` is the same.
FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends libsndfile1 && rm -rf /var/lib/apt/lists/*

WORKDIR /code
COPY docker/requirements.txt /code/requirements.txt
RUN pip install --no-cache-dir -r /code/requirements.txt

# Training and inference code, the data made here (synthetic conversations, validation split, transcripts) and the scripts
# that ran the experiments; then the merged models.
COPY src/lifemate /code/lifemate
COPY scripts /code/scripts
COPY tests /code/tests
COPY pyproject.toml /code/pyproject.toml
COPY docker/README.md /code/README.md
COPY docker/predict.sh /code/predict.sh
COPY package_data /code/data
COPY models /code/models

ENV PYTHONPATH=/code HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONHASHSEED=2026
CMD ["bash", "/code/predict.sh"]
