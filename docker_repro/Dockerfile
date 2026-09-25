FROM python:3.14.3-slim-bookworm
ENV PYTHONUNBUFFERED=1 PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HOME=/tmp/hf \
    NVIDIA_VISIBLE_DEVICES=all NVIDIA_DRIVER_CAPABILITIES=compute,utility
WORKDIR /app
COPY requirements.txt environment_observed.txt /app/
RUN python -m pip install --no-cache-dir torch==2.14.0 torchvision==0.29.0 --index-url https://download.pytorch.org/whl/cu130 \
 && python -m pip install --no-cache-dir -r /app/requirements.txt -c /app/environment_observed.txt
COPY src /app/src
COPY metadata /app/metadata
COPY weights /app/weights
COPY reproduce.py MANIFEST.json /app/
ENTRYPOINT ["python", "/app/reproduce.py"]
CMD ["check"]
