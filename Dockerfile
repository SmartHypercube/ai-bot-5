FROM registry.digitalocean.com/hypercube/debug:v3.0.0

RUN set -ex; \
    sudo apt-get update; \
    sudo apt-get install -y python3-venv sqlite3; \
    sudo apt-get clean; \
    sudo rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./

RUN set -ex; \
    python3 -m venv venv; \
    venv/bin/pip install --no-cache-dir -r requirements.txt
