apt-get update && apt-get install -y libsm6 libxext6 libxrender-dev
pip install -r /labs/docker/requirements.txt
cd /labs/src
python main.py
