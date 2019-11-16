### Commands to run docker in dev
`docker build --tag nsiete-project project/docker`
`docker run -p 8888:8888 -p 6006:6006 -v ${pwd}/project:/labs --name nsiete-project -it nsiete-project`

### Run project in cloud and execute main.py script
`cd /home/project/nsiete-project/project`
`sudo docker run --gpus all -d -it --rm --name nn -v `pwd`:/labs tensorflow/tensorflow:latest-gpu-py3 bash labs/script.sh` 

### Urls with train test data
https://www.kaggle.com/c/13333/download-all
