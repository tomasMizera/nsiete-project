### Commands to run docker in dev
`docker build --tag nsiete-project project/docker`
`docker run -p 8888:8888 -p 6006:6006 -v ${pwd}/project:/labs --name nsiete-project -it nsiete-project`

### Run project in cloud and execute main.py script
`sudo docker run --gpus all -d -it --rm --name nn -v /home/project/nsiete-project/project:/labs tensorflow/tensorflow:latest-gpu-py3 bash labs/script.sh`
To view logs from docker container run 
`sudo docker logs -f nn` 

### Urls with train test data
https://www.kaggle.com/c/13333/download-all


### Notebooks
run jupyter notebook from `nsiete-project/project` directory
