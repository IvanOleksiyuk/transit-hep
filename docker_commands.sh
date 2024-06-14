# Use files that were messed up by docker
chown -R $USER .

#!!!!!!!!!!!!!!!!!!! Use only to clean everything
# docker system prune -a
#!!!!!!!!!!!!!!!!!!! Use only to clean everything

sudo dockerd

docker login gitlab-registry.cern.ch -u ioleksiy
#failed attempts:
#docker pull gitlab-registry.cern.ch/curtains/docker/curtains:jraine-flows4flows
#docker pull gitlab-registry.cern.ch/curtains/dbbm/latest

cp docker/Dockerfile .
docker build -t curtains-ivan .
docker images

#docker run --rm -it -v ${PWD}:/workspace gitlab-registry.cern.ch/curtains/docker/curtains:jraine-flows4flows bash
docker run --gpus=all -v /home:/home -w /home/ivanoleksiyuk/WORK/hyperproject/ -i -t  hyperproject-ivan
docker run -v /home:/home -w /home/ivanoleksiyuk/WORK/hyperproject/ -d -t --gpus=all hyperproject-ivan

docker stop $(docker ps -a -q)
docker rm $(docker ps -a -q)