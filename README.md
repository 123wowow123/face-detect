# create the virtualenv in the current folder (tf2)
pipenv --python 3.7
# run a new shell that uses the just created virtualenv
pipenv shell
# install, in the current virtualenv, tensorflow
pip install tensorflow==2.0
#or for GPU support: pip install tensorflow-gpu==2.0