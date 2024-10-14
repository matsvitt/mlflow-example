 1011   less conda.yaml
 1017  conda create --name exploratory python=3.8
 1018  conda activate exploratory
 1019  conda env export --name exploratory > conda_env.yml
 1020  less conda_env.yml
 1021  less conda_env.yml
 1022  emacs conda_env.yml
 1023  conda env update --file conda_env.yml --prune
 1024  conda deactivate
 1025  conda activate exploratory
 1026  history | grep conda
