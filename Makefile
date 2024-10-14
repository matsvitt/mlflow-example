default: run.local

run:
	mlflow run git@github.com:matsvitt/mlflow-example.git

run.local:
	mlflow run .

update_yaml:
	conda env export --name mlexample > conda_env.yml

create_env:
	conda create --name mlbase2 python=3.11

prune:
	conda env update --file conda_env.yml --prune
