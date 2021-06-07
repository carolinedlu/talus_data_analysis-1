.venv:
	poetry install

format: .venv
	poetry run isort .
	poetry run black .
	poetry run flake8 .

test: .venv
	poetry run python -m pytest -s --durations=0 $(FILTER)

login:
	aws ecr get-login-password --region us-west-2 | docker login --username AWS --password-stdin 622568582929.dkr.ecr.us-west-2.amazonaws.com

build:
	docker build -t talus_streamlit .

tag:
	docker tag talus_streamlit:latest 622568582929.dkr.ecr.us-west-2.amazonaws.com/talus_streamlit:latest

push:
	docker push 622568582929.dkr.ecr.us-west-2.amazonaws.com/talus_streamlit:latest

pull:
	docker pull 622568582929.dkr.ecr.us-west-2.amazonaws.com/talus_streamlit:latest

run:
	docker container run -p 8501:8501 -d 622568582929.dkr.ecr.us-west-2.amazonaws.com/talus_streamlit:latest
