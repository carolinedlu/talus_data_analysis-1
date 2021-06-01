FROM python:3.8

WORKDIR /

# The enviroment variable ensures that the python output is set straight
# to the terminal without buffering it first
ENV PYTHONUNBUFFERED 1

WORKDIR /app

# streamlit-specific commands
RUN mkdir -p /root/.streamlit
RUN bash -c 'echo -e "\
[general]\n\
email = \"\"\n\
" > /root/.streamlit/credentials.toml'
RUN bash -c 'echo -e "\
[server]\n\
enableCORS = false\n\
" > /root/.streamlit/config.toml'

# exposing default port for streamlit
EXPOSE 8501

RUN apt-get update -y && apt-get install vim tmux -y

ENV POETRY_VERSION="1.1.6"

COPY pyproject.toml .

# Install poetry
RUN pip install poetry==$POETRY_VERSION
RUN poetry install

# copying all files over
COPY . .

CMD poetry run streamlit run streamlit_demo_val.py
