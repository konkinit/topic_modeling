FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime

COPY . /topic_modeling

WORKDIR /topic_modeling

RUN apt-get -y install gcc

RUN bash package_installing.sh -e 'docker_imaging'

EXPOSE 8501

ENTRYPOINT ["streamlit", "run", "./src/frontend/Onboarding.py"]
