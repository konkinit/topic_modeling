FROM python:3.10-slim

ARG USERNAME=appuser

ARG USER_UID=1000

RUN apt-get update

RUN useradd --uid $USER_UID -m $USERNAME

USER ${USERNAME}

ENV PATH="${PATH}:/home/${USERNAME}}/.local/bin"

COPY --chown=${USERNAME}:${USERNAME} . /home/${USERNAME}/topic_modeling

WORKDIR /home/${USERNAME}/topic_modeling

RUN --chown=${USERNAME}:${USERNAME} apt-get -y install gcc

RUN bash package_installing.sh

EXPOSE 8501

ENTRYPOINT ["streamlit", "run", "./src/frontend/Onboarding.py"]
