FROM python:3.10-slim

ARG USERNAME=appuser
ARG USER_UID=1000
ARG USER_GID=$USER_UID

RUN groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME

USER ${USERNAME}

ENV PATH="${PATH}:/home/${USERNAME}}/.local/bin"

COPY --chown=${USERNAME}:${USERNAME} . ./topic_modeling

WORKDIR /home/${USERNAME}/topic_modeling

RUN ${USERNAME} apt-get update && \
    ${USERNAME} apt-get -y install gcc

RUN pip install --upgrade pip --user && \
    pip install -r requirements.txt --user && \
    python3 -m spacy download fr_core_news_md && \
    python3 -m spacy download en_core_web_sm

ENTRYPOINT ["streamlit", "run", "./src/frontend/Onboarding.py"]
