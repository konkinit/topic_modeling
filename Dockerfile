FROM rapidsai/rapidsai:cuda11.8-runtime-ubuntu22.04-py3.10

# Define args
ARG USERNAME=appuser
ARG USER_UID=1000
ARG USER_GID=$USER_UID

RUN apt-get update

# Create the user
RUN source activate rapids \
    && groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME \
    # [Optional] Add sudo support. Omit if you don't need to install software after connecting.
    && apt-get update \
    && apt-get install -y sudo \
    && echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME \
    && chmod 0440 /etc/sudoers.d/$USERNAME

# Set user
USER ${USERNAME}

ENV PATH="${PATH}:/home/${USERNAME}}/.local/bin"

COPY --chown=${USERNAME}:${USERNAME} . /home/${USERNAME}/topic_modeling

WORKDIR /home/${USERNAME}/topic_modeling

RUN source activate rapids && sudo apt-get -y install gcc

RUN source activate rapids && bash package_installing.sh -e 'ci_environment'

EXPOSE 8501

ENTRYPOINT ["streamlit", "run", "./src/frontend/Onboarding.py"]
