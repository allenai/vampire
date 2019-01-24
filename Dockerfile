FROM allennlp/commit:b6cc9d39651273e8ec2a7e334908ffa9de5c2026

LABEL maintainer="suching@allenai.org"

WORKDIR /stage/allennlp

COPY requirements.txt .
COPY scripts/install_requirements.sh scripts/install_requirements.sh
RUN ./scripts/install_requirements.sh

COPY scripts/ scripts/
COPY vae/ vae/
COPY training_config/ training_config/
COPY .pylintrc .pylintrc

# Optional argument to set an environment variable with the Git SHA
ARG SOURCE_COMMIT
ENV ALLENAI_VAE_SOURCE_COMMIT $SOURCE_COMMIT

EXPOSE 8000

ENTRYPOINT ["./bin/bash"]
