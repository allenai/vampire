FROM allennlp/allennlp:v0.8.1

LABEL maintainer="suching@allenai.org"

WORKDIR /stage/allennlp

# Install postgres binary
RUN pip install pandas
RUN pip install pytest
RUN pip install torchvision
RUN pip install tabulate
RUN pip install regex

COPY scripts/ scripts/
COPY vae/ vae/
COPY training_config/ training_config/

# Optional argument to set an environment variable with the Git SHA
ARG SOURCE_COMMIT
ENV ALLENAI_VAE_SOURCE_COMMIT $SOURCE_COMMIT

EXPOSE 8000

ENTRYPOINT ["./bin/bash"]
