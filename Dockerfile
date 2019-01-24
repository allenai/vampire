FROM allennlp/allennlp:v0.8.0

LABEL maintainer="suching@allenai.org"

WORKDIR /stage/allennlp

# Install postgres binary
RUN pip install numpy
RUN pip install pandas
RUN pip install pytest
RUN pip install torchvision
RUN pip install tabulate
RUN pip install regex

COPY bin/ bin/
COPY common/ common/
COPY data/ data/
COPY models/ models/
COPY modules/ modules/
COPY tests/ tests/
COPY training_config/ training_config/


# Optional argument to set an environment variable with the Git SHA
ARG SOURCE_COMMIT
ENV ALLENAI_VAE_SOURCE_COMMIT $SOURCE_COMMIT

EXPOSE 8000

ENTRYPOINT ["./bin/bash"]
