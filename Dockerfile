FROM allennlp/commit:234fb18fc253d8118308da31c9d3bfaa9e346861

LABEL maintainer="suching@allenai.org"

WORKDIR /stage/allennlp

ENV AWS_ACCESS_KEY_ID AKIAI3BY4Z3LJD6J6KKQ
ENV AWS_SECRET_ACCESS_KEY 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

RUN pip install pandas
RUN pip install pytest
RUN pip install torchvision
RUN pip install tabulate
RUN pip install regex
RUN pip install pylint==1.8.1
RUN pip install mypy==0.521
RUN pip install codecov
RUN pip install pytest-cov

COPY scripts/ scripts/
COPY vae/ vae/
COPY training_config/ training_config/
COPY .pylintrc .pylintrc

# Optional argument to set an environment variable with the Git SHA
ARG SOURCE_COMMIT
ENV ALLENAI_VAE_SOURCE_COMMIT $SOURCE_COMMIT

EXPOSE 8000

ENTRYPOINT ["python"]
