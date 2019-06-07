FROM allennlp/commit:234fb18fc253d8118308da31c9d3bfaa9e346861

LABEL maintainer="suching@allenai.org"

WORKDIR /vampire

RUN pip install pandas
RUN pip install pytest
RUN pip install torchvision
RUN pip install tabulate
RUN pip install regex
RUN pip install pylint==1.8.1
RUN pip install mypy==0.521
RUN pip install codecov
RUN pip install pytest-cov

RUN python -m spacy download en

COPY scripts/ scripts/
COPY environments/ environments/
COPY vampire/ vampire/
COPY training_config/ training_config/
COPY .pylintrc .pylintrc

# Optional argument to set an environment variable with the Git SHA
ARG SOURCE_COMMIT
ENV ALLENAI_VAMPIRE_SOURCE_COMMIT $SOURCE_COMMIT

EXPOSE 8000

ENTRYPOINT ["/bin/bash"]