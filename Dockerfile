FROM python:3.9

WORKDIR /code

COPY ./requirements.txt /code/requirements.txt

RUN \
  apk --update --no-cache add p7zip && \
  rm -rf /var/cache/apk/* /tmp/*

RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

COPY ./app /code/app

CMD uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-5000}
# CMD ["uvicorn", "app.main:app"]
