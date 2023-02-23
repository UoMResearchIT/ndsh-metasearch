FROM python:3.9

WORKDIR /code

COPY ./requirements.txt /code/requirements.txt

RUN apt-get update && \
    apt-get install -y --no-install-recommends p7zip-full && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* && \
    7z -h

RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

COPY ./app /code/app

CMD uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-5000}
# CMD ["uvicorn", "app.main:app"]
