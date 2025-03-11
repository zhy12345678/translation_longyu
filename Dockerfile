# base stage
FROM python:3.10.16

WORKDIR /translation

COPY . /translation

RUN pip install --no-cache-dir -r requirements.txt

ENV FLASK_APP translate_file.py
ENV FLASK_RUN_HOST=0.0.0.0
ENV FLASK_RUN_PORT=5000

EXPOSE 5000

CMD ["flask","run"]