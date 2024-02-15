FROM nvcr.io/nvidia/pytorch:24.01-py3
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt
COPY service.py /home/app/service.py
WORKDIR /home/app
CMD BENTOML_PORT=8000 bentoml serve service:svc