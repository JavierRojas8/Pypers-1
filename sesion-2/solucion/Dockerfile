FROM python:3.10
RUN pip install fastapi "uvicorn[standard]" numpy
COPY fake_model.py /home/app/fake_model.py
WORKDIR /home/app/
CMD ["uvicorn", "fake_model:app", "--host", "0.0.0.0"]
