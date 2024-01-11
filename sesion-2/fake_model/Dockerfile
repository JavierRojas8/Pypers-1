FROM python:3.10
RUN pip install fastapi numpy uvicorn
COPY server.py /home/app/server.py
WORKDIR /home/app/
CMD ["uvicorn","server:app" , "--host", "0.0.0.0"]