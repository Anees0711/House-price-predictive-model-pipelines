FROM python:3.10-slim
WORKDIR /opt/project

COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

COPY . /opt/project/streamlit_app

ENV PYTHONPATH=/opt/project
EXPOSE 8501

CMD ["streamlit","run","/opt/project/streamlit_app/app.py","--server.port=8501","--server.address=0.0.0.0"]
