#use python base image
FROM python:3.8-slim


#Set working directory
WORKDIR /app

#copy all item to containner under /app
COPY . /app

#install requirements
RUN pip install --no-cache-dir -r requirements.txt

#define environment variable
CMD ["python","simple_version.py","What is 5G?"]


