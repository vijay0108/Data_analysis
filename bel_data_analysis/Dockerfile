# syntax=docker/dockerfile:1
   
FROM python3
WORKDIR /Mlproject1
COPY . .
RUN yarn install --production
CMD ["python3", "/home/crl/Desktop/Mlproject1/final_freq_prediction.py"]
EXPOSE 3000
