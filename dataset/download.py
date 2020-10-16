import gdown
import os

url = 'https://drive.google.com/uc?id=%s'%os.getenv('DATASET')
output = os.getenv('DATA_DIR')
gdown.download(url, '/app/flower_photos.tgz', quiet=False)

os.system('tar -xzvf /app/flower_photos.tgz -C %s'%output)
os.system('ls -al %s'%output)
