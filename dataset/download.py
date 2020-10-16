import gdown
import os

url = 'https://drive.google.com/uc?id=%s'%os.getenv('DATASET')
output = os.path.join('/app',os.getenv('DATA_FOLDER'))
gdown.download(url, '/app/flower_photos.tgz', quiet=False)

os.makedirs(output)
os.system('tar -xzvf /app/flower_photos.tgz -C %s'%output)
os.system('ls -al %s'%output)

print('finished')
