#!/usr/bin/env python
import os
import zipfile
from os.path import basename

def zipdir(path, ziph):
    # ziph is zipfile handle
    for root, dirs, files in os.walk(path):
        for file in files:
            ziph.write(os.path.join(root, file),"/result/"+ str(file))

if __name__ == '__main__':
    zipf = zipfile.ZipFile('resultados.zip', 'w', zipfile.ZIP_DEFLATED)
    path = '/home/servidor-lasos/Thiago/Robson/DW/dwmlssp/result/'
    zipdir(path, zipf)
    zipf.close()
