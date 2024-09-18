#extracting the compressed dataset "sentiment140.zip"
from zipfile import ZipFile
dataset = '/Users/davidmajek/Desktop/Python/X sentiment Analysis /sentiment140.zip'

#reading the data in the zip file
with ZipFile(dataset,'r') as zip:
    zip.extractall()        
    print("Dataset is completely extraxted")