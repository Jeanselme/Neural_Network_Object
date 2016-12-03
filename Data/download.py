import io
import os
import urllib.request
import gzip

def downloadDecompress(url, fileName, saveDirectory):
	"""
	Downloads and extract the content of the given fileName
	"""
	if not(os.path.exists(saveDirectory + fileName)):
		response = urllib.request.urlopen(url + fileName)
		compressedFile = io.BytesIO(response.read())

		with open(saveDirectory + fileName, 'wb') as out:
		    out.write(compressedFile.read())

if __name__ == '__main__':
	print("Download")
	fileNames= ["train-labels-idx1-ubyte.gz", "train-images-idx3-ubyte.gz",
		"t10k-labels-idx1-ubyte.gz", "t10k-images-idx3-ubyte.gz"]
	for fileName in fileNames:
		downloadDecompress("http://yann.lecun.com/exdb/mnist/", fileName, "Data/")
