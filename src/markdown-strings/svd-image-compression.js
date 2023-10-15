import SVDImg from '../assets/svd-image-compression/svd.jpg'
import SVD1Img from '../assets/svd-image-compression/SVD1.png'
import CompressionImg from '../assets/svd-image-compression/compression.png'

export const markdownContent = `# SVD Image Compression

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/17IbzCC-dK-7qQinwu07AFMKdZYcEkvDW?usp=sharing)

I applied Singular Value Decomposition (SVD) to compress images. SVD is a matrix factorization technique which can be used to reduce the dimensionality of the data. It is also used for image compression. The SVD decomposes a matrix into 3 matrices. The first matrix contains the left singular vectors, the second matrix contains the singular values and the third matrix contains the right singular vectors. The singular values are the diagonal elements of the second matrix. The singular values are sorted in descending order. The first few singular values contain most of the information about the original matrix. The rest of the singular values contain very little information. We can use this property to compress images. We can keep only the first few singular values and discard the rest. This will reduce the size of the image. We can then reconstruct the image using the first few singular values. The reconstructed image will be very similar to the original image.

![CompressionImg](${CompressionImg})

![SVDImg](${SVDImg})

## Image Compression

![SVD1Img](${SVD1Img})



## Implementation of SVD Image Compression in Python using NumPy.
`

export const codeString =  `import numpy
from PIL import Image

# open the image and return 3 matrices, each corresponding to one channel (R, G and B channels)
def openImage(imagePath):
    imgOrig = Image.open(imagePath)
    im = numpy.array(imgOrig)

    aRed = im[:, :, 0]
    aGreen = im[:, :, 1]
    aBlue = im[:, :, 2]

    return [aRed, aGreen, aBlue, imgOrig]


# compress the matrix of a single channel
def compressSingleChannel(channelDataMatrix, singularValuesLimit):
    uChannel, sChannel, vhChannel = numpy.linalg.svd(channelDataMatrix)
    aChannelCompressed = numpy.zeros((channelDataMatrix.shape[0], channelDataMatrix.shape[1]))
    k = singularValuesLimit

    leftSide = numpy.matmul(uChannel[:, 0:k], numpy.diag(sChannel)[0:k, 0:k])
    aChannelCompressedInner = numpy.matmul(leftSide, vhChannel[0:k, :])
    aChannelCompressed = aChannelCompressedInner.astype('uint8')
    return aChannelCompressed

# importing image
aRed, aGreen, aBlue, originalImage = openImage('toy.jpg')

# image width and height
imageWidth = 512
imageHeight = 512

# number of singular values to use for reconstructing the compressed image
singularValuesLimit = 160

aRedCompressed = compressSingleChannel(aRed, singularValuesLimit)
aGreenCompressed = compressSingleChannel(aGreen, singularValuesLimit)
aBlueCompressed = compressSingleChannel(aBlue, singularValuesLimit)

imr = Image.fromarray(aRedCompressed, mode=None)
img = Image.fromarray(aGreenCompressed, mode=None)
imb = Image.fromarray(aBlueCompressed, mode=None)

newImage = Image.merge("RGB", (imr, img, imb))

newImage.save("compressed-toy.jpg")

# Calculate and display the compression ratio
mr = imageHeight
mc = imageWidth

originalSize = mr * mc * 3
compressedSize = singularValuesLimit * (1 + mr + mc) * 3

print('original size:')
print(originalSize)

print('compressed size:')
print(compressedSize)

print('Ratio compressed size / original size:')
ratio = compressedSize * 1.0 / originalSize
print(ratio)

print('Compressed image size is ' + str(round(ratio * 100, 2)) + '% of the original image ')
`
