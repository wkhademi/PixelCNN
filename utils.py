import numpy as np
import matplotlib.pyplot as plt

def binarize(images):
	return (np.random.uniform(size=images.shape) < images).astype(np.float32)


# create set of images with bottom half of image missing
def trim_images(images):
	trimmed_test_images = images
	starting_trim = trimmed_test_images.shape[1]/2
	trimmed_test_images[:, starting_trim:, :, :] = 0

	return trimmed_test_images


def save_samples(samples, height, width):
	fig=plt.figure(figsize=(8, 8))
	columns = 4
	rows = 5

	for i in range(1, columns*rows +1):
		img = samples[i].reshape((height, width))
		fig.add_subplot(rows, columns, i)
		plt.imshow(img, cmap='gray')

	plt.savefig('samples_output.pdf')
