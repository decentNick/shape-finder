import argparse
import cv2
import numpy as np

APPROX_EPSILON = 2e-2
ZERO_EPSILON = 1e-6

class Transform:
	def __init__(self, obj):
		self.scale = obj['scale'] if 'scale' in obj else 1
		self.angle = obj['angle'] * np.pi / 180.0 if 'angle' in obj else 0
		self.x0 = obj['x0'] if 'x0' in obj else 0
		self.y0 = obj['y0'] if 'y0' in obj else 0
		self.index = obj['index'] if 'index' in obj else -1
		self.polygon = obj['polygon'] if 'polygon' in obj else None
		self.metric = -1
	def print(self):
		if self.polygon is not None:
			print(self.index, self.x0, self.y0, self.scale, self.angle, sep=', ')
	@staticmethod
	def calc_geometric(polygon, shape, scale):
		norm_poly = (polygon[1] - polygon[0]) / np.linalg.norm(polygon[1] - polygon[0])
		norm_shape = (shape[1] - shape[0]) / np.linalg.norm(shape[1] - shape[0])
		if np.abs(norm_shape[0]) <= ZERO_EPSILON:
			s = -norm_poly[0] / norm_shape[1]
			c = norm_poly[1] / norm_shape[1]
		elif np.abs(norm_shape[1]) <= ZERO_EPSILON:
			s = norm_poly[1] / norm_shape[0]
			c = norm_poly[0] / norm_shape[0]
		else:
			s = (norm_poly[1] / norm_shape[1] - norm_poly[0] / norm_shape[0]) / \
				(norm_shape[0] / norm_shape[1] + norm_shape[1] / norm_shape[0])
			c = (norm_poly[0] / norm_shape[1] + norm_poly[1] / norm_shape[0]) / \
				(norm_shape[0] / norm_shape[1] + norm_shape[1] / norm_shape[0])
		angl = np.arcsin(s) if c >= 0 else np.pi - np.arcsin(s)
		sc_shape = shape[0] * scale
		x0 = polygon[0][0] - (sc_shape[0] * c - sc_shape[1] * s)
		y0 = polygon[0][1] - (sc_shape[0] * s + sc_shape[1] * c)
		return angl, x0, y0
	def is_better_shape(self, shape, scale):
		changed = False
		if self.polygon.shape[0] != shape.shape[0]:
			return None
		for i in range(self.polygon.shape[0]):
			angle, x0, y0 = Transform.calc_geometric(self.polygon, shape, scale)
			# m = calc_metric(polygon, shape, scale, a, x, y)
			m = Transform.calc_metric(self, shape, scale, angle, x0, y0)
			if self.metric < 0 or self.metric > m:
				self.angle = int(round(angle * 180 / np.pi))
				self.x0 = int(round(x0)) - 1
				self.y0 = int(round(y0)) - 1
				self.metric = m
				changed = True
			self.polygon = np.roll(self.polygon, 1, 0)
		return changed
	def find_shape(self, shapes):
		poly_square = cv2.contourArea(self.polygon)
		for i in range(len(shapes)):
			scale = np.sqrt(poly_square / cv2.contourArea(shapes[i]))
			if Transform.is_better_shape(self, shapes[i], scale):
				self.index = i
				self.scale = int(round(scale))
	def calc_metric(self, shape, scale, angle, x0, y0):
		com_sum = 0
		for i in range(self.polygon.shape[0]):
			x1, y1 = shape[i] * scale
			x2 = x0 + x1 * np.cos(angle) - y1 * np.sin(angle)
			y2 = y0 + x1 * np.sin(angle) + y1 * np.cos(angle)
			com_sum += np.linalg.norm(self.polygon[i] - (x2, y2))
		return com_sum / self.polygon.shape[0]
	def is_shape_matched(self):
		return self.metric != -1

def get_input_data():
	parser = argparse.ArgumentParser()
	parser.add_argument('-s', type=str)
	parser.add_argument('-i', type=str)
	args = parser.parse_args()
	shapes = []
	with open(args.s) as file:
		for _ in range(int(file.readline())):
			shape = np.array(list(map(int, file.readline().split(', ')))).reshape((-1, 2))
			shape = shape[::-1]
			shapes.append(shape)
	return shapes, cv2.imread(args.i, cv2.IMREAD_GRAYSCALE)

def get_polygons(image):
	img = cv2.copyMakeBorder(image, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
	contours, hierarchy = cv2.findContours(image, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
	contours = np.array(contours)[hierarchy[0, :, 3] != -1]
	new_image = np.ones_like(img)
	cv2.drawContours(new_image, contours, -1, 0, -1)
	img[new_image == 1] = 255
	img = cv2.morphologyEx(img, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
	plgs = []
	contours, hierarchy = cv2.findContours(img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
	for contour in contours[:-1]:
		epsilon = APPROX_EPSILON * cv2.arcLength(contour, True)
		plg = cv2.approxPolyDP(contour, epsilon, True)
		plgs.append(np.squeeze(plg, 1))
	return plgs

def find_shapes(shapes, image):
	result = []
	polygons = get_polygons(image)
	for polygon in polygons:
		transform = Transform({'polygon': polygon})
		transform.find_shape(shapes)
		if transform.is_shape_matched():
			result.append(transform)
	return result

def draw(inpt, gt, shape, transform, color=255):
	assert (inpt.shape == gt.shape)
	new_shape = shape.copy().astype(np.float)
	# Scale
	new_shape *= transform.scale
	# Rotation
	tmp = new_shape.copy()
	for i in [0, 1]:
		new_shape[:, i] = np.cos(transform.angle) * tmp[:, i] - ((-1) ** i) * np.sin(transform.angle) * tmp[:, 1 - i]
	# Shift
	new_shape[:, 0] += transform.dx
	new_shape[:, 1] += transform.dy
	cv2.fillPoly(gt, [new_shape.astype(np.int32)], color)
	cv2.polylines(inpt, [new_shape.astype(np.int32)], True, color)

if __name__ == '__main__':
	basic_shapes, image = get_input_data()
	shapes = find_shapes(basic_shapes, image)
	print(len(shapes))
	for transform in shapes:
		transform.print()
