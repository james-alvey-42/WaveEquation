import json
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import imageio
import os
import time
from progress.bar import FillingCirclesBar


class WaveEquation():

	def __init__(self, params_file='params.json'):
		self.params = load_params(filename=params_file)
		self.a = self.params['dt']/self.params['dx']
		self.x = np.arange(start=self.params['x0'], 
						   stop=self.params['xmax'] + self.params['dx'], 
						   step=self.params['dx'])
		self.t = np.arange(start=self.params['t0'], 
						   stop=self.params['tmax'] + self.params['dt'], 
						   step=self.params['dt'])
		self.X, self.T = np.meshgrid(self.x, self.t)
		self.u = np.zeros((len(self.t), len(self.x)))
		self.u[0, :] = self.f(self.x)
		self.steps = 0
		self.done = False

	def f(self, x):
		return np.sin(2 * np.pi * x / np.max(self.x))

	def g(self, x):
		return np.cos(2 * np.pi * x / np.max(self.x))

	def first_step(self):
		x_plus = np.roll(self.x, -1)
		x_minus = np.roll(self.x, 1)
		self.u[1, :] = self.params['dt'] * self.g(self.x) \
					+ (1 - self.a**2) * self.f(self.x) \
					+ 0.5 * self.a**2 * (self.f(x_plus) + self.f(x_minus))
		self.steps += 1
		if self.steps == len(self.t):
			self.done = True

	def take_step(self):
		if not self.done:
			u_avg = np.roll(self.u[self.steps, :], -1) + np.roll(self.u[self.steps, :], 1)
			self.u[self.steps + 1, :] = -self.u[self.steps - 1, :] + 2 * (1 -self.a**2) * self.u[self.steps, :] + self.a**2 * u_avg
			self.steps += 1

		if self.steps == len(self.t) - 1:
			self.done = True



def load_params(filename='params.json'):
	with open(filename, 'r') as json_file:
		params = json.load(json_file)
	return params

def make_video(solution, gifname='movie.gif', duration=0.1, xkcd=False):
	params = solution.params
	step = int(0.01/params['dt'])
	bar = FillingCirclesBar('Loading', suffix='%(percent)d%%', max=int((solution.steps - 1)/step))
	images = []
	figsize = (6, 6)
	for subplot in range(1, solution.steps, step):
		if xkcd:
			plt.rcParams['text.usetex'] = False
			plt.xkcd()
		fig = plt.figure(figsize=figsize)
		ax = plt.subplot(1, 1, 1)
		plt.sca(ax)
		plt.plot(solution.x, solution.u[subplot - 1, :],
			c='#F61067',
			lw=3.5)
		if xkcd:
			plt.xlabel(r'x')
			plt.ylabel(r'u(x, t)')
		else:
			plt.xlabel(r'$x$')
			plt.ylabel(r'$u(x, t)$')
		plt.title('t = {:.2f}s'.format(params['t0'] + (subplot - 1) * params['dt']))
		if subplot > 1:
			plt.axis(axis)
		if subplot == 1:
			axis = plt.axis()
		filename = 'temp.png'
		plt.savefig(filename)
		plt.close()
		images.append(Image.open(filename))
		os.remove(filename)
		bar.next()
	bar.finish()
	print('', end='\r\r')
	if xkcd:
		imageio.mimsave('xkcd_' + gifname, images, duration=duration)
	else:
		imageio.mimsave(gifname, images, duration=duration)

def elapsed_time(start):
	return time.strftime("%s", time.localtime((time.time() - start)))


if __name__ == '__main__':
	start = time.time()
	
	print("\t\t\t\t\t\t\tElapsed")
	print("[Step 1] \t Initialising WaveEquation Instance \t{}s".format(elapsed_time(start)))
	
	solution = WaveEquation(params_file='precision-params.json')
	
	print("[Step 2] \t Taking first step \t\t\t{}s".format(elapsed_time(start)))
	
	solution.first_step()
	
	print("[Step 3] \t Solving full evolution \t\t{}s".format(elapsed_time(start)))
	
	while not solution.done:
		solution.take_step()
	
	print("[Step 4] \t Finised numerical solver \t\t{}s".format(elapsed_time(start)))
	print("[Step 5] \t Making gif \t\t\t\t{}s".format(elapsed_time(start)))
	
	make_video(solution, xkcd=False)
	
	print("[Complete] \t\t\t\t\t\t{}s".format(elapsed_time(start)))
