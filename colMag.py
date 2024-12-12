#!/usr/bin/env python3
import matplotlib
#matplotlib.use('MacOSX')
#import matplotlib
#matplotlib.use('TkAgg')
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from tkinter import *
from tkinter import simpledialog, messagebox
from PIL import Image, ImageTk
import os
import numpy as np


##############################################################################
#
# Program for laboratory exercise in ASTA33, Galaxies and Cosmology.
# Developed by Eric Andersson, October 2018.
#
##############################################################################

class Main(object):
	""" Class for the main running program, which holds information about the
	canvas, and how it is manipulated. This class depends on functions that 
	it calls from the Galaxy class. Ensure to initialize this class before 
	initializing this class.
	"""
	def __init__(self):
		""" Function for creating the canvas and setting up all buttons and 
		events.
		"""
		self.root = Tk() 
		self.history = [] # List that holds canvas objects.
		self.id = [] # List that holds galaxy id for data retrival.
		self.plotwindowdestroyed = True # Flag for plot window.
		
		# Set program title.
		self.root.title("Hubble Ultra Deep Field")
		
		# Retrive and set up image.
		image = Image.open('./files/colMag.jpg') # Load image file
		image = image.resize((int(730*scale),int(750*scale)), Image.LANCZOS) # Resize and smooth
		photo = ImageTk.PhotoImage(image) # Create photo object for canvas.
		
		# Create frame to hold image..
		frame = Frame(self.root, width=int(scale*730), height=int(scale*750), cursor="cross")
		frame.pack(side=TOP)
		
		# Set up canvas for image.
		self.canvas = Canvas(frame, width=int(scale*730), height=int(750*scale))
		self.canvas.create_image(0, 0, image=photo, anchor="nw")
		self.canvas.pack()
		
		# Create undo button.
		undo = Button(self.root, text="Undo", command=self.undo)
		undo.pack(side="left", padx=5, pady=1)
		
		# Create undo button.
		clear = Button(self.root, text="Clear", command=self.clear)
		clear.pack(side="left", padx=5, pady=1)
		
		# Create button for writing data to file.
		write = Button(self.root, text="Write to file", command=self.save_data)
		write.pack(side="left", padx=5, pady=1)
		
		# Create exit button.
		exit = Button(self.root, text="Close", command=self.exit_program)
		exit.pack(side="right", padx=20, pady=1)
		
		# Create button for showing Colour-Magnitude plot.
		colMagPlot = Button(self.root, text="Show plot", command=self.setupplot)
		colMagPlot.pack(side="left", padx=5, pady=1)
		
		# Add label for colour and magnitude of galaxy at hover.
		self.label = Label(self.root, text='Magniture: -    Colour: -')
		self.label.pack(side="left", padx=5, pady=1)
		
		# Bind mouse actions.
		self.canvas.bind("<ButtonPress-1>", self.select)
		
		# Initiate program.
		mainloop()
	
	def exit_program(self):
		""" Terminates the program.
		"""
		try:
			if not self.plotwindowdestroyed:
				self.plot.destroy()
				self.root.destroy()
			else:
				self.root.destroy()
		except AttributeError:
			self.root.destroy()
	
	def killed(self):
		""" Function that catches it plot window is force quit prematurly.
		"""
		self.plotwindowdestroyed = True
		self.plot.destroy()
	
	def setupplot(self):
		""" Function to add colour-magnitude diagram from selected data.
		"""
		if self.plotwindowdestroyed:
			# Set up the figure and add labels.
			self.plotwindowdestroyed = False
			self.plot = Tk()
			self.plot.title("Colour-Magnitude diagram")
			self.plot.attributes("-topmost", True)
			self.fig = Figure(figsize=(6,4))
			self.sp = self.fig.add_subplot(111)
			self.sp.set_title("Colour-Magnitude diagram")
			self.sp.set_ylabel("Colour (V-I)")
			self.sp.set_xlabel("Magnitude (I)")
			self.sp.set_ylim(0,4)
			self.sp.set_xlim(18,26)
			
			# Create canvas object for plot.
			canvas = FigureCanvasTkAgg(self.fig, master=self.plot)
			canvas.get_tk_widget().pack(fill='both', side=TOP)
			canvas.draw()
			self.plot.protocol("WM_DELETE_WINDOW", self.killed)
			
			# Add already selected point to plot.
			self.updateplot()
			self.fig.canvas.draw_idle()
	
	def updateplot(self):
		""" Function that updates the points plotted in the colour-magnitude 
		diagram.
		"""
		if not self.plotwindowdestroyed:
			x = np.zeros(len(self.id))
			y = np.zeros(len(self.id))
			xerr = np.zeros(len(self.id))
			yerr = np.zeros(len(self.id))
			
			for i in range(len(self.id)):
				_, x[i], xerr[i], y[i], yerr[i] = galaxy.get_data(self.id[i])
			
			# Clean up plot.
			try:
				self.ebar.remove()
				# Add points to plot.
				self.ebar = self.sp.errorbar(x,y,xerr=xerr,yerr=yerr,color='r', fmt=' ', 
					ecolor='k', elinewidth=1, capsize=2)
			except AttributeError:
				# Add points to plot.
				self.ebar = self.sp.errorbar(x,y,xerr=xerr,yerr=yerr,color='r', fmt=' ', 
					ecolor='k', elinewidth=1, capsize=2)
				
			self.fig.canvas.draw_idle()
	
	def undo(self):
		""" Function to remove selected points from history. Called by undo 
		button.
		"""
		if not len(self.history) == 0:
			self.canvas.delete(self.history[-1]) # Remove last canvas object.
			self.history = self.history[:-1] # Clean up history.
			galaxy.revoke_mark(self.id[-1]) # Remove mark from galaxy.
			self.id = self.id[:-1] # Remove saved id.
			self.updateplot()

	def clear(self):
		""" Function to clear all marked galaxies, including the plot.
		"""
		if not len(self.history) == 0:
			for i in range(len(self.history)):
				self.canvas.delete(self.history[i]) # Remove last canvas object.
				galaxy.revoke_mark(self.id[i]) # Remove mark from galaxy.
			self.history = [] # Clean up history.
			self.id = [] # Remove saved id.
			self.updateplot()
				
	def save_data(self):
		""" Function for writing data to file. Called by write button.
		"""
		# Prompt to ask for filename.
		filename = simpledialog.askstring('Save data', 'Enter filename:', 
					parent=self.root)
		if filename == None:
			return
		
		if filename == '':
			print("Did not provide name!")
			return
		
		# Set up directory and file extension.
		filename = './' + filename + '.dat'
		if os.path.isfile(filename):
			prompt = messagebox.askyesno('File already exists!', 
					filename+' already exists. Overwrite?')
			if not prompt:
				return
		
		# Set up file.
		with open(filename, 'w') as f: f.write('#mag\temag\tcol\tecol\n')
		
		# Write all saved data.
		for index in self.id:
			with open(filename, 'a') as f:
				f.write('%.2f\t%.2f\t%.2f\t%.2f\n' % galaxy.get_data(index)[1:])
		print('Saved data!')
	
	def select(self, event):
		""" Function that selects, stores data about, and marks points on 
		canvas. Called by click event.
		"""
		# Call galaxy to check if nearby galaxy exists.
		valid, x0, y0, index = galaxy.check_pos(event.x, event.y)
		
		# Write magnitude and colour to label.
		if valid:
			_, mag, _, col, _ = galaxy.get_data(index)
			self.label.config(
				text='Magniture: {}    Colour: {}'.format(mag[0][0], col[0][0]))
		
		# Add mark to nearest galaxy.
		if valid:
			color = "#FF0000"
			x1, y1 = (x0-1), (y0-1)
			x2, y2 = (x0+1), (y0+1)
			mark = self.canvas.create_oval(x1, y1, x2, y2, fill='', 
					outline=color, width=5)	# Add oval.
			self.history.append(mark) # Store object for undo.
			self.id.append(index) # Save id for data retrival.
			self.updateplot()

class Galaxies():
	""" Class that holds information about all the galaxies on the Hubble
	Ultra Deep Field image. Data is stored in external .txt file.
	"""
	def __init__(self):
		""" Function that initializes the Galaxy class and loads the data 
		used by it.
		"""
		datadir = './files/colMag.txt'
		self.id, self.x, self.y, self.mag, self.emag, self.col, \
				self.ecol = self._read_data(datadir)
		self.marked = np.zeros(self.x.size, dtype=bool)
	
	def check_pos(self, x1, y1):
		""" Function that checks if galaxy is close to the coordinates x1 and
		y1. If that is the case it returns the exact coordinates of the closest
		galaxy. Keeps track of already selected galaxies to prevent selecting 
		the same galaxy multiple times.
		"""
		dist = np.sqrt((self.x - x1)**2 + (self.y - y1)**2)
		arg = np.argmin(dist)
		index = self.id[arg]
		if dist[arg] < 15 and not self.marked[arg]:
			self.marked[arg] = True
			return True, self.x[arg], self.y[arg], index
		else:
			return False, 0, 0, index
	
	def revoke_mark(self, index):
		""" Function that unmarks a previously selected galaxy using the 
		position of that galaxy.
		"""
		arg = np.argwhere(self.id == index)
		self.marked[arg] = False
	
	def _read_data(self, datadir):
		""" Function that reads in the data. 
		"""
		data = np.loadtxt(datadir)
		return data[...,0].astype(int), (data[...,1]*scale), (data[...,2]-20)*scale, \
				data[...,3], data[...,4], data[...,5], data[...,6]
	
	def get_data(self, index):
		""" Function that returns data for given galaxy id.
		"""
		arg = np.argwhere(index == self.id)
		return (self.id[arg], self.mag[arg], self.emag[arg], self.col[arg], \
				self.ecol[arg])

if __name__ == "__main__":
	# Runs program.
	scale=0.8
	galaxy = Galaxies()
	Main()
