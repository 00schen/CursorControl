import pygame as pg
import numpy as np
from numpy.linalg import norm
from .base_oracles import UserInputOracle
SCREEN_SIZE = 300

class MouseOracle(UserInputOracle): # Not in use
	def _query(self):
		mouse_pos = pg.mouse.get_pos()
		new_mouse_pos = np.array(mouse_pos)-np.array([SCREEN_SIZE//2,SCREEN_SIZE//2])
		self.mouse_pos = mouse_pos
		radians = (np.arctan2(*new_mouse_pos) - (np.pi/3) + (2*np.pi)) % (2*np.pi)
		index = np.digitize([radians],np.linspace(0,2*np.pi,7,endpoint=True))[0]
		inputs = {
			1:	'right',
			4:	'left',
			2:	'forward',
			5:	'backward',
			3:	'up',
			6:	'down',
		}
		if norm(new_mouse_pos) > 50:
			self.action = inputs[index]
		else:
			self.action = 'noop'
		return {"mouse_pos": mouse_pos, "action": self.action}

	def reset(self):
		self.mouse_pos = pg.mouse.get_pos()