import numpy as np
from anisotropicTMM import anisotropicTMM
from scipy.optimize import least_squares
import scipy.io
import os

class lorentz_fit:
	'''
	Fit exciton parameters in yy component and off-diagonal xy,yx components in anisotropic epsilon tensor with Lorentz model.
	parameters:
		asc_path: str
			angular resolved spectrum asc file path.
		d: list [nm]
			thickness of each non-air and non-excitonic layer, assumed to be surrounded by air.
		wavelength_range: list [nm]
			wavelength range of light, dim=(2,).
		phi: float [rad]
			incident angle, assumed to be in x-z plane.
		eps_list: list [1]
			list of relative dielectric tensors for non-excitonic layers, dim=(num_layers-1, 3, 3, len(wavelength)).
		i: int
			index of the layer to fit exciton parameters, start with 1.
	'''
	def __init__(self, asc_path, d, wavelength_range, phi=0, eps_list=[], i=1):
		self.d=d
		self.phi=phi
		self.eps_list=eps_list
		self.i=i
		self.num_layers=len(d)+1

		#data from asc file
		with open(asc_path, 'r') as file:
			content = file.read().strip()
		segments = content.split('\n')
		data_list = []
		for segment in segments:
			if segment.startswith('[') and segment.endswith(']'):
				continue
			row_data = [float(x) for x in segment.split(',') if x.strip()]
			if row_data:
				data_list.append(row_data)
		data_list=np.array(data_list)
		reflectance_data=np.sum(data_list[:,520:800],axis=1) # 520-800 is index range of sin(theta), depends on the asc file
		self.reflectance_data=reflectance_data/np.max(reflectance_data) # normalize

		wavelength=np.linspace(wavelength_range[0], wavelength_range[1], len(reflectance_data))
		self.wavelength=wavelength
		self.omega=2*np.pi*3e8/self.wavelength/1e-9

	def lorentz_eps(self, parameters):
		"""
		Lorentz model for dielectric function.
		"""
		eps_bg, f1, omega1, gamma1, f2, omega2, gamma2 = parameters[:-3]
		omega= self.omega
		return eps_bg + f1 / (omega1**2 - omega**2 - 1j * gamma1 * omega) + f2 / (omega2**2 - omega**2 - 1j * gamma2 * omega)


	def build_epsilon_tensor(self, parameters):
		"""
		anisotropic epsilon tensor with i-th layer is Lorentz model in yy component and has off-diagonal xy, yx.
		"""
		num_layers = self.num_layers
		i = self.i
		omega = self.omega
		eps = np.zeros((num_layers, 3, 3, len(omega)), dtype=complex)
		for layer in range(num_layers):
			if layer == i-1:
				# Lorentz model in yy component
				yy = self.lorentz_eps(parameters)
				# Off-diagonal components
				xy = np.array([parameters[-3] + 1j * parameters[-2]]*len(omega))
				yx = np.conj(xy)
				eps[layer] = np.array([[[9]*len(omega), xy,            [0]*len(omega)],
									  [yx,              yy,            [0]*len(omega)],
									  [[0]*len(omega), [0]*len(omega), [0]*len(omega)]])
			else:
				eps[layer] = self.eps_list[layer if layer < i-1 else layer - 1]
		return eps


	def reflectance_from_params(self, params):
		epsilon = self.build_epsilon_tensor(params)
		d=self.d.insert(self.i-1, params[-1])
		tmm = anisotropicTMM(epsilon, d, self.wavelength, self.phi)
		R = tmm.reflectance(white_light=True)
		return R/np.max(R)


	def residuals(self, params):
		R_model = self.reflectance_from_params(params)
		return R_model - self.reflectance_data # residuals


	def fit_exciton_and_offdiagonals(self, p0=None):
		# initial guess
		if p0 is None:
			# eps_bg, f1, omega1, gamma1, f2, omega2, gamma2, xy_real, xy_imag, thickness[nm]
			p0 = [10, 1, 2.1e15, 0.0, 1, 2.2e15, 0.0, 0, 0, 50]
		res = least_squares(self.residuals, p0)
		return res
	
lorentz_fit('./data/new_0T-2.asc', d=[], wavelength_range=[860,970], phi=0, eps_list=[], i=1).fit_exciton_and_offdiagonals()