import numpy as np
from anisotropicTMM import anisotropicTMM
from scipy.optimize import least_squares
from scipy import constants
import matplotlib.pyplot as plt

class lorentz_fit:
	'''
	Fit exciton parameters of a single anisotropic excitonic layer with Lorentz model.
	parameters:
		asc_path: str
			angular resolved spectrum asc file path.
		d: list [nm]
			thickness of each non-air and non-excitonic layer, assumed to be surrounded by air and Silicon substrate.
		wavelength_range: list [nm]
			wavelength range of light, dim=(2,).
		phi: float [rad]
			incident angle, assumed to be in x-z plane.
		eps_list: list [1]
			list of relative dielectric constants for non-excitonic layers, dim=(num_layers-1, 3, 3).
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
		reflectance_data=np.mean(data_list[:,655:665], axis=(1))/100 # 520-800 is index range of sin(theta), depends on the asc file
		self.reflectance_data=reflectance_data

		#wavelength range
		wavelength=np.linspace(wavelength_range[0], wavelength_range[1], len(reflectance_data))
		self.wavelength=wavelength
		self.hbar_omega=constants.h*3e8/self.wavelength/1e-9/constants.e # [eV]

		#obtain reference reflectance
		self.reference_reflectance=self.reference()

	def reference(self):
		"""
		Calculate reference reflectance without excitonic layer.
		"""
		n = self.num_layers-1
		wlnm = len(self.wavelength)
		eps = np.zeros((n, 3, 3, wlnm), dtype=complex)
		for layer in range(n):
			eps_layer = self.eps_list[layer]
			eps[layer] = np.array([[[eps_layer[0,0]]*wlnm, [eps_layer[0,1]]*wlnm, [eps_layer[0,2]]*wlnm],
								  [[eps_layer[1,0]]*wlnm, [eps_layer[1,1]]*wlnm, [eps_layer[1,2]]*wlnm],
								  [[eps_layer[2,0]]*wlnm, [eps_layer[2,1]]*wlnm, [eps_layer[2,2]]*wlnm]])
		d_ref = self.d.copy()
		d_ref = np.array(d_ref)
		tmm = anisotropicTMM(eps, d_ref, self.wavelength, self.phi)
		R_ref = tmm.reflectance(linear_polarized=True)
		return R_ref
	
	def lorentz_eps(self, parameters):
		"""
		Lorentz model for dielectric function.
		"""
		eps_bg, f1, exciton1, gamma1, f2, exciton2, gamma2 = 11, 1.32, 1.367, 1.52e-3, 0.25, 1.3814, 6.5e-3
		omega= self.hbar_omega
		return eps_bg + f1 / (exciton1**2 - omega**2 - 1j * gamma1 * omega) + f2 / (exciton2**2 - omega**2 - 1j * gamma2 * omega)


	def build_epsilon_tensor(self, parameters):
		"""
		anisotropic epsilon tensor with i-th layer is Lorentz model in yy component and has off-diagonal xy, yx.
		"""
		num_layers = self.num_layers
		i = self.i
		wlnm = len(self.wavelength)
		eps = np.zeros((num_layers, 3, 3, wlnm), dtype=complex)
		for layer in range(num_layers):
			if layer == i-1:
				# Lorentz model in yy component
				yy = self.lorentz_eps(parameters)
				# Off-diagonal components
				xy = np.array([parameters[-3] + 1j * parameters[-2]]*wlnm)
				yx = np.conj(xy)
				eps[layer] = np.array([[[9]*wlnm, xy,      [0]*wlnm],
									  [yx,        yy,      [0]*wlnm],
									  [[0]*wlnm, [0]*wlnm, [5]*wlnm]])
			else:
				eps_layer = self.eps_list[layer if layer < i-1 else layer - 1]
				eps[layer] = np.array([[[eps_layer[0,0]]*wlnm, [eps_layer[0,1]]*wlnm, [eps_layer[0,2]]*wlnm],
									  [[eps_layer[1,0]]*wlnm, [eps_layer[1,1]]*wlnm, [eps_layer[1,2]]*wlnm],
									  [[eps_layer[2,0]]*wlnm, [eps_layer[2,1]]*wlnm, [eps_layer[2,2]]*wlnm]])
		return eps


	def reflectance_from_params(self, params):
		epsilon = self.build_epsilon_tensor(params)
		d = self.d.copy()
		d.insert(self.i-1, params[-1])
		d = np.array(d)
		tmm = anisotropicTMM(epsilon, d, self.wavelength, self.phi)
		R = tmm.reflectance(linear_polarized=True)
		return R/self.reference_reflectance


	def residuals(self, params):
		R_model = self.reflectance_from_params(params)
		mask=self.wavelength>900
		return (R_model - self.reflectance_data)[mask] # residuals with mask


	def fit_exciton_and_offdiagonals(self, p0=None):
		# initial guess
		if p0 is None:
			# eps_bg, f1, exciton1, gamma1, f2, exciton2, gamma2, xy_real, xy_imag, thickness[nm]
			p0 = [ 0, 0, 50]
		# 	p0 = [10, 1, 1.37, 0.0, 1, 1.35, 0.0, 0, 0, 50]
		lower = [-1e3, -1e3, 1]
		# lower = [1, 0, 1, -1, 0, 1, -1, -1e3, -1e3, 1]
		upper = [1e3, 1e3, 1e2]
		# upper = [1e2, 10, 2, 1, 10, 2, 1, 1e3, 1e3, 1e2]
		# call least_squares with verbose to observe progress; use x_scale to help scaling
		res = least_squares(self.residuals, p0, bounds=(lower, upper), method='trf', x_scale='jac', loss='soft_l1', f_scale=0.1, max_nfev=10000, ftol=1e-12, xtol=1e-12, verbose=2)
		m = res.fun.size
		cost = res.cost
		residual_norm = np.sqrt(2*cost)
		rmse = np.sqrt(np.sum(res.fun**2)/m)
		print('\n=== Fit summary ===')
		print('success:', res.success)
		print('message:', res.message)
		if cost is not None:
			print('cost (0.5 * sum(f^2)):', cost)
			print('residual norm ||f||_2:', residual_norm)
		if rmse is not None:
			print('RMSE:', rmse)
		print('fitted parameters (x):')
		print(res.x)
		print('===================\n')
		# compute model reflectance from fitted parameters and plot vs experiment
		R_model = self.reflectance_from_params(res.x)
		plt.figure(figsize=(6,4))
		plt.plot(self.wavelength, self.reflectance_data, color='red', label='experiment')
		plt.plot(self.wavelength, R_model, color='blue', label='model')
		plt.xlabel('wavelength (nm)')
		plt.ylabel('normalized reflectance')
		plt.legend()
		plt.tight_layout()
		plt.show()
		return res

eps_siO2 = np.array([[1.45, 0, 0], [0, 1.45, 0], [0, 0, 1.45]])  # SiO2
lorentz_fit('./data/new_0T-2.asc', d=[280], wavelength_range=[860,970], phi=0, eps_list=[eps_siO2], i=1).fit_exciton_and_offdiagonals()
