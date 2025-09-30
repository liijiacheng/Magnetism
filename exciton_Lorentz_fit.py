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

		#obtain experiment data from file
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
		reflectance_data=np.mean(data_list[:,655:665], axis=(1))/100 # 655-665 is the approximate index range of sin(theta)=0, depends on the asc file
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
		R_ref = tmm.reflectance(y_polarized=True)
		return R_ref
	
	def lorentz_eps(self, parameters):
		"""
		Lorentz model for dielectric function.
		"""
		eps_bg, f1, exciton1, gamma1, f2, exciton2, gamma2 = 11.1, 1.32, 1.367, 1.52e-3, 0.25, 1.3814, 6.5e-3
		omega= self.hbar_omega
		return eps_bg + f1 / (exciton1**2 - omega**2 - 1j * gamma1 * omega) + f2 / (exciton2**2 - omega**2 - 1j * gamma2 * omega)


	def build_epsilon_tensor(self, parameters):
		"""
		anisotropic epsilon tensor with i-th layer having Lorentz-model-type yy component and off-diagonal components xy, yx.
		"""
		num_layers = self.num_layers
		i = self.i
		wlnm = len(self.wavelength)
		eps = np.zeros((num_layers, 3, 3, wlnm), dtype=complex)
		for layer in range(num_layers):
			if layer == i-1:
				# Lorentz model in yy component
				yy = self.lorentz_eps(parameters)
				xy = np.array([parameters[-3]+1j*parameters[-2]]*wlnm)
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
		'''
		Calculate reflectance from given parameters.
		'''
		epsilon = self.build_epsilon_tensor(params)
		d = self.d.copy()
		d.insert(self.i-1, params[-1])
		d = np.array(d)
		tmm = anisotropicTMM(epsilon, d, self.wavelength, self.phi)
		R = tmm.reflectance(y_polarized=True)
		return R

	def residuals(self, params, use_log=True):
		'''
		Calculate residuals between model and experiment.
		'''
		R_model = self.reflectance_from_params(params)
		ref = self.reference_reflectance
		y = self.reflectance_data
		model = R_model/ref

		if use_log:
			resid = np.log(y) - np.log(model)
		else:
			resid = y - model
		mask = self.wavelength > 895
		resid = resid[mask]
		return resid

	def _mad(self, x):
		'''
		Median Absolute Deviation for f_scale evaluation
		'''
		med = np.median(x)
		return np.median(np.abs(x - med)) * 1.4826

	def fit_exciton_and_offdiagonals(self, p0=[10, 10, 38], lower = [-1e2, -1e2, 35], upper = [1e2, 1e2,40]):
		'''
		Fit exciton parameters, layer thicknesses and off-diagonal components.
		'''
		# setting f_scale, x_scale, diff_step
		r0 = self.residuals(p0, use_log=False)
		f_scale = max(self._mad(r0), 1e-3)
		x_scale = np.maximum(np.abs(p0), 1.0)

		# least squares fitting
		res = least_squares(
			lambda x: self.residuals(x, use_log=False),
			p0,
			bounds=(lower, upper),
			method='trf',
			loss='linear',
			f_scale=f_scale,
			x_scale=x_scale,
			max_nfev=10000,
			ftol=1e-12,
			xtol=1e-12,
			gtol=1e-12,
			verbose=2
		)
		m = res.fun.size
		cost = res.cost
		residual_norm = np.sqrt(2*cost) if cost is not None else np.nan
		rmse = np.sqrt(np.sum(res.fun**2)/m) if m>0 else np.nan

		# print summary
		print('\n=== Fit summary ===')
		print('success:', res.success)
		print('message:', res.message)
		print('cost (0.5 * sum(f^2)):', cost)
		print('residual norm ||f||_2:', residual_norm)
		print('RMSE:', rmse)
		print('x:', res.x)
		print('===================\n')

		# plot
		R_model = self.reflectance_from_params(res.x)
		ref = self.reference_reflectance
		plt.figure(figsize=(6,4))
		plt.plot(self.wavelength, self.reflectance_data, color='red', label='experiment')
		plt.plot(self.wavelength, R_model/ref, color='blue', label='TMM fit')
		plt.xlabel('wavelength (nm)')
		plt.ylabel('relative reflectance')
		plt.legend()
		plt.tight_layout()
		plt.show()
		return res


eps_siO2 = np.array([[1.45**2, 0, 0], [0, 1.45**2, 0], [0, 0, 1.45**2]])  # SiO2
fit=lorentz_fit('./data/new_0T-2.asc', d=[280], wavelength_range=[870,950], phi=0, eps_list=[eps_siO2], i=1)
fit.fit_exciton_and_offdiagonals()
