import numpy as np
from anisotropicTMM import anisotropicTMM
from scipy.optimize import least_squares
from scipy import constants
import matplotlib.pyplot as plt
import time

class lorentz_fit:
	'''
	Fit parameters of single anisotropic excitonic layer with Lorentz model in multilayer thin films.
	parameters:
		file_path: str
			angular resolved spectrum file path.
		wavelength_range: list [nm]
			wavelength range of experiment data, dim=(2,).
		fit_names: list of str
			names of parameters to fit, possible names include: 'eps_bg', 'f1', 'exciton1', 'gamma1', 'f2', 'exciton2', 'gamma2', 'xy_re', 'xy_im', 'thickness_nm'.
		parameters: dict
			dictionary of parameters that are possible to fit, for names included in fit_names, with initial values; and for others, fixed values.
		phi: float [rad]
			incident angle, assumed to be in x-z plane.
		d: list [nm]
			thickness of each non-excitonic layer, assumed to be surrounded by air and Silicon substrate.
		eps_list: list [1]
			list of relative dielectric tensors for non-excitonic layers, dim=(num_layers-1, 3, 3).
		i: int
			index of the excitonic layer, start with 1.
		mask_value: list [nm]
			wavelength range during fitting.
	'''
	def __init__(self, file_path, wavelength_range, fit_names, parameters, phi=0, d=[], eps_list=[], i=1, mask_value=None, polarization='y'):
		self.fit_names=fit_names
		self.parameters=parameters
		self.phi=phi
		self.d=d
		self.eps_list=eps_list
		self.i=i
		self.mask_value=mask_value
		self.polarization=polarization
		self.num_layers=len(d)+1

		# obtain experiment data from file
		with open(file_path, 'r') as file:
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

		# wavelength range
		self.wlnm=len(reflectance_data)
		wavelength=np.linspace(wavelength_range[0], wavelength_range[1], self.wlnm)
		self.wavelength=wavelength
		self.hbar_omega=constants.h*3e8/self.wavelength/1e-9/constants.e # [eV]

		# obtain reference reflectance
		self.reference_reflectance=self.reference()

	def reference(self):
		"""
		Calculate reference reflectance without excitonic layer.
		"""
		n = self.num_layers-1
		wlnm = self.wlnm
		eps = np.zeros((n, 3, 3, wlnm), dtype=complex)
		for layer in range(n):
			eps_layer = self.eps_list[layer]
			eps[layer] = np.array([[[eps_layer[0,0]]*wlnm, [eps_layer[0,1]]*wlnm, [eps_layer[0,2]]*wlnm],
								  [[eps_layer[1,0]]*wlnm, [eps_layer[1,1]]*wlnm, [eps_layer[1,2]]*wlnm],
								  [[eps_layer[2,0]]*wlnm, [eps_layer[2,1]]*wlnm, [eps_layer[2,2]]*wlnm]])
		d_ref = self.d.copy()
		d_ref = np.array(d_ref)
		tmm = anisotropicTMM(eps, d_ref, self.wavelength, self.phi)
		R_ref = tmm.reflectance(polarization=self.polarization)
		return R_ref
	
	def lorentz_eps(self, parameters):
		"""
		Lorentz model for dielectric function.
		"""
		eps_bg = parameters['eps_bg']
		f1 = parameters['f1']
		exciton1 = parameters['exciton1']
		gamma1 = parameters['gamma1']
		f2 = parameters['f2']
		exciton2 = parameters['exciton2']
		gamma2 = parameters['gamma2']
		hbar_omega= self.hbar_omega
		return eps_bg + f1 / (exciton1**2 - hbar_omega**2 - 1j * gamma1 * hbar_omega) + f2 / (exciton2**2 - hbar_omega**2 - 1j * gamma2 * hbar_omega)

	def build_epsilon_tensor(self, parameters):
		"""
		anisotropic epsilon tensor with i-th layer having Lorentz-model-type yy component and off-diagonal components xy, yx.
		"""
		num_layers = self.num_layers
		i = self.i
		wlnm = self.wlnm
		eps = np.zeros((num_layers, 3, 3, wlnm), dtype=complex)
		for layer in range(num_layers):
			if layer == i-1:
				# Lorentz model in yy component
				yy = self.lorentz_eps(parameters)
				xy = np.array([parameters["xy_re"] + 1j*parameters["xy_im"]]*wlnm)
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


	def reflectance_from_params(self, parameters):
		'''
		Calculate reflectance from given parameters.
		'''
		epsilon = self.build_epsilon_tensor(parameters)
		d = self.d.copy()
		d.insert(self.i-1, parameters["thickness_nm"])
		d = np.array(d)
		tmm = anisotropicTMM(epsilon, d, self.wavelength, self.phi)
		R = tmm.reflectance(polarization=self.polarization)
		return R

	def residuals(self, params, use_log=True):
		'''
		Calculate residuals between model and experiment.
		'''
		parameters = self.full_params(params)
		R_model = self.reflectance_from_params(parameters)
		ref = self.reference_reflectance
		y = self.reflectance_data
		model = R_model/ref

		if use_log:
			resid = np.log(y) - np.log(model)
		else:
			resid = y - model
		if self.mask_value is not None:
			mask = (self.wavelength > self.mask_value[0]) & (self.wavelength < self.mask_value[1])
			resid = resid[mask]
		return resid
	
	def full_params(self, params):
		'''
		Convert the input params that are under fitting into full parameters dict.
		'''
		parameters = dict(self.parameters)
		for name, val in zip(self.fit_names, params):
			parameters[name] = float(val)
		return parameters

	def _mad(self, x):
		'''
		Median Absolute Deviation for f_scale evaluation
		'''
		med = np.median(x)
		return np.median(np.abs(x - med)) * 1.4826

	def fit_exciton_and_offdiagonals(self, lower_bound, upper_bound):
		'''
		Fit exciton parameters, layer thicknesses and off-diagonal components.
		'''
		# setting p0, lower, upper, f_scale, x_scale
		p0 = [self.parameters[name] for name in self.fit_names]
		lower = [lower_bound[name] for name in self.fit_names]
		upper = [upper_bound[name] for name in self.fit_names]
		r0 = self.residuals(p0, use_log=True)
		f_scale = max(self._mad(r0), 1e-3)
		x_scale = np.maximum(np.abs(p0), 1.0)

		# least squares fitting
		res = least_squares(
			lambda x: self.residuals(x, use_log=True),
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

		# print summary
		m = res.fun.size
		cost = res.cost
		residual_norm = np.sqrt(2*cost) if cost is not None else np.nan
		rmse = np.sqrt(np.sum(res.fun**2)/m) if m>0 else np.nan
		print('\n=== Fit summary ===')
		print('success:', res.success)
		print('message:', res.message)
		print('cost (0.5 * sum(f^2)):', cost)
		print('residual norm ||f||_2:', residual_norm)
		print('RMSE:', rmse)
		print('fitted parameters:', {n: float(v) for n, v in zip(self.fit_names, res.x)})
		print('===================\n')

		# plot
		parameters = self.full_params(res.x)
		R_model = self.reflectance_from_params(parameters)
		ref = self.reference_reflectance
		y = self.reflectance_data
		plt.figure(figsize=(6,4))
		plt.plot(self.wavelength, y*ref, color='red', label='experiment')
		plt.plot(self.wavelength, R_model, color='blue', label='TMM fit')
		plt.xlabel('wavelength (nm)')
		plt.ylabel('reflectance')
		plt.legend()
		plt.tight_layout()
		plt.show()
		return res


time_start=time.time()
eps_siO2 = np.array([[1.45**2, 0, 0], [0, 1.45**2, 0], [0, 0, 1.45**2]])  # SiO2
fit=lorentz_fit(
	file_path='./data/new_0T-2.asc',
	wavelength_range=[870,950],
	fit_names=['eps_bg','f1','exciton1','gamma1','f2','exciton2','gamma2','xy_re', 'xy_im', 'thickness_nm'],
	parameters={'eps_bg': 11, 'f1': 1.32, 'exciton1': 1.3670, 'gamma1': 1.52e-3, 'f2': 0.25, 'exciton2': 1.3814, 'gamma2': 6.5e-3, 'xy_re': 0, 'xy_im': 0, 'thickness_nm': 38},
	phi=0,
	d=[280],
	eps_list=[eps_siO2],
	i=1,
	mask_value=[890, 951],
	polarization='y'
)
fit.fit_exciton_and_offdiagonals(
	lower_bound={'eps_bg':1, 'f1': 0, 'exciton1': 1, 'gamma1': -1, 'f2': 0, 'exciton2': 1, 'gamma2': -1, 'xy_re': -2, 'xy_im': -2, 'thickness_nm': 30},
	upper_bound={'eps_bg':20, 'f1': 10, 'exciton1': 2, 'gamma1': 1, 'f2': 10, 'exciton2': 2, 'gamma2': 1, 'xy_re': 2, 'xy_im': 2, 'thickness_nm': 45}
)
time_end=time.time()
print('time cost:', time_end-time_start, 's')