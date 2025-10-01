import numpy as np
from anisotropicTMM import anisotropicTMM
from epsilon_data import epsilon
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
			thickness of non-excitonic layers, assumed to be surrounded by air and Silicon substrate.
		layer_names: list str
			names of non-excitonic layers, assumed to be surrounded by air and Silicon substrate.
		i: int
			index of the excitonic layer, start with 1.
		mask_value: list [nm]
			wavelength range during fitting.
	'''
	def __init__(self, file_path, wavelength_range, fit_names, parameters, phi=0, d=[], layer_names=[], i=1, mask_value=None, polarization='y'):
		self.fit_names=fit_names
		self.parameters=parameters
		self.phi=phi
		self.d=d
		self.layer_names=layer_names
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

		# dielectric data cache
		self.eps_data = epsilon(wavelength=self.wavelength)
		self.eps_cache = {name: self.eps_data.get_epsilon(name, tensor=True) for name in self.layer_names}

		# obtain reference reflectance
		self.reference_reflectance=self.reference()

		# precompute left/right transfer matrices for fixed layers (do once)
		self._precompute()

	def reference(self):
		'''
		Calculate reference reflectance without excitonic layer.
		'''
		eps_ref = []
		for name in self.layer_names:
			eps_ref.append(self.eps_cache[name])
		eps_ref = np.array(eps_ref)
		d_ref = self.d.copy()
		d_ref = np.array(d_ref)
		tmm = anisotropicTMM(eps_ref, d_ref, self.wavelength, self.phi)
		R_ref = tmm.reflectance(polarization=self.polarization)
		return R_ref
	
	def lorentz_eps(self, parameters):
		'''
		Lorentz model for dielectric function.
		'''
		eps_bg = parameters['eps_bg']
		f1 = parameters['f1']
		exciton1 = parameters['exciton1']
		gamma1 = parameters['gamma1']
		f2 = parameters['f2']
		exciton2 = parameters['exciton2']
		gamma2 = parameters['gamma2']
		hbar_omega= self.hbar_omega
		return eps_bg + f1 / (exciton1**2 - hbar_omega**2 - 1j * gamma1 * hbar_omega) + f2 / (exciton2**2 - hbar_omega**2 - 1j * gamma2 * hbar_omega)

	def _precompute(self):
		'''
		Precompute M_left and M_right (L,4,4) for fixed layers outside the excitonic layer.
		'''
		L = self.wlnm

		left_names = self.layer_names[: self.i-1]
		left_d = self.d[: self.i-1]
		left_eps = np.array([self.eps_cache[nm] for nm in left_names])

		right_names = self.layer_names[self.i-1:]
		right_d = self.d[self.i-1:]
		right_eps = np.array([self.eps_cache[nm] for nm in right_names])

		self.M_left = np.broadcast_to(np.eye(4, dtype=complex), (L, 4, 4)).copy()
		self.M_right = np.broadcast_to(np.eye(4, dtype=complex), (L, 4, 4)).copy()
		# compute left product
		if left_names:
			left_d_arr = np.array(left_d)
			tmm_left = anisotropicTMM(left_eps, left_d_arr, self.wavelength, self.phi)
			self.M_left = tmm_left.transfer_matrix()  # (L,4,4)
		# compute right product
		if right_names:
			right_d_arr = np.array(right_d)
			tmm_right = anisotropicTMM(right_eps, right_d_arr, self.wavelength, self.phi)
			self.M_right = tmm_right.transfer_matrix()  # (L,4,4)


	def reflectance_from_params(self, parameters):
		'''
		Calculate reflectance from given parameters.
		'''
		# excitonic layer epsilon tensor
		wlnm = self.wlnm
		eps_i = np.zeros((1, 3, 3, wlnm), dtype=complex)
		yy = self.lorentz_eps(parameters)
		xy = np.array([parameters["xy_re"] + 1j*parameters["xy_im"]]*wlnm)
		yx = np.conj(xy)
		eps_i[0] = np.array([[[9]*wlnm, xy,      [0]*wlnm],
							 [yx,        yy,      [0]*wlnm],
							 [[0]*wlnm, [0]*wlnm, [5]*wlnm]])
		tmm_exi = anisotropicTMM(eps_i, np.array([parameters["thickness_nm"]]), self.wavelength, self.phi)
		M_i = tmm_exi.transfer_matrix()  # (L,4,4)
		M_total = self.M_right @ M_i @ self.M_left
		R = tmm_exi.reflectance(polarization=self.polarization, M_precomputed=M_total)
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
			loss='soft_l1',
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


# CrSBr-SiO2 fitting
time_start=time.time()
fit=lorentz_fit(
	file_path='./data/new_0T-2.asc',
	wavelength_range=[870,950],
	fit_names=['eps_bg', 'f1', 'exciton1', 'gamma1', 'f2', 'exciton2', 'gamma2', 'xy_re', 'xy_im', 'thickness_nm'],
	parameters={'eps_bg': 11, 'f1': 1.32, 'exciton1': 1.3670, 'gamma1': 1.52e-3, 'f2': 0.25, 'exciton2': 1.3814, 'gamma2': 6.5e-3, 'xy_re': 0, 'xy_im': 0, 'thickness_nm': 38},
	phi=0,
	d=[280],
	layer_names=['SiO2'],
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