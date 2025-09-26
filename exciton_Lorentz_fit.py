import numpy as np
from anisotropicTMM import anisotropicTMM
from scipy.optimize import least_squares
import scipy.io
import os


def read_asc(filepath):
	"""
	简单读取两列或多列的 ASC 文件，假定第一列为波长（nm），第二列为反射率（或强度）。
	返回 wavelength (nm) 和 reflectance (linear, 0-1)
	"""
	data = np.loadtxt(filepath)
	if data.ndim == 1:
		data = data.reshape(-1, len(data))
	wl = data[:, 0]
	r = data[:, 1]
	# 如果反射率以百分比给出，转换
	if np.max(r) > 1.1:
		r = r / 100.0
	return wl, r


def lorentz_eps(omega, eps_inf, f, omega0, gamma):
	"""Lorentz振子模型（以角频率为单位 omega）
	eps(omega) = eps_inf + f * omega0**2 / (omega0**2 - omega**2 - i*gamma*omega)
	这里 f 为振幅（耦合强度/振子强度）。
	"""
	return eps_inf + f * (omega0**2) / (omega0**2 - omega**2 - 1j * gamma * omega)


def build_epsilon_tensor(wl_nm, eps_bg, lorentz_params, xy, yx):
	"""
	构建每个层的介电张量：假设只对某一层（索引1）使用各向异性张量，其他层为标量 eps_bg。
	lorentz_params: dict with keys eps_inf, f, omega0_nm, gamma_nm
	xy, yx: 复数或实数，张量的 off-diagonal 项

	返回 epsilon array shape=(num_layers, 3, 3, len(wl))
	假设结构为 [air(0), sample(1), substrate(2)]，三层示例；用户可修改 d 和 num_layers
	"""
	# 这里做最小化实现：三个层的例子
	num_layers = 3
	wl_m = wl_nm * 1e-9
	omega = 2 * np.pi * 3e8 / wl_m

	eps = np.zeros((num_layers, 3, 3, len(wl_nm)), dtype=complex)

	# air layer (index 0)
	eps[0, :, :, :] = np.eye(3)[:, :, None]

	# sample layer (index 1): yy 含 Lorentz，其他对角为 eps_bg
	eps_inf = lorentz_params.get('eps_inf', eps_bg)
	f = lorentz_params['f']
	omega0 = 2 * np.pi * 3e8 / (lorentz_params['omega0_nm'] * 1e-9)
	gamma = 2 * np.pi * 3e8 / (lorentz_params['gamma_nm'] * 1e-9)
	eps_yy = lorentz_eps(omega, eps_inf, f, omega0, gamma)
	for i in range(len(wl_nm)):
		eps[1, 0, 0, i] = eps_bg  # xx
		eps[1, 1, 1, i] = eps_yy[i]  # yy with exciton
		eps[1, 2, 2, i] = eps_bg  # zz
		eps[1, 0, 1, i] = xy
		eps[1, 1, 0, i] = yx

	# substrate (index 2)
	eps[2, :, :, :] = np.eye(3)[:, :, None] * eps_bg

	return eps


def reflectance_from_params(params, wl_nm, d_nm, measured_wl, measured_R, phi=0.0):
	"""
	params: array-like, 包含 [f, omega0_nm, gamma_nm, Re(xy), Im(xy), Re(yx), Im(yx), eps_bg]
	返回模型反射率（与 measured_wl 对齐）
	"""
	f, omega0_nm, gamma_nm, re_xy, im_xy, re_yx, im_yx, eps_bg = params
	xy = re_xy + 1j * im_xy
	yx = re_yx + 1j * im_yx

	lorentz_params = {'eps_inf': eps_bg, 'f': f, 'omega0_nm': omega0_nm, 'gamma_nm': gamma_nm}
	epsilon = build_epsilon_tensor(wl_nm, eps_bg, lorentz_params, xy, yx)
	d = np.array(d_nm)
	tmm = anisotropicTMM(epsilon, d, wl_nm, phi)
	R = tmm.reflectance(white_light=True)
	# 如果 measured_wl 与 wl_nm 不完全相同，做插值
	R_interp = np.interp(measured_wl, wl_nm, R)
	return R_interp


def residuals(params, wl_nm, d_nm, measured_wl, measured_R, phi=0.0):
	R_model = reflectance_from_params(params, wl_nm, d_nm, measured_wl, measured_R, phi)
	return R_model - measured_R


def fit_exciton_and_offdiagonals(asc_path, d_nm, wl_nm=None, phi=0.0, p0=None, bounds=None):
	"""
	读取 asc 文件并拟合激子参数以及 xy,yx 分量。
	asc_path: 文件路径
	d_nm: list of layer thicknesses in nm (长度应与 epsilon 中层数一致)
	wl_nm: 可选的模型计算波长数组。如果为 None，则使用测量波长作为模型波长。
	返回拟合结果 (res: OptimizeResult)
	"""
	measured_wl, measured_R = read_asc(asc_path)
	if wl_nm is None:
		wl_nm = measured_wl

	# 初始猜测
	if p0 is None:
		# f, omega0_nm, gamma_nm, Re(xy), Im(xy), Re(yx), Im(yx), eps_bg
		p0 = [0.5, measured_wl[np.argmax(measured_R)], 5.0, 0.0, 0.0, 0.0, 0.0, 2.5]
	if bounds is None:
		lower = [0, np.min(measured_wl)*0.8, 0.1, -1, -1, -1, -1, 1.0]
		upper = [10, np.max(measured_wl)*1.2, 50.0, 1, 1, 1, 1, 10.0]
		bounds = (lower, upper)

	res = least_squares(residuals, p0, bounds=bounds, args=(wl_nm, d_nm, measured_wl, measured_R, phi))
	return res


if __name__ == '__main__':
	# 简单示例：用户需修改 asc 路径和厚度
	asc_path = os.path.join(os.path.dirname(__file__), 'example.asc')
	if not os.path.exists(asc_path):
		print('请将测量的 .asc 文件放在同目录并命名为 example.asc，或修改脚本中的路径。')
	else:
		d_nm = [0, 100, 0]  # air/sample/substrate thicknesses
		res = fit_exciton_and_offdiagonals(asc_path, d_nm)
		print('拟合完成，结果：')
		print(res.x)