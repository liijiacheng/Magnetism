import numpy as np
from scipy import constants
import scipy.linalg as la
import matplotlib.pyplot as plt
from epsilon_data import epsilon as eps_data

class anisotropicTMM:
    '''
    Anisotropic Transfer Matrix Method.
    Parameters:
        epsilon: array
            complex dielectric constant tensor, dim=(num_layers, 3, 3, len(wavelength)).
        d: array [nm]
            thickness of each layer, assumed to be surrounded by air and Silicon substrate.
        wavelength: array [nm]
            wavelength of light.
        phi: float [rad]
            incident angle, assumed to be in x-z plane.
    '''
    def __init__(self, epsilon, d, wavelength, phi, substrate='Si'):
        self.epsilon = epsilon
        self.d = d * 1e-9 # nm to m
        self.wavelength = wavelength * 1e-9 # nm to m
        self.phi = phi
        self.num_layers = len(d)
        self.ratio_0 = np.sqrt(constants.epsilon_0 / constants.mu_0)
        self.alpha = np.sin(phi)
        self.beta = 0 # incident plane is x-z plane
        self.k0 = 2 * np.pi / self.wavelength # wavevector in vacuum
        self.wlnm = len(wavelength)
        self.eps_data = eps_data(wavelength=self.wavelength*1e9)
        self.air_refractive_index = self.eps_data.get_epsilon('Air', tensor=False)
        self.substrate_refractive_index = self.eps_data.get_epsilon(substrate, tensor=False)

    def propagate_matrix(self):
        '''
        Calculate the propagate matrixs for each layer.
        Returns:
            S: array
                propagate matrix, dim=(num_layers, 4, 4, len(wavelength)).
        '''
        S = np.zeros((self.num_layers, 4, 4, self.wlnm), dtype=complex)
        for i in range(self.num_layers):
            eps = self.epsilon[i]
            a = self.alpha
            b = self.beta
            S[i,:,:,:] = np.array([
                [-a*eps[2,0]/eps[2,2], -a*eps[2,1]/eps[2,2], a*b/eps[2,2]/self.ratio_0, (1-a**2/eps[2,2])/self.ratio_0],
                [-b*eps[2,0]/eps[2,2], -b*eps[2,1]/eps[2,2], -(1-b**2/eps[2,2])/self.ratio_0, -a*b/eps[2,2]/self.ratio_0],
                [(-a*b-eps[1,0]+eps[2,0]*eps[1,2]/eps[2,2])*self.ratio_0, (a**2-eps[1,1]+eps[2,1]*eps[1,2]/eps[2,2])*self.ratio_0, -b*eps[1,2]/eps[2,2], a*eps[1,2]/eps[2,2]],
                [(-b**2+eps[0,0]-eps[2,0]*eps[0,2]/eps[2,2])*self.ratio_0, (a*b+eps[0,1]-eps[0,2]*eps[2,1]/eps[2,2])*self.ratio_0, b*eps[0,2]/eps[2,2], -a*eps[0,2]/eps[2,2]]
            ])
        return S
    
    def transfer_matrix(self):
        '''
        Calculate the transfer matrixs for each layer.
        Returns:
            M: array
                transfer matrix, dim=(4, 4, len(wavelength)).
        '''
        kS = np.einsum('l,ijkl->ijkl', self.k0, self.propagate_matrix())
        kS = np.transpose(kS, (0, 3, 1, 2))  # (num_layers, len(wavelength), 4, 4)
        M = np.broadcast_to(np.eye(4, dtype=complex), (self.wlnm, 4, 4)).copy()
        for i in range(self.num_layers):
            d = self.d[i]
            kS_i = kS[i,:,:,:]  # (L,4,4)
            # in case scipy.linalg.expm does not support batched input, use the following line instead
            # M_i = np.stack([la.expm(1j * kS_i[l] * d) for l in range(self.wlnm)], axis=0)  # (L,4,4)
            M_i = la.expm(1j * kS_i * d)  # (L,4,4)
            M = M_i @ M
        return M
    
    def reflectance(self, polarization=None, M_precomputed=None):
        '''
        Calculate the reflectance.
        parameters:
            polarization: str
                'white_light', 'y', or 'x' to specify the polarization direction.
            M_precomputed: array
                precomputed transfer matrix, dim=(len(wavelength), 4, 4). If provided, skip transfer_matrix() calculation.
        Returns:
            r: array
                reflection coefficient matrix, dim=(len(wavelength), 2, 2).
        elif polarization='white_light':
            R: array
                reflectance for white light, dim=(len(wavelength),).
        elif polarization='y':
            R: array
                reflectance for linear polarized light along y axis, dim=(len(wavelength),).
        elif polarization='x':
            R: array
                reflectance for linear polarized light along x axis, dim=(len(wavelength),).
        '''
        L = self.wlnm

        ratio_air = self.air_refractive_index * self.ratio_0  # (L,)
        cos_phi_air = np.cos(self.phi)  # scalar
        A = np.zeros((L, 2, 2), dtype=complex)
        A[:, 0, 1] = -ratio_air * cos_phi_air
        A[:, 1, 0] =  ratio_air / cos_phi_air
        B = np.zeros((L, 2, 2), dtype=complex)
        B[:, 0, 1] =  ratio_air * cos_phi_air
        B[:, 1, 0] = -ratio_air / cos_phi_air

        ratio_sub = self.substrate_refractive_index * self.ratio_0  # (L,)
        cos_phi_sub = np.sqrt(1 - (self.air_refractive_index**2) * (self.alpha**2) / (self.substrate_refractive_index**2) + 0j)
        C = np.zeros((L, 2, 2), dtype=complex)
        C[:, 0, 1] = -ratio_sub * cos_phi_sub
        C[:, 1, 0] =  ratio_sub / cos_phi_sub

        M = M_precomputed if M_precomputed is not None else self.transfer_matrix() # (L,4,4)
        M11 = M[:, :2, :2]
        M12 = M[:, :2, 2:]
        M21 = M[:, 2:, :2]
        M22 = M[:, 2:, 2:]

        E = C @ M11 + (C @ M12) @ A - M21 - M22 @ A
        F = M21 + M22 @ B - C @ M11 - (C @ M12) @ B

        try:
            res = np.linalg.solve(F, E)  # (L,2,2)
            if not np.isfinite(res).all():
                raise np.linalg.LinAlgError("Non-finite result from batched solve")
        except Exception:
            import warnings
            warnings.warn(
                "Singular or ill-conditioned system in reflectance; using batched pseudo-inverse.",
                RuntimeWarning
            )
            res = np.linalg.pinv(F) @ E  # (L,2,2)

        r = res  # (L,2,2)
        if polarization=='white_light':
            R = np.sum(np.abs(r)**2, axis=(1, 2))/2
            return R
        elif polarization=='y':
            R = np.abs(r[:,0,1])**2 + np.abs(r[:,1,1])**2
            return R
        elif polarization=='x':
            R = np.abs(r[:,0,0])**2 + np.abs(r[:,1,0])**2
            return R
        return r