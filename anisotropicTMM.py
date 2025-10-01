import numpy as np
from scipy import constants
import scipy.linalg as la
import matplotlib.pyplot as plt

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
    def __init__(self, epsilon, d, wavelength, phi, substrate_refractive_index=3.58):
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
        self.substrate_refractive_index = substrate_refractive_index

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
    
    def reflectance(self, polarization=None):
        '''
        Calculate the reflectance.
        parameters:
            polarization: str
                'white_light', 'y', or 'x' to specify the polarization direction.
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
        A=np.array([
            [0, -self.ratio_0*np.cos(self.phi)],
            [self.ratio_0/np.cos(self.phi), 0]
        ])
        B=np.array([
            [0, self.ratio_0*np.cos(self.phi)],
            [-self.ratio_0/np.cos(self.phi), 0]
        ])
        cos_phi_s=np.sqrt(1-self.alpha**2/self.substrate_refractive_index**2)
        ratio_s = self.ratio_0 * self.substrate_refractive_index
        C=np.array([
            [0, -ratio_s*cos_phi_s],
            [ratio_s/cos_phi_s, 0]
        ])
        M = self.transfer_matrix() # (L,4,4)
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