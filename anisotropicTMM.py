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
    def __init__(self, epsilon, d, wavelength, phi):
        self.epsilon = epsilon
        self.d = d * 1e-9 # nm to m
        self.wavelength = wavelength * 1e-9 # nm to m
        self.phi = phi
        self.num_layers = len(d)
        self.ratio_0 = np.sqrt(constants.epsilon_0 / constants.mu_0)
        self.alpha = np.sin(phi)
        self.beta = 0 # incident plane is x-z plane
        self.k0 = 2 * np.pi / self.wavelength # wavevector in vacuum

    def propagate_matrix(self):
        '''
        Calculate the propagate matrixs for each layer.
        Returns:
            S: array
                propagate matrix, dim=(num_layers, 4, 4, len(wavelength)).
        '''
        S = np.zeros((self.num_layers, 4, 4, len(self.wavelength)), dtype=complex)
        for i in range(self.num_layers):
            eps = self.epsilon[i]
            a = self.alpha
            b = self.beta
            S[i,:,:,:] = np.array([
                [-a*eps[2,0]/eps[2,2], -a*eps[2,1]/eps[2,2], a*b/eps[2,2]/self.ratio_0, (1-a**2/eps[2,2])/self.ratio_0],
                [-b*eps[2,0]/eps[2,2], -b*eps[2,1]/eps[2,2], -(1-b**2/eps[2,2])/self.ratio_0, -a*b/eps[2,2]/self.ratio_0],
                [(-a*b-eps[1,0]+eps[2,0]*eps[1,2]/eps[2,2])*self.ratio_0, (a**2-eps[1,1]+eps[2,1]*eps[1,2]/eps[2,2])*self.ratio_0, -b*eps[1,2]/eps[2,2], a*eps[1,2]/eps[2,2]],
                [(-b**2+eps[0,0]-eps[2,0]*eps[0,2]/eps[2,2])*self.ratio_0, (a*b+eps[0,1]-eps[2,1]*eps[0,2]/eps[2,2])*self.ratio_0, b*eps[0,2]/eps[2,2], -a*eps[0,2]/eps[2,2]]
            ]).reshape(4,4,len(self.wavelength))
        return S
    
    def transfer_matrix(self):
        '''
        Calculate the transfer matrixs for each layer.
        Returns:
            M: array
                transfer matrix, dim=(4, 4, len(wavelength)).
        '''
        kS = np.einsum('l,ijkl->ijkl', self.k0, self.propagate_matrix())
        M = np.zeros((4, 4, len(self.wavelength)), dtype=complex)
        for l in range(len(self.wavelength)):
            M[:,:,l] = np.eye(4, dtype=complex)
            for i in range(self.num_layers):
                d = self.d[i]
                kS_il = kS[i,:,:,l]
                # compute matrix exponential of (1j * kS_il * d)
                M_il = la.expm(1j * kS_il * d)
                M[:,:,l] = M_il @ M[:,:,l]
        return M
    
    def reflectance(self, white_light=False, linear_polarized=False):
        '''
        Calculate the reflectance.
        parameters:
            white_light: bool
                if True, average the reflection coefficients over the x and y direction.
        Returns:
            r: array
                reflection coefficient matrix, dim=(len(wavelength), 2, 2).
        or white_light=True:
            R: array
                reflectance, dim=(len(wavelength),).
        or linear_polarized=True:
            R: array
                reflectance for linear polarized light along y axis, dim=(len(wavelength),).
        '''
        A=np.array([
            [0, -self.ratio_0*np.cos(self.phi)],
            [self.ratio_0/np.cos(self.phi), 0]
        ])
        B=np.array([
            [0, self.ratio_0*np.cos(self.phi)],
            [-self.ratio_0/np.cos(self.phi), 0]
        ])
        cos_phi_s=np.sqrt(1-self.alpha**2/3.6**2)
        ratio_s = self.ratio_0 * 3.6
        C=np.array([
            [0, -ratio_s*cos_phi_s],
            [ratio_s/cos_phi_s, 0]
        ])
        M = self.transfer_matrix()
        r = np.zeros((len(self.wavelength), 2, 2), dtype=complex)
        for l in range(len(self.wavelength)):
            M_l = M[:,:,l]
            M11=M_l[:2,:2]
            M12=M_l[:2,2:]
            M21=M_l[2:,:2]
            M22=M_l[2:,2:]
            E = C @ M11 + C @ M12 @ A - M21 - M22 @ A
            F = M21 + M22 @ B - C @ M11 - C @ M12 @ B

            # try stable solve first; fallback to pseudo-inverse on failure
            try:
                res = np.linalg.solve(F, E)
                if not np.isfinite(res).all():
                    raise np.linalg.LinAlgError("Non-finite result from solve")
            except np.linalg.LinAlgError:
                import warnings
                warnings.warn(
                    "Singular or ill-conditioned matrix encountered in reflectance calculation; using pseudo-inverse.",
                    RuntimeWarning
                )
                res = np.linalg.pinv(F) @ E
            r[l] = res
        if white_light:
            R = np.sum(np.abs(r)**2, axis=(1, 2))/2
            return R
        if linear_polarized:
            R = np.abs(r[:,0,1])**2 + np.abs(r[:,1,1])**2
            return R
        return r