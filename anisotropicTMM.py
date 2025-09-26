import numpy as np
from scipy import constants
import matplotlib.pyplot as plt

class anisotropicTMM:
    '''
    Anisotropic Transfer Matrix Method.
    Parameters:
        epsilon: array
            complex dielectric constant tensor, dim=(num_layers, 3, 3, len(wavelength)).
        d: list [nm]
            thickness of each layer, assumed to be surrounded by air.
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
        kS = np.linalg.einsum('l,ijkl->ijkl', self.k0, self.propagate_matrix())
        M = np.zeros((4, 4, len(self.wavelength)), dtype=complex)
        for l in range(len(self.wavelength)):
            M[:,:,l] = np.eye(4, dtype=complex)
            for i in range(self.num_layers):
                d = self.d[i]
                kS_il = kS[i,:,:,l]
                eigvals, eigvecs = np.linalg.eig(kS_il)
                M_il = np.diag(np.exp(1j * eigvals * d))
                M_il = eigvecs @ M_il @ np.linalg.inv(eigvecs)
                M[:,:,l] = M_il @ M[:,:,l]
        # kSd=np.einsum('ijkl,i->ijkl', kS_i, d)
        # M_lis=np.exp(1j * kSd)
        # M = M_lis[0]
        # for i in range(1,self.num_layers):
        #     M = np.einsum('jkl,kml->jml', M_lis[i], M)
        return M
    
    def reflectance(self, white_light=False):
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
        '''
        A=np.array([
            [0, -self.ratio_0*np.cos(self.phi)],
            [self.ratio_0/np.cos(self.phi), 0]
        ])
        B=np.array([
            [0, self.ratio_0*np.cos(self.phi)],
            [-self.ratio_0/np.cos(self.phi), 0]
        ])
        M=self.transfer_matrix()
        r = np.zeros((len(self.wavelength), 2, 2), dtype=complex)
        for l in range(len(self.wavelength)):
            M_l = M[:,:,l]
            M11=M_l[:2,:2]
            M12=M_l[:2,2:]
            M21=M_l[2:,:2]
            M22=M_l[2:,2:]
            E = A @ M11 + A @ M12 @ A - M21 - M22 @ A
            F = M21 + M22 @ B - A @ M11 - A @ M12 @ B
            if np.linalg.det(F) == 0:
                raise ValueError("Singular matrix encountered in reflectance calculation.")
            r[l::] = np.linalg.inv(F) @ E
        if white_light:
            R = np.sum(np.abs(r)**2, axis=(1, 2))/2
            return R
        return r