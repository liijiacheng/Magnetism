import numpy as np
from scipy.interpolate import PchipInterpolator

class epsilon:
    '''
    Get dielectric data for specified material and wavelength range. All materials are assumed isotropic.
    Parameters:
        wavelength: array [nm]
                wavelength range, dim=(L,).
    '''
    def __init__(self, wavelength):
        self.wavelength = wavelength * 1e-3  # nm to um
        self.wlnm = len(wavelength)

    def get_epsilon(self, material, tensor=True):
        '''
        Parameters:
            material: str
                material name, options available: 'Si', 'SiO2', 'Air'.
            tensor: bool
                whether to return dielectric tensor dim=(3, 3, L) or refractive index dim=(L,).
        '''
        self.material = material
        self.tensor = tensor
        if self.material == 'Si':
            return self.get_si_epsilon()
        elif self.material == 'SiO2':
            return self.get_sio2_epsilon()
        elif self.material == 'Air':
            return self.get_air_epsilon()
        else:
            raise ValueError("Unavailable material")

    def get_sio2_epsilon(self):
        # Source for SiO2 refractive index: https://refractiveindex.info/?shelf=main&book=SiO2&page=Malitson
        # valid wavelength range: 0.21-6.70 um
        n=np.sqrt(1+0.6961663/(1-(0.0684043/self.wavelength)**2)+0.4079426/(1-(0.1162414/self.wavelength)**2)+0.8974794/(1-(9.896161/self.wavelength)**2))
        if self.tensor:
            eps = np.einsum('ij,k->ijk', np.eye(3, dtype=complex), n**2)
            return eps
        else:
            return n

    def get_air_epsilon(self):
        # Source for Air refractive index: https://refractiveindex.info/?shelf=other&book=air&page=Ciddor
        # valid wavelength range: 0.23-1.69 um
        n=1+0.05792105/(238.0185-self.wavelength**-2)+0.00167917/(57.362-self.wavelength**-2)
        if self.tensor:
            eps = np.einsum('ij,k->ijk', np.eye(3, dtype=complex), n**2)
            return eps
        else:
            return n

    def get_si_epsilon(self):
        # Source for Si refractive index: https://refractiveindex.info/?shelf=main&book=Si&page=Wang-25C
        # valid wavelength range: 0.40-1.31 um
        wl_n, n, wl_k, k = self._read_nk_two_block('./data/Si-Wang-25C.csv')
        n_interp = PchipInterpolator(wl_n, n, extrapolate=True)
        k_interp = PchipInterpolator(wl_k, k, extrapolate=True)
        n_eval = n_interp(self.wavelength)
        k_eval = k_interp(self.wavelength)
        m = n_eval + 1j * k_eval
        if self.tensor:
            eps = np.einsum('ij,k->ijk', np.eye(3, dtype=complex), m**2)
            return eps
        else:
            return m

    def _read_nk_two_block(self, path):
        '''
        Read n/k data file with two sections:
        - first section starts with "wl,n", followed by "wavelength,n" data
        - separated by blank lines
        - second section starts with "wl,k", followed by "wavelength,k" data
        '''
        with open(path, 'r', encoding='utf-8') as f:
            content = f.read().strip()
        sections = [s for s in content.split('\n\n') if s.strip()]
        wl_n, n = self.parse_section(sections[0], 'n')
        wl_k, k = self.parse_section(sections[1], 'k')
        return wl_n, n, wl_k, k

    def parse_section(self, sec_text, expect):
        '''
        Parse a section of the n/k data file.
        '''
        lines = [ln.strip() for ln in sec_text.splitlines() if ln.strip()]
        header = lines[0].lower().replace(' ', '')
        if not header.startswith(f'wl,{expect}'):
            raise ValueError(f'Unexpected section header: {lines[0]!r}, expected "wl,{expect}"')
        wl_list = []
        val_list = []
        for ln in lines[1:]:
            parts = [p for p in ln.replace(';', ',').replace('\t', ',').split(',') if p.strip()]
            try:
                wl_list.append(float(parts[0]))
                val_list.append(float(parts[1]))
            except ValueError:
                continue
        wl_arr = np.asarray(wl_list, dtype=float)
        val_arr = np.asarray(val_list, dtype=float)
        order = np.argsort(wl_arr)
        return wl_arr[order], val_arr[order]