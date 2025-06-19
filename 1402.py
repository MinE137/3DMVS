import sys
import numpy as np
import pubchempy as pcp
import requests
from rdkit import Chem
from collections import Counter
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QHBoxLayout, QTextEdit, QLineEdit, QPushButton, QCheckBox, QLabel, QFrame
from PyQt5.QtGui import QFont, QPalette, QColor
from PyQt5.QtCore import Qt
import pyvista as pv
from pyvistaqt import QtInteractor
from itertools import combinations, permutations
import common
import _1403

class HydrogenOrbital:
    def generate_isosurfaces(self):
        psi = self.wavefunc()
        psi_real = np.real(psi)
        max_val = np.max(np.abs(psi_real))
        outer_level = max_val
        levels = []
        levels.append(outer_level)
        level = outer_level / 10
        while level > 0.01 * outer_level:
            levels.append(level)
            level /= 10
        import pyvista as pv
        grid = pv.ImageData()
        grid.dimensions = psi_real.shape
        grid.origin = (-self.grid_range, -self.grid_range, -self.grid_range)
        grid.spacing = (2*self.grid_range/(self.grid_size-1),) * 3
        grid.point_data['psi'] = psi_real.flatten(order='F')
        surfaces = []
        for i, level in enumerate(levels):
            for sign, color in [(1, 'red'), (-1, 'blue')]:
                surf = grid.contour([sign*level])
                if surf.n_points > 0:
                    surf.points *= self.scale
                    surfaces.append({
                        'type': 'outer' if i == 0 else 'subshell',
                        'surface': surf,
                        'level': sign*level,
                        'color': color
                    })
        return surfaces

def get_orbital_rotation(orb_type):
    if orb_type == 'pz':
        return np.eye(3)
    elif orb_type == 'px':
        return np.array([[0,0,1],[0,1,0],[1,0,0]])
    elif orb_type == 'py':
        return np.array([[1,0,0],[0,0,1],[0,1,0]])
    else:
        return np.eye(3)


def get_lone_pair_count(element_property_value, bond_count):
   
    v = element_property_value
    b = bond_count
    if v in (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 18):
        return 0
    if v == 15:
        if b == 1:
            return 2
        elif b == 2 or b == 3:
            return 1
        elif b == 4 or b == 5:
            return 0
    if v == 16:
        if b == 1 or b == 2:
            return 2
        elif b == 3 or b == 4:
            return 1
        elif b == 5 or b == 6:
            return 0
    if v == 17:
        if b == 1 or b == 2:
            return 3
        elif b == 3 or b == 4:
            return 2
        elif b == 5 or b == 6:
            return 1
        elif b == 7:
            return 0
    return 0

class MoleculeApp(QWidget):
    def on_atom_pick(self, picked_point, event):
        if not self.ao_checkbox.isChecked():
            return
        if self.atom_positions is None or len(self.atom_positions) == 0:
            return
        picked_pos = np.array(picked_point)
        dists = np.linalg.norm(self.atom_positions + self.view_offset - picked_pos, axis=1)
        idx = int(np.argmin(dists))
        self.selected_atom_idx = idx
        self.analyze_hybridization()  # 혼성화 정보 판정 추가
        self.redraw()