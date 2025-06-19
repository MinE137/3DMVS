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

class HydrogenOrbital:
    ORBITAL_MAP = {
        's': (1, 0, 0),
        'pz': (2, 1, 0),
        'px': (2, 1, 1),
        'py': (2, 1, -1),
    }
    def __init__(self, orb_type, HYDROGEN_ORBITAL_RADII, ELEMENT_ORBITAL_RADII, atom_symbol):
        n, l, m = self.ORBITAL_MAP[orb_type]
        self.n = n
        self.l = l
        self.m = m
        self.grid_size = 50
        self.grid_range = 10.0
        self.x = np.linspace(-self.grid_range, self.grid_range, self.grid_size)
        self.y = np.linspace(-self.grid_range, self.grid_range, self.grid_size)
        self.z = np.linspace(-self.grid_range, self.grid_range, self.grid_size)
        self.X, self.Y, self.Z = np.meshgrid(self.x, self.y, self.z, indexing='ij')
        self.R = np.sqrt(self.X**2 + self.Y**2 + self.Z**2)
        self.PHI = np.arctan2(self.Y, self.X)
        self.THETA = np.arccos(np.clip(self.Z / (self.R + 1e-10), -1, 1))
        # 오비탈 크기 축적: 수소 오비탈 크기(Å) → 선택 원소 오비탈 크기(Å)로 비율 조정
        self.hydrogen_r = HYDROGEN_ORBITAL_RADII[orb_type]
        self.element_r = ELEMENT_ORBITAL_RADII[atom_symbol][orb_type]
        self.scale = self.element_r / self.hydrogen_r

    def radial_wavefunc(self, r):
        a0 = 0.529  # Bohr 반지름(Å)
        r_bohr = r / a0
        if self.n == 1 and self.l == 0:
            return 2 * np.exp(-r_bohr)
        elif self.n == 2 and self.l == 0:
            return (1 - r_bohr/2) * np.exp(-r_bohr/2) / (2 * np.sqrt(2))
        elif self.n == 2 and self.l == 1:
            return r_bohr * np.exp(-r_bohr/2) / (2 * np.sqrt(6))
        else:
            return np.zeros_like(r)

    def angular_wavefunc(self, theta, phi):
        from scipy.special import sph_harm_y
        if self.l == 0 and self.m == 0:
            return sph_harm_y(0, 0, theta, phi).real
        elif self.l == 1:
            if self.m == 0:
                return sph_harm_y(1, 0, theta, phi).real  # pz
            elif self.m == 1:
                return np.sqrt(2) * sph_harm_y(1, 1, theta, phi).real  # px
            elif self.m == -1:
                return np.sqrt(2) * sph_harm_y(1, 1, theta, phi).imag  # py
        else:
            return np.zeros_like(theta)

    def wavefunc(self):
        R_part = self.radial_wavefunc(self.R)
        Y_part = self.angular_wavefunc(self.THETA, self.PHI)
        psi = R_part * Y_part
        return psi


class MoleculeApp(QWidget):
    def analyze_hybridization(self):
        if not hasattr(self, "molecule") or self.selected_atom_idx is None or self.atom_positions is None:
            self.hybridization_info = None
            return

        atom = self.molecule.atoms[self.selected_atom_idx]
        symbol = atom.symbol

    # (1) 결합수: 다중결합도 1로 계산
        bond_count = len(atom.neighbors)
        if bond_count == 0:
            self.hybridization_info = None
            return

    # (3) 족 번호 추출
        prop = ELEMENT_PROPERTIES.get(symbol)
        group_number = prop[-2] if prop and len(prop) >= 2 else None

    # (4) get_lone_pair_count로 lone_pairs 계산
        if group_number is not None:
            lone_pairs = get_lone_pair_count(group_number, bond_count)
        else:
            lone_pairs = 0

    # (5) SN 계산: 결합수(다중결합도 1로) + lone_pairs
        SN = bond_count + lone_pairs

    # (6) 구조/혼성화 판정은 기존 방식대로
        if SN <= 1:
            structure, hybrid = None, None
        elif SN == 2:
            structure, hybrid = '직선형', 'sp'
        elif SN == 3 and lone_pairs == 0:
            structure, hybrid = '평면삼각형', 'sp2'
        elif SN == 3 and lone_pairs == 1:
            structure, hybrid = '굽은형1', 'sp2'
        elif SN == 4 and lone_pairs == 0:
            structure, hybrid = '사면체형', 'sp3'
        elif SN == 4 and lone_pairs == 1:
            structure, hybrid = '삼각뿔형', 'sp3'
        elif SN == 4 and lone_pairs == 2:
            structure, hybrid = '굽은형2', 'sp3'
        else:
            structure, hybrid = None, None

        self.hybridization_info = {
        'atom_idx': self.selected_atom_idx,
        'symbol': symbol,
        'bond_count': bond_count,  # 다중결합도 1로 센 결합수
        'lone_pairs': lone_pairs,
        'SN': SN,
        'structure': structure,
        'hybrid': hybrid
    }

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