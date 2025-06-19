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

ELEMENT_PROPERTIES = {
    'H':  (1.20, 2.20, 13.6, 0.75, '#B0BEC5', ['1s'], 0.53, (False, False), '비금속', 1, False),
    'He': (1.40, None, 24.6, None, '#D9FFFF', ['1s'], 0.31, (False, False), '비금속', 2, False),
    'Li': (1.82, 0.98, 5.39, 0.62, '#CC80FF', ['1s','2s'], 1.52, (False, False), '금속', 3, False),
    'Be': (1.53, 1.57, 9.32, 0.00, '#C2FF00', ['1s','2s'], 1.12, (False, False), '금속', 4, False),
    'B':  (1.92, 2.04, 8.30, 0.28, '#FFB5B5', ['1s','2s','2p'], 0.87, (False, False), '준금속', 5, False),
    'C':  (1.70, 2.55, 11.26, 1.26, '#222222', ['1s','2s','2p'], 0.68, (True, True), '비금속', 6, False),
    'N':  (1.55, 3.04, 14.53, -0.07, '#1976D2', ['1s','2s','2p'], 0.56, (True, True), '비금속', 7, False),
    'O':  (1.52, 3.44, 13.62, 1.46, '#EF5350', ['1s','2s','2p'], 0.48, (True, True), '비금속', 8, False),
    'F':  (1.47, 3.98, 17.42, 3.40, '#43A047', ['1s','2s','2p'], 0.42, (True, True), '비금속', 9, False),
    'Ne': (1.54, None, 21.56, None, '#B3E3F5', ['1s','2s','2p'], 0.38, (False, False), '비금속', 10, False),
    'Na': (2.27, 0.93, 5.14, 0.55, '#AB5CF2', ['1s','2s','2p','3s'], 1.86, (False, False), '금속', 11, False),
    'Mg': (1.73, 1.31, 7.65, 0.00, '#8AFF00', ['1s','2s','2p','3s'], 1.60, (False, False), '금속', 12, False),
    'Al': (1.84, 1.61, 5.99, 0.44, '#BFA6A6', ['1s','2s','2p','3s','3p'], 1.43, (False, False), '금속', 13, False),
    'Si': (2.10, 1.90, 8.15, 1.39, '#F0C8A0', ['1s','2s','2p','3s','3p'], 1.17, (False, False), '준금속', 14, False),
    'P':  (1.80, 2.19, 10.49, 0.75, '#FF8000', ['1s','2s','2p','3s','3p'], 1.10, (True, True), '비금속', 15, False),
    'S':  (1.80, 2.58, 10.36, 2.08, '#FDD835', ['1s','2s','2p','3s','3p'], 1.04, (True, True), '비금속', 16, False),
    'Cl': (1.75, 3.16, 12.97, 3.61, '#26A69A', ['1s','2s','2p','3s','3p'], 0.99, (True, True), '비금속', 17, False),
    'Ar': (1.88, None, 15.76, None, '#80D1E3', ['1s','2s','2p','3s','3p'], 0.71, (False, False), '비금속', 18, False),
    'K':  (2.75, 0.82, 4.34, 0.50, '#8F40D4', ['1s','2s','2p','3s','3p','4s'], 2.27, (False, False), '금속', 19, False),
    'Ca': (2.31, 1.00, 6.11, 0.00, '#3DFF00', ['1s','2s','2p','3s','3p','4s'], 1.97, (False, False), '금속', 20, False),
    'Sc': (2.30, 1.36, 6.56, 0.19, '#E6E6E6', ['1s','2s','2p','3s','3p','4s','3d'], 1.62, (False, True), '전이금속', 21, False),
    'Ti': (2.15, 1.54, 6.82, 0.08, '#BFC2C7', ['1s','2s','2p','3s','3p','4s','3d'], 1.47, (False, True), '전이금속', 22, False),
    'V':  (2.05, 1.63, 6.74, 0.53, '#A6A6AB', ['1s','2s','2p','3s','3p','4s','3d'], 1.34, (False, True), '전이금속', 23, False),
    'Cr': (2.05, 1.66, 6.77, 0.68, '#8A99C7', ['1s','2s','2p','3s','3p','4s','3d'], 1.28, (False, True), '전이금속', 24, False),
    'Mn': (2.05, 1.55, 7.43, 0.00, '#9C7AC7', ['1s','2s','2p','3s','3p','4s','3d'], 1.27, (False, True), '전이금속', 25, False),
    'Fe': (2.00, 1.83, 7.87, 0.15, '#E06633', ['1s','2s','2p','3s','3p','4s','3d'], 1.26, (False, True), '전이금속', 26, False),
    'Co': (2.00, 1.88, 7.86, 0.66, '#F090A0', ['1s','2s','2p','3s','3p','4s','3d'], 1.25, (False, True), '전이금속', 27, False),
    'Ni': (1.97, 1.91, 7.64, 1.16, '#50D050', ['1s','2s','2p','3s','3p','4s','3d'], 1.24, (False, True), '전이금속', 28, False),
    'Cu': (1.96, 1.90, 7.73, 1.24, '#C88033', ['1s','2s','2p','3s','3p','4s','3d'], 1.28, (False, True), '전이금속', 29, False),
    'Zn': (2.01, 1.65, 9.39, 0.00, '#7D80B0', ['1s','2s','2p','3s','3p','4s','3d'], 1.34, (False, True), '전이금속', 30, False),
    'Ga': (1.87, 1.81, 5.99, 0.30, '#C28F8F', ['1s','2s','2p','3s','3p','4s','3d','4p'], 1.35, (False, False), '금속', 31, False),
    'Ge': (2.11, 2.01, 7.90, 1.23, '#668F8F', ['1s','2s','2p','3s','3p','4s','3d','4p'], 1.22, (False, False), '준금속', 32, False),
    'As': (1.85, 2.18, 9.81, 0.81, '#BD80E3', ['1s','2s','2p','3s','3p','4s','3d','4p'], 1.19, (True, True), '준금속', 33, False),
    'Se': (1.90, 2.55, 9.75, 2.02, '#FFA100', ['1s','2s','2p','3s','3p','4s','3d','4p'], 1.16, (True, True), '비금속', 34, False),
    'Br': (1.85, 2.96, 11.81, 3.36, '#A1887F', ['1s','2s','2p','3s','3p','3d','4s','4p'], 1.14, (True, True), '비금속', 35, False),
    'Kr': (2.02, 3.00, 14.00, 0.00, '#5CB8D1', ['1s','2s','2p','3s','3p','3d','4s','4p'], 1.12, (False, False), '비금속', 36, False),
    'Rb': (3.03, 0.82, 4.18, 0.47, '#702EB0', ['1s','2s','2p','3s','3p','3d','4s','4p','5s'], 2.48, (False, False), '금속', 37, False),
    'Sr': (2.49, 0.95, 5.69, 0.00, '#00FF00', ['1s','2s','2p','3s','3p','3d','4s','4p','5s'], 2.15, (False, False), '금속', 38, False),
    'Y':  (2.36, 1.22, 6.38, 0.31, '#94FFFF', ['1s','2s','2p','3s','3p','3d','4s','4p','5s','4d'], 1.80, (False, True), '전이금속', 39, False),
    'Zr': (2.23, 1.33, 6.84, 0.43, '#94E0E0', ['1s','2s','2p','3s','3p','3d','4s','4p','5s','4d'], 1.60, (False, True), '전이금속', 40, False),
    'Nb': (2.18, 1.60, 6.88, 0.92, '#73C2C9', ['1s','2s','2p','3s','3p','3d','4s','4p','5s','4d'], 1.46, (False, True), '전이금속', 41, False),
    'Mo': (2.17, 2.16, 7.10, 0.75, '#54B5B5', ['1s','2s','2p','3s','3p','3d','4s','4p','5s','4d'], 1.39, (False, True), '전이금속', 42, False),
    'Tc': (2.16, 1.90, 7.28, 0.55, '#3B9E9C', ['1s','2s','2p','3s','3p','3d','4s','4p','5s','4d'], 1.36, (False, True), '전이금속', 43, True),
    'Ru': (2.13, 2.20, 7.36, 1.05, '#248F8F', ['1s','2s','2p','3s','3p','3d','4s','4p','5s','4d'], 1.34, (False, True), '전이금속', 44, False),
    'Rh': (2.10, 2.28, 7.46, 1.14, '#0A7D8C', ['1s','2s','2p','3s','3p','3d','4s','4p','5s','4d'], 1.34, (False, True), '전이금속', 45, False),
    'Pd': (2.10, 2.20, 8.34, 0.56, '#006985', ['1s','2s','2p','3s','3p','3d','4s','4p','5s','4d'], 1.37, (False, True), '전이금속', 46, False),
    'Ag': (1.72, 1.93, 7.58, 1.30, '#C0C0C0', ['1s','2s','2p','3s','3p','3d','4s','4p','5s','4d'], 1.44, (False, True), '전이금속', 47, False),
    'Cd': (1.58, 1.69, 8.99, 0.00, '#FFD98F', ['1s','2s','2p','3s','3p','3d','4s','4p','5s','4d'], 1.51, (False, True), '전이금속', 48, False),
    'In': (1.93, 1.78, 5.79, 0.30, '#A67573', ['1s','2s','2p','3s','3p','3d','4s','4p','5s','4d','5p'], 1.67, (False, False), '금속', 49, False),
    'Sn': (2.17, 1.96, 7.34, 1.11, '#668080', ['1s','2s','2p','3s','3p','3d','4s','4p','5s','4d','5p'], 1.58, (False, False), '금속', 50, False),
    'Sb': (2.06, 2.05, 8.61, 1.05, '#9E63B5', ['1s','2s','2p','3s','3p','3d','4s','4p','5s','4d','5p'], 1.45, (True, True), '준금속', 51, False),
    'Te': (2.06, 2.10, 9.01, 1.97, '#D47A00', ['1s','2s','2p','3s','3p','3d','4s','4p','5s','4d','5p'], 1.40, (True, True), '준금속', 52, False),
    'I':  (1.98, 2.66, 10.45, 3.06, '#8E24AA', ['1s','2s','2p','3s','3p','3d','4s','4p','5s','4d','5p'], 1.33, (True, True), '비금속', 53, False),
    'Xe': (2.16, 2.60, 12.13, 0.00, '#429EB0', ['1s','2s','2p','3s','3p','3d','4s','4p','5s','4d','5p'], 1.08, (False, False), '비금속', 54, False),
    'Cs': (3.43, 0.79, 3.89, 0.47, '#57178F', ['1s','2s','2p','3s','3p','3d','4s','4p','5s','4d','5p','6s'], 2.65, (False, False), '금속', 55, False),
    'Ba': (2.68, 0.89, 5.21, 0.00, '#00C900', ['1s','2s','2p','3s','3p','3d','4s','4p','5s','4d','5p','6s'], 2.22, (False, False), '금속', 56, False),
    'La': (2.50, 1.10, 5.58, 0.47, '#70D4FF', ['1s','2s','2p','3s','3p','3d','4s','4p','5s','4d','5p','6s','5d'], 1.87, (False, True), '란타넘족', 57, False),
    'Ce': (2.48, 1.12, 5.47, 0.50, '#FFFFC7', ['1s','2s','2p','3s','3p','3d','4s','4p','5s','4d','5p','6s','4f','5d'], 1.83, (False, True), '란타넘족', 58, False),
    'Pr': (2.47, 1.13, 5.42, 0.50, '#D9FFC7', ['1s','2s','2p','3s','3p','3d','4s','4p','5s','4d','5p','6s','4f'], 1.82, (False, True), '란타넘족', 59, False),
    'Nd': (2.45, 1.14, 5.53, 0.50, '#C7FFC7', ['1s','2s','2p','3s','3p','3d','4s','4p','5s','4d','5p','6s','4f'], 1.81, (False, True), '란타넘족', 60, False),
    'Pm': (2.43, 1.13, 5.55, 0.50, '#A3FFC7', ['1s','2s','2p','3s','3p','3d','4s','4p','5s','4d','5p','6s','4f'], 1.80, (False, True), '란타넘족', 61, True),
    'Sm': (2.42, 1.17, 5.64, 0.50, '#8FFFC7', ['1s','2s','2p','3s','3p','3d','4s','4p','5s','4d','5p','6s','4f'], 1.80, (False, True), '란타넘족', 62, False),
    'Eu': (2.40, 1.20, 5.67, 0.50, '#61FFC7', ['1s','2s','2p','3s','3p','3d','4s','4p','5s','4d','5p','6s','4f'], 1.99, (False, True), '란타넘족', 63, False),
    'Gd': (2.38, 1.20, 6.14, 0.50, '#45FFC7', ['1s','2s','2p','3s','3p','3d','4s','4p','5s','4d','5p','6s','4f','5d'], 1.79, (False, True), '란타넘족', 64, False),
    'Tb': (2.37, 1.20, 5.86, 0.50, '#30FFC7', ['1s','2s','2p','3s','3p','3d','4s','4p','5s','4d','5p','6s','4f'], 1.76, (False, True), '란타넘족', 65, False),
    'Dy': (2.35, 1.22, 5.94, 0.50, '#1FFFC7', ['1s','2s','2p','3s','3p','3d','4s','4p','5s','4d','5p','6s','4f'], 1.75, (False, True), '란타넘족', 66, False),
    'Ho': (2.33, 1.23, 6.02, 0.50, '#00FF9C', ['1s','2s','2p','3s','3p','3d','4s','4p','5s','4d','5p','6s','4f'], 1.74, (False, True), '란타넘족', 67, False),
    'Er': (2.32, 1.24, 6.10, 0.50, '#00E675', ['1s','2s','2p','3s','3p','3d','4s','4p','5s','4d','5p','6s','4f'], 1.73, (False, True), '란타넘족', 68, False),
    'Tm': (2.30, 1.25, 6.18, 0.50, '#00D452', ['1s','2s','2p','3s','3p','3d','4s','4p','5s','4d','5p','6s','4f'], 1.72, (False, True), '란타넘족', 69, False),
    'Yb': (2.28, 1.10, 6.25, 0.50, '#00BF38', ['1s','2s','2p','3s','3p','3d','4s','4p','5s','4d','5p','6s','4f'], 1.94, (False, True), '란타넘족', 70, False),
    'Lu': (2.27, 1.27, 5.43, 0.50, '#00AB24', ['1s','2s','2p','3s','3p','3d','4s','4p','5s','4d','5p','6s','4f','5d'], 1.72, (False, True), '란타넘족', 71, False),
    'Hf': (2.25, 1.30, 6.65, 0.00, '#4DC2FF', ['1s','2s','2p','3s','3p','3d','4s','4p','5s','4d','5p','6s','4f','5d'], 1.59, (False, True), '전이금속', 72, False),
    'Ta': (2.20, 1.50, 7.89, 0.31, '#4DA6FF', ['1s','2s','2p','3s','3p','3d','4s','4p','5s','4d','5p','6s','4f','5d'], 1.46, (False, True), '전이금속', 73, False),
    'W':  (2.18, 2.36, 7.98, 0.82, '#2194D6', ['1s','2s','2p','3s','3p','3d','4s','4p','5s','4d','5p','6s','4f','5d'], 1.39, (False, True), '전이금속', 74, False),
    'Re': (2.17, 1.90, 7.88, 0.15, '#267DAB', ['1s','2s','2p','3s','3p','3d','4s','4p','5s','4d','5p','6s','4f','5d'], 1.37, (False, True), '전이금속', 75, False),
    'Os': (2.16, 2.20, 8.44, 1.10, '#266696', ['1s','2s','2p','3s','3p','3d','4s','4p','5s','4d','5p','6s','4f','5d'], 1.35, (False, True), '전이금속', 76, False),
    'Ir': (2.13, 2.20, 8.97, 1.57, '#175487', ['1s','2s','2p','3s','3p','3d','4s','4p','5s','4d','5p','6s','4f','5d'], 1.36, (False, True), '전이금속', 77, False),
    'Pt': (2.13, 2.28, 8.96, 2.13, '#D0D0E0', ['1s','2s','2p','3s','3p','3d','4s','4p','5s','4d','5p','6s','4f','5d'], 1.39, (False, True), '전이금속', 78, False),
    'Au': (2.14, 2.54, 9.23, 2.31, '#FFD123', ['1s','2s','2p','3s','3p','3d','4s','4p','5s','4d','5p','6s','4f','5d'], 1.44, (False, True), '전이금속', 79, False),
    'Hg': (1.55, 2.00, 10.44, 0.00, '#B8B8D0', ['1s','2s','2p','3s','3p','3d','4s','4p','4d','5s','5p','5d','6s'], 1.60, (False, False), '금속', 80, True),
    'Tl': (1.96, 1.62, 6.11, 0.20, '#A6544D', ['1s','2s','2p','3s','3p','3d','4s','4p','5s','4d','5p','6s','4f','5d','6p'], 1.71, (False, False), '금속', 81, False),
    'Pb': (2.02, 2.33, 7.42, 0.00, '#575961', ['1s','2s','2p','3s','3p','3d','4s','4p','5s','4d','5p','6s','4f','5d','6p'], 1.75, (False, False), '금속', 82, True),
    'Bi': (2.07, 2.02, 7.29, 0.94, '#9E4FB5', ['1s','2s','2p','3s','3p','3d','4s','4p','5s','4d','5p','6s','4f','5d','6p'], 1.70, (True, True), '금속', 83, True),
    'Po': (1.97, 2.00, 8.42, 1.90, '#AB5C00', ['1s','2s','2p','3s','3p','3d','4s','4p','5s','4d','5p','6s','4f','5d','6p'], 1.68, (True, True), '준금속', 84, True),
    'At': (2.02, 2.20, 9.65, 2.80, '#754F45', ['1s','2s','2p','3s','3p','3d','4s','4p','5s','4d','5p','6s','4f','5d','6p'], 1.43, (True, True), '준금속', 85, True),
    'Rn': (2.45, None, 10.75, 0.00, '#428296', ['1s','2s','2p','3s','3p','3d','4s','4p','5s','4d','5p','6s','4f','5d','6p'], 1.20, (False, False), '비금속', 86, True),
    'Fr': (3.50, 0.70, 4.07, 0.00, '#420066', ['1s','2s','2p','3s','3p','3d','4s','4p','5s','4d','5p','6s','4f','5d','6p','7s'], 3.10, (False, False), '금속', 87, True),
    'Ra': (2.83, 0.90, 5.28, 0.00, '#007D00', ['1s','2s','2p','3s','3p','3d','4s','4p','5s','4d','5p','6s','4f','5d','6p','7s'], 2.15, (False, False), '금속', 88, True),
    'Ac': (2.65, 1.10, 5.17, 0.00, '#70ABFA', ['1s','2s','2p','3s','3p','3d','4s','4p','5s','4d','5p','6s','4f','5d','6p','7s'], 1.95, (False, True), '악티늄족', 89, True),
    'Th': (2.50, 1.30, 6.31, 0.00, '#00BAFF', ['1s','2s','2p','3s','3p','3d','4s','4p','5s','4d','5p','6s','4f','5d','6p','7s','5f'], 1.80, (False, True), '악티늄족', 90, True),
    'Pa': (2.40, 1.50, 5.89, 0.00, '#00A1FF', ['1s','2s','2p','3s','3p','3d','4s','4p','5s','4d','5p','6s','4f','5d','6p','7s','5f'], 1.61, (False, True), '악티늄족', 91, True),
    'U':  (1.86, 1.38, 6.08, 0.00, '#008FFF', ['1s','2s','2p','3s','3p','3d','4s','4p','5s','4d','5p','6s','4f','5d','6p','7s','5f'], 1.58, (False, True), '악티늄족', 92, True),
    'Np': (2.25, 1.36, 6.27, 0.00, '#0080FF', ['1s','2s','2p','3s','3p','3d','4s','4p','5s','4d','5p','6s','4f','5d','6p','7s','5f'], 1.55, (False, True), '악티늄족', 93, True),
    'Pu': (2.20, 1.28, 6.03, 0.00, '#006BFF', ['1s','2s','2p','3s','3p','3d','4s','4p','5s','4d','5p','6s','4f','5d','6p','7s','5f'], 1.59, (False, True), '악티늄족', 94, True),
    'Am': (2.15, 1.13, 6.00, 0.00, '#545CF2', ['1s','2s','2p','3s','3p','3d','4s','4p','5s','4d','5p','6s','4f','5d','6p','7s','5f'], 1.73, (False, True), '악티늄족', 95, True),
    'Cm': (2.10, 1.28, 6.02, 0.00, '#785CE3', ['1s','2s','2p','3s','3p','3d','4s','4p','5s','4d','5p','6s','4f','5d','6p','7s','5f'], 1.74, (False, True), '악티늄족', 96, True),
    'Bk': (2.05, 1.30, 6.23, 0.00, '#8A4FE3', ['1s','2s','2p','3s','3p','3d','4s','4p','5s','4d','5p','6s','4f','5d','6p','7s','5f'], 1.75, (False, True), '악티늄족', 97, True),
    'Cf': (2.00, 1.30, 6.30, 0.00, '#A136D4', ['1s','2s','2p','3s','3p','3d','4s','4p','5s','4d','5p','6s','4f','5d','6p','7s','5f'], 1.76, (False, True), '악티늄족', 98, True),
    'Es': (1.95, 1.30, 6.42, 0.00, '#B31FD4', ['1s','2s','2p','3s','3p','3d','4s','4p','5s','4d','5p','6s','4f','5d','6p','7s','5f'], 1.77, (False, True), '악티늄족', 99, True),
    'Fm': (1.90, 1.30, 6.50, 0.00, '#B31FBA', ['1s','2s','2p','3s','3p','3d','4s','4p','5s','4d','5p','6s','4f','5d','6p','7s','5f'], 1.78, (False, True), '악티늄족', 100, True),
    'Md': (1.85, 1.30, 6.58, 0.00, '#B30DA6', ['1s','2s','2p','3s','3p','3d','4s','4p','5s','4d','5p','6s','4f','5d','6p','7s','5f'], 1.79, (False, True), '악티늄족', 101, True),
    'No': (1.80, 1.30, 6.65, 0.00, '#BD0D87', ['1s','2s','2p','3s','3p','3d','4s','4p','5s','4d','5p','6s','4f','5d','6p','7s','5f'], 1.80, (False, True), '악티늄족', 102, True),
    'Lr': (1.75, 1.30, 6.79, 0.00, '#C70066', ['1s','2s','2p','3s','3p','3d','4s','4p','5s','4d','5p','6s','4f','5d','6p','7s','5f','6d'], 1.82, (False, True), '악티늄족', 103, True),
    'Rf': (1.70, None, None, None, '#D9D9D9', ['1s','2s','2p','3s','3p','3d','4s','4p','5s','4d','5p','6s','4f','5d','6p','7s','5f','6d'], 1.60, (False, True), '전이금속', 104, True),
    'Db': (1.65, None, None, None, '#C7C7C7', ['1s','2s','2p','3s','3p','3d','4s','4p','5s','4d','5p','6s','4f','5d','6p','7s','5f','6d'], 1.59, (False, True), '전이금속', 105, True),
    'Sg': (1.60, None, None, None, '#B0B0B0', ['1s','2s','2p','3s','3p','3d','4s','4p','5s','4d','5p','6s','4f','5d','6p','7s','5f','6d'], 1.58, (False, True), '전이금속', 106, True),
    'Bh': (1.55, None, None, None, '#A0A0A0', ['1s','2s','2p','3s','3p','3d','4s','4p','5s','4d','5p','6s','4f','5d','6p','7s','5f','6d'], 1.57, (False, True), '전이금속', 107, True),
    'Hs': (1.50, None, None, None, '#909090', ['1s','2s','2p','3s','3p','3d','4s','4p','5s','4d','5p','6s','4f','5d','6p','7s','5f','6d'], 1.56, (False, True), '전이금속', 108, True),
    'Mt': (1.45, None, None, None, '#808080', ['1s','2s','2p','3s','3p','3d','4s','4p','5s','4d','5p','6s','4f','5d','6p','7s','5f','6d'], 1.55, (False, True), '전이금속', 109, True),
    'Ds': (1.40, None, None, None, '#707070', ['1s','2s','2p','3s','3p','3d','4s','4p','5s','4d','5p','6s','4f','5d','6p','7s','5f','6d'], 1.54, (False, True), '전이금속', 110, True),
    'Rg': (1.35, None, None, None, '#606060', ['1s','2s','2p','3s','3p','3d','4s','4p','5s','4d','5p','6s','4f','5d','6p','7s','5f','6d'], 1.53, (False, True), '전이금속', 111, True),
    'Cn': (1.30, None, None, None, '#505050', ['1s','2s','2p','3s','3p','3d','4s','4p','5s','4d','5p','6s','4f','5d','6p','7s','5f','6d'], 1.52, (False, False), '금속', 112, True),
    'Nh': (1.25, None, None, None, '#404040', ['1s','2s','2p','3s','3p','3d','4s','4p','5s','4d','5p','6s','4f','5d','6p','7s','5f','6d','7p'], 1.51, (False, False), '금속', 113, True),
    'Fl': (1.20, None, None, None, '#303030', ['1s','2s','2p','3s','3p','3d','4s','4p','5s','4d','5p','6s','4f','5d','6p','7s','5f','6d','7p'], 1.50, (False, False), '금속', 114, True),
    'Mc': (1.15, None, None, None, '#202020', ['1s','2s','2p','3s','3p','3d','4s','4p','5s','4d','5p','6s','4f','5d','6p','7s','5f','6d','7p'], 1.49, (False, False), '금속', 115, True),
    'Lv': (1.10, None, None, None, '#101010', ['1s','2s','2p','3s','3p','3d','4s','4p','5s','4d','5p','6s','4f','5d','6p','7s','5f','6d','7p'], 1.48, (False, False), '금속', 116, True),
    'Ts': (1.05, None, None, None, '#3B9E3B', ['1s','2s','2p','3s','3p','3d','4s','4p','5s','4d','5p','6s','4f','5d','6p','7s','5f','6d','7p'], 1.47, (True, False), '준금속', 117, True),
    'Og': (1.00, None, None, None, '#FFFFFF', ['1s','2s','2p','3s','3p','3d','4s','4p','5s','4d','5p','6s','4f','5d','6p','7s','5f','6d','7p'], 1.46, (False, False), '비금속', 118, True),
}

def atomic_number_to_group(Z: int) -> int:
    return ((Z - 1) % 18) + 1

ELEMENT_LIST = []
for sym, props in sorted(ELEMENT_PROPERTIES.items(), key=lambda x: x[1][9]):
    atomic_number = props[9]
    assert atomic_number is not None, f"{sym}의 원자번호가 없습니다."
    radius = props[0]
    color = props[4]
    classification = props[8]
    group = atomic_number_to_group(atomic_number)
    ELEMENT_LIST.append((sym, radius, group, classification, color))

KOR_TO_ENG = {
    "물": "water",
    "아세트산": "acetic acid",
    "포름산": "formic acid",
    "에탄올": "ethanol",
    "암모니아": "ammonia",
    "메탄": "methane",
    "벤젠": "benzene",
    "이산화탄소": "carbon dioxide",
    "이산화황": "sulfur dioxide",
    "이산화질소": "nitrogen dioxide",
    "삼산화황": "sulfur trioxide",
    "염산": "hydrochloric acid",
    "황산": "sulfuric acid",
    "질산": "nitric acid",
    "과산화수소": "hydrogen peroxide",
    "아세톤": "acetone",
    "포도당": "glucose",
    "자당": "sucrose",
    "요오드": "iodine",
    "염소": "chlorine",
    "브로민": "bromine",
    "암모늄": "ammonium",
    "질소": "nitrogen",
    "산소": "oxygen",
    "수소": "hydrogen",
    "탄소": "carbon",
    "황": "sulfur",
    "인": "phosphorus",
    "나트륨": "sodium",
    "칼륨": "potassium",
    "칼슘": "calcium",
    "마그네슘": "magnesium",
    "알루미늄": "aluminum",
    "철": "iron",
    "구리": "copper",
    "아연": "zinc",
    "납": "lead",
    "은": "silver",
    "금": "gold",
    "수은": "mercury",
    "플루오르": "fluorine",
    "네온": "neon",
    "아르곤": "argon",
    "크립톤": "krypton",
    "제논": "xenon",
    "라돈": "radon",
    "리튬": "lithium",
    "베릴륨": "beryllium",
    "붕소": "boron",
    "실리콘": "silicon",
    "스트론튬": "strontium",
    "바륨": "barium",
    "루비듐": "rubidium",
    "세슘": "cesium",
    "프랑슘": "francium",
    "라듐": "radium",
    "란타넘": "lanthanum",
    "세륨": "cerium",
    "프라세오디뮴": "praseodymium",
    "네오디뮴": "neodymium",
    "프로메튬": "promethium",
    "사마륨": "samarium",
    "유로퓸": "europium",
    "가돌리늄": "gadolinium",
    "테르븀": "terbium",
    "디스프로슘": "dysprosium",
    "홀뮴": "holmium",
    "에르븀": "erbium",
    "툴륨": "thulium",
    "이터븀": "ytterbium",
    "루테튬": "lutetium",
    "하프늄": "hafnium",
    "탄탈럼": "tantalum",
    "텅스텐": "tungsten",
    "레늄": "rhenium",
    "오스뮴": "osmium",
    "이리듐": "iridium",
    "백금": "platinum",
    "팔라듐": "palladium",
    "로듐": "rhodium",
    "루테늄": "ruthenium",
    "니켈": "nickel",
    "코발트": "cobalt",
    "망가니즈": "manganese",
    "크로뮴": "chromium",
    "바나듐": "vanadium",
    "니오븀": "niobium",
    "몰리브데넘": "molybdenum",
    "테크네튬": "technetium",
    "팔라듐": "palladium",
    "인듐": "indium",
    "주석": "tin",
    "안티몬": "antimony",
    "텔루륨": "tellurium",
    "비스무트": "bismuth",
    "폴로늄": "polonium",
    "아스타틴": "astatine",
    "라돈": "radon",
    "라듐": "radium",
    "악티늄": "actinium",
    "토륨": "thorium",
    "프로탁티늄": "protactinium",
    "우라늄": "uranium",
    "넵투늄": "neptunium",
    "플루토늄": "plutonium",
    "아메리슘": "americium",
    "퀴륨": "curium",
    "버클륨": "berkelium",
    "캘리포늄": "californium",
    "아인슈타이늄": "einsteinium",
    "페르뮴": "fermium",
    "멘델레븀": "mendelevium",
    "노벨륨": "nobelium",
    "로렌슘": "lawrencium",
    "러더포듐": "rutherfordium",
    "더브늄": "dubnium",
    "시보귬": "seaborgium",
    "보륨": "bohrium",
    "하슘": "hassium",
    "마이트너륨": "meitnerium",
    "다름슈타튬": "darmstadtium",
    "뢴트게늄": "roentgenium",
    "코페르니슘": "copernicium",
    "니호늄": "nihonium",
    "플레로븀": "flerovium",
    "모스코븀": "moscovium",
    "리버모륨": "livermorium",
    "테네신": "tennessine",
    "오가네손": "oganesson"
}


COMMON_NAMES = {
    "h2o": ("물", "Water"), "water": ("물", "Water"),
    "aceticacid": ("아세트산", "Acetic acid"), "ethanoicacid": ("아세트산", "Acetic acid"),
    "ch3cooh": ("아세트산", "Acetic acid"), "c2h4o2": ("아세트산", "Acetic acid"),
    "formicacid": ("포름산", "Formic acid"), "methanoicacid": ("포름산", "Formic acid"),
    "hcooh": ("포름산", "Formic acid"), "ethanol": ("에탄올", "Ethanol"),
    "c2h5oh": ("에탄올", "Ethanol"), "ammonia": ("암모니아", "Ammonia"),
    "nh3": ("암모니아", "Ammonia"), "methane": ("메탄", "Methane"),
    "ch4": ("메탄", "Methane"), "benzene": ("벤젠", "Benzene"),
    "c6h6": ("벤젠", "Benzene")
}

ELEMENT_ORBITAL_RADII = {
    'H':   {'s': 0.53},
    'He':  {'s': 0.31},
    'Li':  {'s': 1.67},
    'Be':  {'s': 1.12},
    'B':   {'s': 0.87, 'px': 0.80, 'py': 0.80, 'pz': 0.80},
    'C':   {'s': 0.67, 'px': 0.57, 'py': 0.57, 'pz': 0.57},
    'N':   {'s': 0.56, 'px': 0.51, 'py': 0.51, 'pz': 0.51},
    'O':   {'s': 0.48, 'px': 0.45, 'py': 0.45, 'pz': 0.45},
    'F':   {'s': 0.42, 'px': 0.40, 'py': 0.40, 'pz': 0.40},
    'Ne':  {'s': 0.38, 'px': 0.38, 'py': 0.38, 'pz': 0.38},
    'Na':  {'s': 1.90},
    'Mg':  {'s': 1.45},
    'Al':  {'s': 1.18, 'px': 1.18, 'py': 1.18, 'pz': 1.18},
    'Si':  {'s': 1.11, 'px': 1.11, 'py': 1.11, 'pz': 1.11},
    'P':   {'s': 0.98, 'px': 0.98, 'py': 0.98, 'pz': 0.98},
    'S':   {'s': 0.88, 'px': 0.88, 'py': 0.88, 'pz': 0.88},
    'Cl':  {'s': 0.79, 'px': 0.79, 'py': 0.79, 'pz': 0.79},
    'Ar':  {'s': 0.71, 'px': 0.71, 'py': 0.71, 'pz': 0.71},
    'K':   {'s': 2.43},
    'Ca':  {'s': 1.94},
    'Sc':  {'s': 1.70, 'dz2': 1.70, 'dxz': 1.70, 'dyz': 1.70, 'dxy': 1.70, 'dx2-y2': 1.70},
    'Ti':  {'s': 1.60, 'dz2': 1.60, 'dxz': 1.60, 'dyz': 1.60, 'dxy': 1.60, 'dx2-y2': 1.60},
    'V':   {'s': 1.53, 'dz2': 1.53, 'dxz': 1.53, 'dyz': 1.53, 'dxy': 1.53, 'dx2-y2': 1.53},
    'Cr':  {'s': 1.39, 'dz2': 1.39, 'dxz': 1.39, 'dyz': 1.39, 'dxy': 1.39, 'dx2-y2': 1.39},
    'Mn':  {'s': 1.39, 'dz2': 1.39, 'dxz': 1.39, 'dyz': 1.39, 'dxy': 1.39, 'dx2-y2': 1.39},
    'Fe':  {'s': 1.32, 'dz2': 1.32, 'dxz': 1.32, 'dyz': 1.32, 'dxy': 1.32, 'dx2-y2': 1.32},
    'Co':  {'s': 1.26, 'dz2': 1.26, 'dxz': 1.26, 'dyz': 1.26, 'dxy': 1.26, 'dx2-y2': 1.26},
    'Ni':  {'s': 1.24, 'dz2': 1.24, 'dxz': 1.24, 'dyz': 1.24, 'dxy': 1.24, 'dx2-y2': 1.24},
    'Cu':  {'s': 1.28, 'dz2': 1.28, 'dxz': 1.28, 'dyz': 1.28, 'dxy': 1.28, 'dx2-y2': 1.28},
    'Zn':  {'s': 1.39, 'dz2': 1.39, 'dxz': 1.39, 'dyz': 1.39, 'dxy': 1.39, 'dx2-y2': 1.39},
    'Ga':  {'s': 1.26, 'px': 1.26, 'py': 1.26, 'pz': 1.26},
    'Ge':  {'s': 1.22, 'px': 1.22, 'py': 1.22, 'pz': 1.22},
    'As':  {'s': 1.21, 'px': 1.21, 'py': 1.21, 'pz': 1.21},
    'Se':  {'s': 1.16, 'px': 1.16, 'py': 1.16, 'pz': 1.16},
    'Br':  {'s': 1.14, 'px': 1.14, 'py': 1.14, 'pz': 1.14},
    'Kr':  {'s': 1.10, 'px': 1.10, 'py': 1.10, 'pz': 1.10},
    'Rb':  {'s': 2.65},
    'Sr':  {'s': 2.19},
    'Y':   {'s': 2.12, 'dz2': 2.12, 'dxz': 2.12, 'dyz': 2.12, 'dxy': 2.12, 'dx2-y2': 2.12},
    'Zr':  {'s': 2.06, 'dz2': 2.06, 'dxz': 2.06, 'dyz': 2.06, 'dxy': 2.06, 'dx2-y2': 2.06},
    'Nb':  {'s': 1.98, 'dz2': 1.98, 'dxz': 1.98, 'dyz': 1.98, 'dxy': 1.98, 'dx2-y2': 1.98},
    'Mo':  {'s': 1.90, 'dz2': 1.90, 'dxz': 1.90, 'dyz': 1.90, 'dxy': 1.90, 'dx2-y2': 1.90},
    'Tc':  {'s': 1.83, 'dz2': 1.83, 'dxz': 1.83, 'dyz': 1.83, 'dxy': 1.83, 'dx2-y2': 1.83},
    'Ru':  {'s': 1.78, 'dz2': 1.78, 'dxz': 1.78, 'dyz': 1.78, 'dxy': 1.78, 'dx2-y2': 1.78},
    'Rh':  {'s': 1.73, 'dz2': 1.73, 'dxz': 1.73, 'dyz': 1.73, 'dxy': 1.73, 'dx2-y2': 1.73},
    'Pd':  {'s': 1.69, 'dz2': 1.69, 'dxz': 1.69, 'dyz': 1.69, 'dxy': 1.69, 'dx2-y2': 1.69},
    'Ag':  {'s': 1.65, 'dz2': 1.65, 'dxz': 1.65, 'dyz': 1.65, 'dxy': 1.65, 'dx2-y2': 1.65},
    'Cd':  {'s': 1.61, 'dz2': 1.61, 'dxz': 1.61, 'dyz': 1.61, 'dxy': 1.61, 'dx2-y2': 1.61},
    'In':  {'s': 1.56, 'px': 1.56, 'py': 1.56, 'pz': 1.56},
    'Sn':  {'s': 1.45, 'px': 1.45, 'py': 1.45, 'pz': 1.45},
    'Sb':  {'s': 1.33, 'px': 1.33, 'py': 1.33, 'pz': 1.33},
    'Te':  {'s': 1.23, 'px': 1.23, 'py': 1.23, 'pz': 1.23},
    'I':   {'s': 1.15, 'px': 1.15, 'py': 1.15, 'pz': 1.15},
    'Xe':  {'s': 1.08, 'px': 1.08, 'py': 1.08, 'pz': 1.08},
    'Cs':  {'s': 2.98},
    'Ba':  {'s': 2.53},
    'La':  {'s': 2.17, 'dz2': 2.17, 'dxz': 2.17, 'dyz': 2.17, 'dxy': 2.17, 'dx2-y2': 2.17},
    'Hf':  {'s': 2.08, 'dz2': 2.08, 'dxz': 2.08, 'dyz': 2.08, 'dxy': 2.08, 'dx2-y2': 2.08},
    'Ta':  {'s': 2.00, 'dz2': 2.00, 'dxz': 2.00, 'dyz': 2.00, 'dxy': 2.00, 'dx2-y2': 2.00},
    'W':   {'s': 1.93, 'dz2': 1.93, 'dxz': 1.93, 'dyz': 1.93, 'dxy': 1.93, 'dx2-y2': 1.93},
    'Re':  {'s': 1.88, 'dz2': 1.88, 'dxz': 1.88, 'dyz': 1.88, 'dxy': 1.88, 'dx2-y2': 1.88},
    'Os':  {'s': 1.85, 'dz2': 1.85, 'dxz': 1.85, 'dyz': 1.85, 'dxy': 1.85, 'dx2-y2': 1.85},
    'Ir':  {'s': 1.80, 'dz2': 1.80, 'dxz': 1.80, 'dyz': 1.80, 'dxy': 1.80, 'dx2-y2': 1.80},
    'Pt':  {'s': 1.77, 'dz2': 1.77, 'dxz': 1.77, 'dyz': 1.77, 'dxy': 1.77, 'dx2-y2': 1.77},
    'Au':  {'s': 1.74, 'dz2': 1.74, 'dxz': 1.74, 'dyz': 1.74, 'dxy': 1.74, 'dx2-y2': 1.74},
    'Hg':  {'s': 1.71, 'dz2': 1.71, 'dxz': 1.71, 'dyz': 1.71, 'dxy': 1.71, 'dx2-y2': 1.71},
    'Tl':  {'s': 1.56, 'px': 1.56, 'py': 1.56, 'pz': 1.56},
    'Pb':  {'s': 1.54, 'px': 1.54, 'py': 1.54, 'pz': 1.54},
    'Bi':  {'s': 1.43, 'px': 1.43, 'py': 1.43, 'pz': 1.43},
    'Po':  {'s': 1.35, 'px': 1.35, 'py': 1.35, 'pz': 1.35},
    'At':  {'s': 1.27, 'px': 1.27, 'py': 1.27, 'pz': 1.27},
    'Rn':  {'s': 1.20, 'px': 1.20, 'py': 1.20, 'pz': 1.20},
    'Fr':  {'s': 3.48},
    'Ra':  {'s': 2.83},
    'Ac':  {'s': 2.23, 'dz2': 2.23, 'dxz': 2.23, 'dyz': 2.23, 'dxy': 2.23, 'dx2-y2': 2.23},
    'Th':  {'s': 2.08, 'dz2': 2.08, 'dxz': 2.08, 'dyz': 2.08, 'dxy': 2.08, 'dx2-y2': 2.08},
    'Pa':  {'s': 2.00, 'dz2': 2.00, 'dxz': 2.00, 'dyz': 2.00, 'dxy': 2.00, 'dx2-y2': 2.00},
    'U':   {'s': 1.96, 'dz2': 1.96, 'dxz': 1.96, 'dyz': 1.96, 'dxy': 1.96, 'dx2-y2': 1.96},
    'Np':  {'s': 1.90, 'dz2': 1.90, 'dxz': 1.90, 'dyz': 1.90, 'dxy': 1.90, 'dx2-y2': 1.90},
    'Pu':  {'s': 1.87, 'dz2': 1.87, 'dxz': 1.87, 'dyz': 1.87, 'dxy': 1.87, 'dx2-y2': 1.87},
    'Am':  {'s': 1.80, 'dz2': 1.80, 'dxz': 1.80, 'dyz': 1.80, 'dxy': 1.80, 'dx2-y2': 1.80},
    'Cm':  {'s': 1.69, 'dz2': 1.69, 'dxz': 1.69, 'dyz': 1.69, 'dxy': 1.69, 'dx2-y2': 1.69},
    'Bk':  {'s': 1.68, 'dz2': 1.68, 'dxz': 1.68, 'dyz': 1.68, 'dxy': 1.68, 'dx2-y2': 1.68},
    'Cf':  {'s': 1.68, 'dz2': 1.68, 'dxz': 1.68, 'dyz': 1.68, 'dxy': 1.68, 'dx2-y2': 1.68},
    'Es':  {'s': 1.65, 'dz2': 1.65, 'dxz': 1.65, 'dyz': 1.65, 'dxy': 1.65, 'dx2-y2': 1.65},
    'Fm':  {'s': 1.67, 'dz2': 1.67, 'dxz': 1.67, 'dyz': 1.67, 'dxy': 1.67, 'dx2-y2': 1.67},
    'Md':  {'s': 1.73, 'dz2': 1.73, 'dxz': 1.73, 'dyz': 1.73, 'dxy': 1.73, 'dx2-y2': 1.73},
    'No':  {'s': 1.76, 'dz2': 1.76, 'dxz': 1.76, 'dyz': 1.76, 'dxy': 1.76, 'dx2-y2': 1.76},
    'Lr':  {'s': 1.61, 'dz2': 1.61, 'dxz': 1.61, 'dyz': 1.61, 'dxy': 1.61, 'dx2-y2': 1.61},
    'Rf':  {'s': 1.57, 'dz2': 1.57, 'dxz': 1.57, 'dyz': 1.57, 'dxy': 1.57, 'dx2-y2': 1.57},
    'Db':  {'s': 1.49, 'dz2': 1.49, 'dxz': 1.49, 'dyz': 1.49, 'dxy': 1.49, 'dx2-y2': 1.49},
    'Sg':  {'s': 1.43, 'dz2': 1.43, 'dxz': 1.43, 'dyz': 1.43, 'dxy': 1.43, 'dx2-y2': 1.43},
    'Bh':  {'s': 1.41, 'dz2': 1.41, 'dxz': 1.41, 'dyz': 1.41, 'dxy': 1.41, 'dx2-y2': 1.41},
    'Hs':  {'s': 1.34, 'dz2': 1.34, 'dxz': 1.34, 'dyz': 1.34, 'dxy': 1.34, 'dx2-y2': 1.34},
    'Mt':  {'s': 1.29, 'dz2': 1.29, 'dxz': 1.29, 'dyz': 1.29, 'dxy': 1.29, 'dx2-y2': 1.29},
    'Ds':  {'s': 1.28, 'dz2': 1.28, 'dxz': 1.28, 'dyz': 1.28, 'dxy': 1.28, 'dx2-y2': 1.28},
    'Rg':  {'s': 1.21, 'dz2': 1.21, 'dxz': 1.21, 'dyz': 1.21, 'dxy': 1.21, 'dx2-y2': 1.21},
    'Cn':  {'s': 1.22, 'dz2': 1.22, 'dxz': 1.22, 'dyz': 1.22, 'dxy': 1.22, 'dx2-y2': 1.22},
    'Fl':  {'s': 1.19, 'px': 1.19, 'py': 1.19, 'pz': 1.19},
    'Lv':  {'s': 1.18, 'px': 1.18, 'py': 1.18, 'pz': 1.18},
    'Ts':  {'s': 1.16, 'px': 1.16, 'py': 1.16, 'pz': 1.16},
    'Og':  {'s': 1.15, 'px': 1.15, 'py': 1.15, 'pz': 1.15},
}

PERIODIC_TABLE_GRID = [
    ["H",  "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "He"],
    ["Li","Be","","", "", "", "", "", "", "", "", "", "B", "C", "N", "O", "F", "Ne"],
    ["Na","Mg","","", "", "", "", "", "", "", "", "", "Al","Si","P", "S", "Cl","Ar"],
    ["K", "Ca","Sc","Ti","V", "Cr","Mn","Fe","Co","Ni","Cu","Zn","Ga","Ge","As","Se","Br","Kr"],
    ["Rb","Sr","Y", "Zr","Nb","Mo","Tc","Ru","Rh","Pd","Ag","Cd","In","Sn","Sb","Te","I", "Xe"],
    ["Cs","Ba","","Hf","Ta","W", "Re","Os","Ir","Pt","Au","Hg","Tl","Pb","Bi","Po","At","Rn"],
    ["Fr","Ra","","Rf","Db","Sg","Bh","Hs","Mt","Ds","Rg","Cn","Nh","Fl","Mc","Lv","Ts","Og"],
    ["", "", "La","Ce","Pr","Nd","Pm","Sm","Eu","Gd","Tb","Dy","Ho","Er","Tm","Yb","Lu",""],
    ["", "", "Ac","Th","Pa","U", "Np","Pu","Am","Cm","Bk","Cf","Es","Fm","Md","No","Lr",""]
]




HYDROGEN_ORBITAL_RADII = {
    's': 1.0, 'pz': 2.0, 'px': 2.0, 'py': 2.0
}

MOLECULE_CACHE = {}

def fetch_3d_sdf_and_iupac_any(inp):
    key = inp.strip().lower()
    if key in KOR_TO_ENG:
        inp = KOR_TO_ENG[key]
    if inp in MOLECULE_CACHE:
        return MOLECULE_CACHE[inp]
    tried = []
    for search_type in ['name', 'formula', 'smiles', 'cid']:
        try:
            if search_type == 'cid' and not inp.isdigit():
                continue
            comps = pcp.get_compounds(inp, search_type)
            if comps:
                comp = comps[0]
                result = _fetch_sdf_and_names(comp, inp, search_type)
                MOLECULE_CACHE[inp] = result
                return result
            tried.append(search_type)
        except Exception:
            tried.append(search_type)
    # --- Fallback for water (H2O) ---
    if inp.strip().lower() in ['h2o', 'water']:
        try:
            comp = pcp.Compound.from_cid(962)
            result = _fetch_sdf_and_names(comp, inp, 'cid')
            MOLECULE_CACHE[inp] = result
            return result
        except Exception:
            pass
    raise RuntimeError(f"PubChem에서 '{inp}'에 대응하는 화합물을 찾지 못함. 시도: {', '.join(tried)}")


def _fetch_sdf_and_names(comp, inp, input_type='name'):
    cid = comp.cid
    iupac_name = getattr(comp, 'iupac_name', None)
    formula = getattr(comp, 'molecular_formula', None)
    synonyms = set([
        inp.lower(),
        getattr(comp, 'iupac_name', '').lower(),
        getattr(comp, 'molecular_formula', '').upper(),
        getattr(comp, 'canonical_smiles', ''),
        getattr(comp, 'synonyms', [])[0].lower() if getattr(comp, 'synonyms', None) else '',
        str(cid)
    ])
    url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{cid}/SDF?record_type=3d"
    resp = requests.get(url, timeout=10)
    if resp.status_code != 200:
        raise RuntimeError(f"PubChem REST API 호출 실패: {resp.status_code}")
    sdf_text = resp.text
    if "$$$$" not in sdf_text:
        raise RuntimeError("다운로드한 데이터에 3D SDF 형식이 포함되어 있지 않음.")
    return sdf_text, iupac_name, formula, synonyms, input_type

def parse_mol(sdf_text):
    mol = Chem.MolFromMolBlock(sdf_text, removeHs=False)
    if mol is None:
        raise ValueError("RDKit이 SDF 텍스트를 파싱하지 못했습니다.")
    return mol

class Atom:
    def __init__(self, idx, symbol, pos):
        self.idx = idx
        self.symbol = symbol
        self.pos = np.array(pos)
        self.neighbors = set()
        self.is_center = False
        self.bond_count = 0

    def update_center_info(self):
        self.bond_count = len(self.neighbors)
        self.is_center = self.bond_count >= 2

class Bond:
    def __init__(self, idx1, idx2, order):
        self.idx1 = idx1
        self.idx2 = idx2
        self.order = int(round(order))

class Molecule:
    def __init__(self, mol):
        self.atoms = []
        self.bonds = []
        self.build_from_rdkit(mol)

    def build_from_rdkit(self, mol):
        conf = mol.GetConformer()
        for atom in mol.GetAtoms():
            idx = atom.GetIdx()
            pos = conf.GetAtomPosition(idx)
            self.atoms.append(Atom(idx, atom.GetSymbol(), [pos.x, pos.y, pos.z]))
        for bond in mol.GetBonds():
            i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            order = bond.GetBondTypeAsDouble()
            self.bonds.append(Bond(i, j, order))
            self.atoms[i].neighbors.add(j)
            self.atoms[j].neighbors.add(i)
        for atom in self.atoms:
            atom.update_center_info()

    def atom_summary(self):
        counts = Counter([a.symbol for a in self.atoms])
        lines = []
        for sym, cnt in sorted(counts.items(), key=lambda x: (-x[1], x[0])):
            prop = ELEMENT_PROPERTIES.get(sym)
            if prop:
                en = prop[1]
                r = prop[0]
                ie = prop[2]
                ea = prop[3]
            else:
                en = r = ie = ea = '?'
            lines.append(f"{sym}({cnt}): EN({en}), R({r}), IE1({ie}), EA({ea})")
        return "\n".join(lines)

    def get_positions(self):
        return np.array([a.pos for a in self.atoms])

    def get_symbols(self):
        return [a.symbol for a in self.atoms]

def build_meshes(molecule):
    atoms, bonds = [], []
    for atom in molecule.atoms:
        prop = ELEMENT_PROPERTIES.get(atom.symbol, None)
        color = prop[4] if prop and prop[4] else '#9E9E9E'
        r = 0.35 + 0.35 * ((prop[0] if prop and prop[0] else 1.5) - 1.0) / (2.2 - 1.0)
        atoms.append((pv.Sphere(radius=r, center=atom.pos.tolist(), theta_resolution=32, phi_resolution=32), color))
    for bond in molecule.bonds:
        p1, p2 = molecule.atoms[bond.idx1].pos, molecule.atoms[bond.idx2].pos
        color = {1: '#78909C', 2: '#AED581', 3: '#33691E'}.get(bond.order, '#78909C')
        bonds.append((pv.Cylinder(center=((p1+p2)/2).tolist(), direction=(p2-p1).tolist(), radius=0.13, height=np.linalg.norm(p2-p1), resolution=24), color))
    return atoms, bonds

def get_bond_info(molecule):
    bonds, centers, lengths = [], [], []
    for bond in molecule.bonds:
        i, j = bond.idx1, bond.idx2
        p1, p2 = molecule.atoms[i].pos, molecule.atoms[j].pos
        bonds.append((i, j))
        centers.append((p1+p2)/2)
        lengths.append(np.linalg.norm(p1-p2))
    angles = []
    for atom in molecule.atoms:
        neighbors = list(atom.neighbors)
        if len(neighbors) >= 2:
            for i in range(len(neighbors)):
                for j in range(i+1, len(neighbors)):
                    n1, n2 = neighbors[i], neighbors[j]
                    v1, v2 = molecule.atoms[n1].pos - atom.pos, molecule.atoms[n2].pos - atom.pos
                    center = atom.pos + (v1 + v2) / 4
                    angle = np.degrees(np.arccos(np.clip(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)), -1, 1)))
                    normal = np.cross(v1, v2)
                    if np.linalg.norm(normal) < 1e-6:
                        normal = np.array([0,0,1])
                    normal = normal / (np.linalg.norm(normal)+1e-8)
                    label_pos = center + normal * 0.5
                    angles.append((label_pos, angle))
    return bonds, centers, lengths, angles