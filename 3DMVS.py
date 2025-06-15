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
    'H':  (1.20, 2.20, 13.6, 0.75, '#B0BEC5', ['1s'], 0.53, (False, False), '비금속', 1),
    'He': (1.40, None, 24.6, None, '#D9FFFF', ['1s'], 0.31, (False, False), '비금속', 2),
    'Li': (1.82, 0.98, 5.39, 0.62, '#CC80FF', ['1s','2s'], 1.52, (False, False), '금속', 3),
    'Be': (1.53, 1.57, 9.32, 0.00, '#C2FF00', ['1s','2s'], 1.12, (False, False), '금속', 4),
    'B':  (1.92, 2.04, 8.30, 0.28, '#FFB5B5', ['1s','2s','2p'], 0.87, (False, False), '준금속', 5),
    'C':  (1.70, 2.55, 11.26, 1.26, '#222222', ['1s','2s','2p'], 0.68, (True, True), '비금속', 6),
    'N':  (1.55, 3.04, 14.53, -0.07, '#1976D2', ['1s','2s','2p'], 0.56, (True, True), '비금속', 7),
    'O':  (1.52, 3.44, 13.62, 1.46, '#EF5350', ['1s','2s','2p'], 0.48, (True, True), '비금속', 8),
    'F':  (1.47, 3.98, 17.42, 3.40, '#43A047', ['1s','2s','2p'], 0.42, (True, True), '비금속', 9),
    'Ne': (1.54, None, 21.56, None, '#B3E3F5', ['1s','2s','2p'], 0.38, (False, False), '비금속', 10),
    'Na': (2.27, 0.93, 5.14, 0.55, '#AB5CF2', ['1s','2s','2p','3s'], 1.86, (False, False), '금속', 11),
    'Mg': (1.73, 1.31, 7.65, 0.00, '#8AFF00', ['1s','2s','2p','3s'], 1.60, (False, False), '금속', 12),
    'Al': (1.84, 1.61, 5.99, 0.44, '#BFA6A6', ['1s','2s','2p','3s','3p'], 1.43, (False, False), '금속', 13),
    'Si': (2.10, 1.90, 8.15, 1.39, '#F0C8A0', ['1s','2s','2p','3s','3p'], 1.17, (False, False), '준금속', 14),
    'P':  (1.80, 2.19, 10.49, 0.75, '#FF8000', ['1s','2s','2p','3s','3p'], 1.10, (True, True), '비금속', 15),
    'S':  (1.80, 2.58, 10.36, 2.08, '#FDD835', ['1s','2s','2p','3s','3p'], 1.04, (True, True), '비금속', 16),
    'Cl': (1.75, 3.16, 12.97, 3.61, '#26A69A', ['1s','2s','2p','3s','3p'], 0.99, (True, True), '비금속', 17),
    'Ar': (1.88, None, 15.76, None, '#80D1E3', ['1s','2s','2p','3s','3p'], 0.71, (False, False), '비금속', 18),
    'K':  (2.75, 0.82, 4.34, 0.50, '#8F40D4', ['1s','2s','2p','3s','3p','4s'], 2.27, (False, False), '금속', 19),
    'Ca': (2.31, 1.00, 6.11, 0.00, '#3DFF00', ['1s','2s','2p','3s','3p','4s'], 1.97, (False, False), '금속', 20),
    'Sc': (2.30, 1.36, 6.56, 0.19, '#E6E6E6', ['1s','2s','2p','3s','3p','4s','3d'], 1.62, (False, True), '전이금속', 21),
    'Ti': (2.15, 1.54, 6.82, 0.08, '#BFC2C7', ['1s','2s','2p','3s','3p','4s','3d'], 1.47, (False, True), '전이금속', 22),
    'V':  (2.05, 1.63, 6.74, 0.53, '#A6A6AB', ['1s','2s','2p','3s','3p','4s','3d'], 1.34, (False, True), '전이금속', 23),
    'Cr': (2.05, 1.66, 6.77, 0.68, '#8A99C7', ['1s','2s','2p','3s','3p','4s','3d'], 1.28, (False, True), '전이금속', 24),
    'Mn': (2.05, 1.55, 7.43, 0.00, '#9C7AC7', ['1s','2s','2p','3s','3p','4s','3d'], 1.27, (False, True), '전이금속', 25),
    'Fe': (2.00, 1.83, 7.87, 0.15, '#E06633', ['1s','2s','2p','3s','3p','4s','3d'], 1.26, (False, True), '전이금속', 26),
    'Co': (2.00, 1.88, 7.86, 0.66, '#F090A0', ['1s','2s','2p','3s','3p','4s','3d'], 1.25, (False, True), '전이금속', 27),
    'Ni': (1.97, 1.91, 7.64, 1.16, '#50D050', ['1s','2s','2p','3s','3p','4s','3d'], 1.24, (False, True), '전이금속', 28),
    'Cu': (1.96, 1.90, 7.73, 1.24, '#C88033', ['1s','2s','2p','3s','3p','4s','3d'], 1.28, (False, True), '전이금속', 29),
    'Zn': (2.01, 1.65, 9.39, 0.00, '#7D80B0', ['1s','2s','2p','3s','3p','4s','3d'], 1.34, (False, True), '전이금속', 30),
    'Ga': (1.87, 1.81, 5.99, 0.30, '#C28F8F', ['1s','2s','2p','3s','3p','4s','3d','4p'], 1.35, (False, False), '금속', 31),
    'Ge': (2.11, 2.01, 7.90, 1.23, '#668F8F', ['1s','2s','2p','3s','3p','4s','3d','4p'], 1.22, (False, False), '준금속', 32),
    'As': (1.85, 2.18, 9.81, 0.81, '#BD80E3', ['1s','2s','2p','3s','3p','4s','3d','4p'], 1.19, (True, True), '준금속', 33),
    'Se': (1.90, 2.55, 9.75, 2.02, '#FFA100', ['1s','2s','2p','3s','3p','4s','3d','4p'], 1.16, (True, True), '비금속', 34),
    'Br': (1.85, 2.96, 11.81, 3.36, '#A1887F', ['1s','2s','2p','3s','3p','3d','4s','4p'], 1.14, (True, True), '비금속', 35),
    'Kr': (2.02, 3.00, 14.00, 0.00, '#5CB8D1', ['1s','2s','2p','3s','3p','3d','4s','4p'], 1.12, (False, False), '비금속', 36),
    'Rb': (3.03, 0.82, 4.18, 0.47, '#702EB0', ['1s','2s','2p','3s','3p','3d','4s','4p','5s'], 2.48, (False, False), '금속', 37),
    'Sr': (2.49, 0.95, 5.69, 0.00, '#00FF00', ['1s','2s','2p','3s','3p','3d','4s','4p','5s'], 2.15, (False, False), '금속', 38),
    'Y':  (2.36, 1.22, 6.38, 0.31, '#94FFFF', ['1s','2s','2p','3s','3p','3d','4s','4p','5s','4d'], 1.80, (False, True), '전이금속', 39),
    'Zr': (2.23, 1.33, 6.84, 0.43, '#94E0E0', ['1s','2s','2p','3s','3p','3d','4s','4p','5s','4d'], 1.60, (False, True), '전이금속', 40),
    'Nb': (2.18, 1.60, 6.88, 0.92, '#73C2C9', ['1s','2s','2p','3s','3p','3d','4s','4p','5s','4d'], 1.46, (False, True), '전이금속', 41),
    'Mo': (2.17, 2.16, 7.10, 0.75, '#54B5B5', ['1s','2s','2p','3s','3p','3d','4s','4p','5s','4d'], 1.39, (False, True), '전이금속', 42),
    'Tc': (2.16, 1.90, 7.28, 0.55, '#3B9E9C', ['1s','2s','2p','3s','3p','3d','4s','4p','5s','4d'], 1.36, (False, True), '전이금속', 43),
    'Ru': (2.13, 2.20, 7.36, 1.05, '#248F8F', ['1s','2s','2p','3s','3p','3d','4s','4p','5s','4d'], 1.34, (False, True), '전이금속', 44),
    'Rh': (2.10, 2.28, 7.46, 1.14, '#0A7D8C', ['1s','2s','2p','3s','3p','3d','4s','4p','5s','4d'], 1.34, (False, True), '전이금속', 45),
    'Pd': (2.10, 2.20, 8.34, 0.56, '#006985', ['1s','2s','2p','3s','3p','3d','4s','4p','5s','4d'], 1.37, (False, True), '전이금속', 46),
    'Ag': (1.72, 1.93, 7.58, 1.30, '#C0C0C0', ['1s','2s','2p','3s','3p','3d','4s','4p','5s','4d'], 1.44, (False, True), '전이금속', 47),
    'Cd': (1.58, 1.69, 8.99, 0.00, '#FFD98F', ['1s','2s','2p','3s','3p','3d','4s','4p','5s','4d'], 1.51, (False, True), '전이금속', 48),
    'In': (1.93, 1.78, 5.79, 0.30, '#A67573', ['1s','2s','2p','3s','3p','3d','4s','4p','5s','4d','5p'], 1.67, (False, False), '금속', 49),
    'Sn': (2.17, 1.96, 7.34, 1.11, '#668080', ['1s','2s','2p','3s','3p','3d','4s','4p','5s','4d','5p'], 1.58, (False, False), '금속', 50),
    'Sb': (2.06, 2.05, 8.61, 1.05, '#9E63B5', ['1s','2s','2p','3s','3p','3d','4s','4p','5s','4d','5p'], 1.45, (True, True), '준금속', 51),
    'Te': (2.06, 2.10, 9.01, 1.97, '#D47A00', ['1s','2s','2p','3s','3p','3d','4s','4p','5s','4d','5p'], 1.40, (True, True), '준금속', 52),
    'I':  (1.98, 2.66, 10.45, 3.06, '#8E24AA', ['1s','2s','2p','3s','3p','3d','4s','4p','5s','4d','5p'], 1.33, (True, True), '비금속', 53),
    'Xe': (2.16, 2.60, 12.13, 0.00, '#429EB0', ['1s','2s','2p','3s','3p','3d','4s','4p','5s','4d','5p'], 1.08, (False, False), '비금속', 54),
    'Cs': (3.43, 0.79, 3.89, 0.47, '#57178F', ['1s','2s','2p','3s','3p','3d','4s','4p','5s','4d','5p','6s'], 2.65, (False, False), '금속', 55),
    'Ba': (2.68, 0.89, 5.21, 0.00, '#00C900', ['1s','2s','2p','3s','3p','3d','4s','4p','5s','4d','5p','6s'], 2.22, (False, False), '금속', 56),
    'La': (2.50, 1.10, 5.58, 0.47, '#70D4FF', ['1s','2s','2p','3s','3p','3d','4s','4p','5s','4d','5p','6s','5d'], 1.87, (False, True), '란타넘족', 57),
    'Ce': (2.48, 1.12, 5.47, 0.50, '#FFFFC7', ['1s','2s','2p','3s','3p','3d','4s','4p','5s','4d','5p','6s','4f','5d'], 1.83, (False, True), '란타넘족', 58),
    'Pr': (2.47, 1.13, 5.42, 0.50, '#D9FFC7', ['1s','2s','2p','3s','3p','3d','4s','4p','5s','4d','5p','6s','4f'], 1.82, (False, True), '란타넘족', 59),
    'Nd': (2.45, 1.14, 5.53, 0.50, '#C7FFC7', ['1s','2s','2p','3s','3p','3d','4s','4p','5s','4d','5p','6s','4f'], 1.81, (False, True), '란타넘족', 60),
    'Pm': (2.43, 1.13, 5.55, 0.50, '#A3FFC7', ['1s','2s','2p','3s','3p','3d','4s','4p','5s','4d','5p','6s','4f'], 1.80, (False, True), '란타넘족', 61),
    'Sm': (2.42, 1.17, 5.64, 0.50, '#8FFFC7', ['1s','2s','2p','3s','3p','3d','4s','4p','5s','4d','5p','6s','4f'], 1.80, (False, True), '란타넘족', 62),
    'Eu': (2.40, 1.20, 5.67, 0.50, '#61FFC7', ['1s','2s','2p','3s','3p','3d','4s','4p','5s','4d','5p','6s','4f'], 1.99, (False, True), '란타넘족', 63),
    'Gd': (2.38, 1.20, 6.14, 0.50, '#45FFC7', ['1s','2s','2p','3s','3p','3d','4s','4p','5s','4d','5p','6s','4f','5d'], 1.79, (False, True), '란타넘족', 64),
    'Tb': (2.37, 1.20, 5.86, 0.50, '#30FFC7', ['1s','2s','2p','3s','3p','3d','4s','4p','5s','4d','5p','6s','4f'], 1.76, (False, True), '란타넘족', 65),
    'Dy': (2.35, 1.22, 5.94, 0.50, '#1FFFC7', ['1s','2s','2p','3s','3p','3d','4s','4p','5s','4d','5p','6s','4f'], 1.75, (False, True), '란타넘족', 66),
    'Ho': (2.33, 1.23, 6.02, 0.50, '#00FF9C', ['1s','2s','2p','3s','3p','3d','4s','4p','5s','4d','5p','6s','4f'], 1.74, (False, True), '란타넘족', 67),
    'Er': (2.32, 1.24, 6.10, 0.50, '#00E675', ['1s','2s','2p','3s','3p','3d','4s','4p','5s','4d','5p','6s','4f'], 1.73, (False, True), '란타넘족', 68),
    'Tm': (2.30, 1.25, 6.18, 0.50, '#00D452', ['1s','2s','2p','3s','3p','3d','4s','4p','5s','4d','5p','6s','4f'], 1.72, (False, True), '란타넘족', 69),
    'Yb': (2.28, 1.10, 6.25, 0.50, '#00BF38', ['1s','2s','2p','3s','3p','3d','4s','4p','5s','4d','5p','6s','4f'], 1.94, (False, True), '란타넘족', 70),
    'Lu': (2.27, 1.27, 5.43, 0.50, '#00AB24', ['1s','2s','2p','3s','3p','3d','4s','4p','5s','4d','5p','6s','4f','5d'], 1.72, (False, True), '란타넘족', 71),
    'Hf': (2.25, 1.30, 6.65, 0.00, '#4DC2FF', ['1s','2s','2p','3s','3p','3d','4s','4p','5s','4d','5p','6s','4f','5d'], 1.59, (False, True), '전이금속', 72),
    'Ta': (2.20, 1.50, 7.89, 0.31, '#4DA6FF', ['1s','2s','2p','3s','3p','3d','4s','4p','5s','4d','5p','6s','4f','5d'], 1.46, (False, True), '전이금속', 73),
    'W':  (2.18, 2.36, 7.98, 0.82, '#2194D6', ['1s','2s','2p','3s','3p','3d','4s','4p','5s','4d','5p','6s','4f','5d'], 1.39, (False, True), '전이금속', 74),
    'Re': (2.17, 1.90, 7.88, 0.15, '#267DAB', ['1s','2s','2p','3s','3p','3d','4s','4p','5s','4d','5p','6s','4f','5d'], 1.37, (False, True), '전이금속', 75),
    'Os': (2.16, 2.20, 8.44, 1.10, '#266696', ['1s','2s','2p','3s','3p','3d','4s','4p','5s','4d','5p','6s','4f','5d'], 1.35, (False, True), '전이금속', 76),
    'Ir': (2.13, 2.20, 8.97, 1.57, '#175487', ['1s','2s','2p','3s','3p','3d','4s','4p','5s','4d','5p','6s','4f','5d'], 1.36, (False, True), '전이금속', 77),
    'Pt': (2.13, 2.28, 8.96, 2.13, '#D0D0E0', ['1s','2s','2p','3s','3p','3d','4s','4p','5s','4d','5p','6s','4f','5d'], 1.39, (False, True), '전이금속', 78),
    'Au': (2.14, 2.54, 9.23, 2.31, '#FFD123', ['1s','2s','2p','3s','3p','3d','4s','4p','5s','4d','5p','6s','4f','5d'], 1.44, (False, True), '전이금속', 79),
    'Hg': (1.55, 2.00, 10.44, 0.00, '#B8B8D0', ['1s','2s','2p','3s','3p','3d','4s','4p','4d','5s','5p','5d','6s'], 1.60, (False, False), '금속', 80),
    'Tl': (1.96, 1.62, 6.11, 0.20, '#A6544D', ['1s','2s','2p','3s','3p','3d','4s','4p','5s','4d','5p','6s','4f','5d','6p'], 1.71, (False, False), '금속', 81),
    'Pb': (2.02, 2.33, 7.42, 0.00, '#575961', ['1s','2s','2p','3s','3p','3d','4s','4p','5s','4d','5p','6s','4f','5d','6p'], 1.75, (False, False), '금속', 82),
    'Bi': (2.07, 2.02, 7.29, 0.94, '#9E4FB5', ['1s','2s','2p','3s','3p','3d','4s','4p','5s','4d','5p','6s','4f','5d','6p'], 1.70, (True, True), '금속', 83),
    'Po': (1.97, 2.00, 8.42, 1.90, '#AB5C00', ['1s','2s','2p','3s','3p','3d','4s','4p','5s','4d','5p','6s','4f','5d','6p'], 1.68, (True, True), '준금속', 84),
    'At': (2.02, 2.20, 9.65, 2.80, '#754F45', ['1s','2s','2p','3s','3p','3d','4s','4p','5s','4d','5p','6s','4f','5d','6p'], 1.43, (True, True), '준금속', 85),
    'Rn': (2.45, None, 10.75, 0.00, '#428296', ['1s','2s','2p','3s','3p','3d','4s','4p','5s','4d','5p','6s','4f','5d','6p'], 1.20, (False, False), '비금속', 86),
    'Fr': (3.50, 0.70, 4.07, 0.00, '#420066', ['1s','2s','2p','3s','3p','3d','4s','4p','5s','4d','5p','6s','4f','5d','6p','7s'], 3.10, (False, False), '금속', 87),
    'Ra': (2.83, 0.90, 5.28, 0.00, '#007D00', ['1s','2s','2p','3s','3p','3d','4s','4p','5s','4d','5p','6s','4f','5d','6p','7s'], 2.15, (False, False), '금속', 88),
    'Ac': (2.65, 1.10, 5.17, 0.00, '#70ABFA', ['1s','2s','2p','3s','3p','3d','4s','4p','5s','4d','5p','6s','4f','5d','6p','7s'], 1.95, (False, True), '악티늄족', 89),
    'Th': (2.50, 1.30, 6.31, 0.00, '#00BAFF', ['1s','2s','2p','3s','3p','3d','4s','4p','5s','4d','5p','6s','4f','5d','6p','7s','5f'], 1.80, (False, True), '악티늄족', 90),
    'Pa': (2.40, 1.50, 5.89, 0.00, '#00A1FF', ['1s','2s','2p','3s','3p','3d','4s','4p','5s','4d','5p','6s','4f','5d','6p','7s','5f'], 1.61, (False, True), '악티늄족', 91),
    'U':  (1.86, 1.38, 6.08, 0.00, '#008FFF', ['1s','2s','2p','3s','3p','3d','4s','4p','5s','4d','5p','6s','4f','5d','6p','7s','5f'], 1.58, (False, True), '악티늄족', 92),
    'Np': (2.25, 1.36, 6.27, 0.00, '#0080FF', ['1s','2s','2p','3s','3p','3d','4s','4p','5s','4d','5p','6s','4f','5d','6p','7s','5f'], 1.55, (False, True), '악티늄족', 93),
    'Pu': (2.20, 1.28, 6.03, 0.00, '#006BFF', ['1s','2s','2p','3s','3p','3d','4s','4p','5s','4d','5p','6s','4f','5d','6p','7s','5f'], 1.59, (False, True), '악티늄족', 94),
    'Am': (2.15, 1.13, 6.00, 0.00, '#545CF2', ['1s','2s','2p','3s','3p','3d','4s','4p','5s','4d','5p','6s','4f','5d','6p','7s','5f'], 1.73, (False, True), '악티늄족', 95),
    'Cm': (2.10, 1.28, 6.02, 0.00, '#785CE3', ['1s','2s','2p','3s','3p','3d','4s','4p','5s','4d','5p','6s','4f','5d','6p','7s','5f'], 1.74, (False, True), '악티늄족', 96),
    'Bk': (2.05, 1.30, 6.23, 0.00, '#8A4FE3', ['1s','2s','2p','3s','3p','3d','4s','4p','5s','4d','5p','6s','4f','5d','6p','7s','5f'], 1.75, (False, True), '악티늄족', 97),
    'Cf': (2.00, 1.30, 6.30, 0.00, '#A136D4', ['1s','2s','2p','3s','3p','3d','4s','4p','5s','4d','5p','6s','4f','5d','6p','7s','5f'], 1.76, (False, True), '악티늄족', 98),
    'Es': (1.95, 1.30, 6.42, 0.00, '#B31FD4', ['1s','2s','2p','3s','3p','3d','4s','4p','5s','4d','5p','6s','4f','5d','6p','7s','5f'], 1.77, (False, True), '악티늄족', 99),
    'Fm': (1.90, 1.30, 6.50, 0.00, '#B31FBA', ['1s','2s','2p','3s','3p','3d','4s','4p','5s','4d','5p','6s','4f','5d','6p','7s','5f'], 1.78, (False, True), '악티늄족', 100),
    'Md': (1.85, 1.30, 6.58, 0.00, '#B30DA6', ['1s','2s','2p','3s','3p','3d','4s','4p','5s','4d','5p','6s','4f','5d','6p','7s','5f'], 1.79, (False, True), '악티늄족', 101),
    'No': (1.80, 1.30, 6.65, 0.00, '#BD0D87', ['1s','2s','2p','3s','3p','3d','4s','4p','5s','4d','5p','6s','4f','5d','6p','7s','5f'], 1.80, (False, True), '악티늄족', 102),
    'Lr': (1.75, 1.30, 6.79, 0.00, '#C70066', ['1s','2s','2p','3s','3p','3d','4s','4p','5s','4d','5p','6s','4f','5d','6p','7s','5f','6d'], 1.82, (False, True), '악티늄족', 103),
    'Rf': (1.70, None, None, None, '#D9D9D9', ['1s','2s','2p','3s','3p','3d','4s','4p','5s','4d','5p','6s','4f','5d','6p','7s','5f','6d'], 1.60, (False, True), '전이금속', 104),
    'Db': (1.65, None, None, None, '#C7C7C7', ['1s','2s','2p','3s','3p','3d','4s','4p','5s','4d','5p','6s','4f','5d','6p','7s','5f','6d'], 1.59, (False, True), '전이금속', 105),
    'Sg': (1.60, None, None, None, '#B0B0B0', ['1s','2s','2p','3s','3p','3d','4s','4p','5s','4d','5p','6s','4f','5d','6p','7s','5f','6d'], 1.58, (False, True), '전이금속', 106),
    'Bh': (1.55, None, None, None, '#A0A0A0', ['1s','2s','2p','3s','3p','3d','4s','4p','5s','4d','5p','6s','4f','5d','6p','7s','5f','6d'], 1.57, (False, True), '전이금속', 107),
    'Hs': (1.50, None, None, None, '#909090', ['1s','2s','2p','3s','3p','3d','4s','4p','5s','4d','5p','6s','4f','5d','6p','7s','5f','6d'], 1.56, (False, True), '전이금속', 108),
    'Mt': (1.45, None, None, None, '#808080', ['1s','2s','2p','3s','3p','3d','4s','4p','5s','4d','5p','6s','4f','5d','6p','7s','5f','6d'], 1.55, (False, True), '전이금속', 109),
    'Ds': (1.40, None, None, None, '#707070', ['1s','2s','2p','3s','3p','3d','4s','4p','5s','4d','5p','6s','4f','5d','6p','7s','5f','6d'], 1.54, (False, True), '전이금속', 110),
    'Rg': (1.35, None, None, None, '#606060', ['1s','2s','2p','3s','3p','3d','4s','4p','5s','4d','5p','6s','4f','5d','6p','7s','5f','6d'], 1.53, (False, True), '전이금속', 111),
    'Cn': (1.30, None, None, None, '#505050', ['1s','2s','2p','3s','3p','3d','4s','4p','5s','4d','5p','6s','4f','5d','6p','7s','5f','6d'], 1.52, (False, False), '금속', 112),
    'Nh': (1.25, None, None, None, '#404040', ['1s','2s','2p','3s','3p','3d','4s','4p','5s','4d','5p','6s','4f','5d','6p','7s','5f','6d','7p'], 1.51, (False, False), '금속', 113),
    'Fl': (1.20, None, None, None, '#303030', ['1s','2s','2p','3s','3p','3d','4s','4p','5s','4d','5p','6s','4f','5d','6p','7s','5f','6d','7p'], 1.50, (False, False), '금속', 114),
    'Mc': (1.15, None, None, None, '#202020', ['1s','2s','2p','3s','3p','3d','4s','4p','5s','4d','5p','6s','4f','5d','6p','7s','5f','6d','7p'], 1.49, (False, False), '금속', 115),
    'Lv': (1.10, None, None, None, '#101010', ['1s','2s','2p','3s','3p','3d','4s','4p','5s','4d','5p','6s','4f','5d','6p','7s','5f','6d','7p'], 1.48, (False, False), '금속', 116),
    'Ts': (1.05, None, None, None, '#3B9E3B', ['1s','2s','2p','3s','3p','3d','4s','4p','5s','4d','5p','6s','4f','5d','6p','7s','5f','6d','7p'], 1.47, (True, False), '준금속', 117),
    'Og': (1.00, None, None, None, '#FFFFFF', ['+1s','2s','2p','3s','3p','3d','4s','4p','5s','4d','5p','6s','4f','5d','6p','7s','5f','6d','7p'], 1.46, (False, False), '비금속', 118),
}



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



def rotation_matrix(axis, theta):
    axis = axis / np.linalg.norm(axis)
    a = np.cos(theta/2)
    b, c, d = -axis*np.sin(theta/2)
    return np.array([
        [a*a + b*b - c*c - d*d, 2*(b*c - a*d), 2*(b*d + a*c)],
        [2*(b*c + a*d), a*a + c*c - b*b - d*d, 2*(c*d - a*b)],
        [2*(b*d - a*c), 2*(c*d + a*b), a*a + d*d - b*b - c*c]
    ])


def make_regular_triangle(center, bonded_atoms, radius=1.5):
    """
    center: 선택 원자 좌표 (np.array, shape=(3,))
    bonded_atoms: 선택 원자와 직접 결합(막대 연결)한 원자들의 좌표 리스트 (list of np.array, shape=(3,))
    radius: 정삼각형의 외접원의 반지름
    """
    n = len(bonded_atoms)
    verts = []
    for i in range(min(n,3)):
        verts.append(bonded_atoms[i] - center)
    if n < 3:
        if n == 0:
            normal = np.array([0,0,1])
            start = np.array([radius,0,0])
        elif n == 1:
            v = verts[0] / np.linalg.norm(verts[0])
            normal = np.cross(v, [0,0,1])
            if np.linalg.norm(normal) < 1e-4:
                normal = np.cross(v, [0,1,0])
            normal = normal / np.linalg.norm(normal)
            start = v * radius
        else:
            v1 = verts[0] / np.linalg.norm(verts[0])
            v2 = verts[1] / np.linalg.norm(verts[1])
            normal = np.cross(v1, v2)
            if np.linalg.norm(normal) < 1e-4:
                normal = np.array([0,0,1])
            normal = normal / np.linalg.norm(normal)
            start = (v1 + v2) / 2 * radius
        for i in range(3-n):
            angle = 2*np.pi/3 * (i+1)
            rot = rotation_matrix(normal, angle)
            pt = np.dot(rot, start)
            verts.append(pt)
    verts = np.array(verts)
    centroid = verts.mean(axis=0)
    verts = verts - centroid + center
    return verts

def make_regular_tetrahedron(center, neighbors, radius=1.5):
    n = len(neighbors)
    verts = []
    if n == 2:
        a, b = neighbors[0], neighbors[1]
        edge = np.linalg.norm(a - b)
        v0 = np.array([edge/2, 0, -edge/(2*np.sqrt(3))])
        v1 = np.array([-edge/2, 0, -edge/(2*np.sqrt(3))])
        h = edge * np.sqrt(2/3)
        v2 = np.array([0, edge/2, h - edge/(2*np.sqrt(3))])
        v3 = np.array([0, -edge/2, h - edge/(2*np.sqrt(3))])
        base = np.array([v0, v1, v2, v3])
        best_verts = None
        min_error = float('inf')
        for i, j in combinations(range(4), 2):
            for perm in permutations([a, b]):
                std_vec = base[j] - base[i]
                tgt_vec = perm[1] - perm[0]
                def rotation_matrix_from_vectors(vec1, vec2):
                    a1 = vec1 / np.linalg.norm(vec1)
                    b1 = vec2 / np.linalg.norm(vec2)
                    v = np.cross(a1, b1)
                    c = np.dot(a1, b1)
                    if np.linalg.norm(v) < 1e-8:
                        return np.eye(3)
                    s = np.linalg.norm(v)
                    kmat = np.array([[0, -v[2], v[1]],
                                    [v[2], 0, -v[0]],
                                    [-v[1], v[0], 0]])
                    return np.eye(3) + kmat + kmat @ kmat * ((1 - c) / (s ** 2))
                R = rotation_matrix_from_vectors(std_vec, tgt_vec)
                base_shifted = base - base[i]
                base_rot = (R @ base_shifted.T).T + perm[0]
                centroid = base_rot.mean(axis=0)
                verts = base_rot - centroid + center
                error = np.linalg.norm(verts[i] - perm[0]) + np.linalg.norm(verts[j] - perm[1])
                if error < min_error:
                    min_error = error
                    best_verts = verts.copy()
        return best_verts
    elif n == 3:
        verts = []
        for i in range(3):
            verts.append(neighbors[i] - center)
        verts = np.array(verts)
        verts = np.vstack([verts, np.array([0,0,-radius])])
        centroid = verts.mean(axis=0)
        verts = verts - centroid + center
        if verts.shape[0] == 4:
            p1, p2, p3 = verts[:3]
            face_center = (p1 + p2 + p3) / 3
            normal = np.cross(p2-p1, p3-p1)
            normal = normal / np.linalg.norm(normal)
            v = face_center - center
            v_len = np.linalg.norm(v)
            verts[3] = face_center - normal * v_len * 3
            centroid = verts.mean(axis=0)
            verts = verts - centroid + center
        return verts
    elif n == 4:
        verts = [neighbors[i] - center for i in range(4)]
        verts = np.array(verts)
        centroid = verts.mean(axis=0)
        verts = verts - centroid + center
        return verts
    else:
        base = np.array([
            [1, 1, 1],
            [1, -1, -1],
            [-1, 1, -1],
            [-1, -1, 1]
        ]) * (radius / np.sqrt(3))
        verts = base[:max(1, n)]
        verts = np.array([v for v in verts])
        centroid = verts.mean(axis=0)
        verts = verts - centroid + center
        return verts
    
def add_sn_shape(plotter, sn, center, neighbor_positions, radius=1.5):
    color = '#00C800'
    width = 3
    if sn == 3:
        verts = make_regular_triangle(center, neighbor_positions, radius)
        edges = [(0,1), (1,2), (2,0)]
    elif sn == 4:
        verts = make_regular_tetrahedron(center, neighbor_positions, radius)
        edges = [(0,1),(0,2),(0,3),(1,2),(2,3),(3,1)]
    else:
        return
    for i,j in edges:
        line = pv.Line(verts[i], verts[j])
        plotter.add_mesh(line, color=color, line_width=width)

def get_periodic_table_positions():
    fr_radius = ELEMENT_PROPERTIES['Fr'][0]
    spacing = fr_radius * 2.2  # ≈ 7.656 Å
    positions = {}
    lanthanide_row = 7
    actinide_row = 8
    for row, line in enumerate(PERIODIC_TABLE_GRID):
        if row == lanthanide_row:
            y = (7 + 1) * spacing  # 7주기 아래
        elif row == actinide_row:
            y = (7 + 2) * spacing  # 란타넘족 아래
        else:
            y = row * spacing
        for col, sym in enumerate(line):
            if sym and sym in ELEMENT_PROPERTIES:
                positions[sym] = (col * spacing, -y, 0)
    return positions

def get_bond_label_pos_perp(atom_positions, bonds, offset=0.5):
    mid = np.mean(atom_positions, axis=0)
    pos_list = []
    for i, j in bonds:
        p1, p2 = atom_positions[i], atom_positions[j]
        center = (p1 + p2) / 2
        bond_vec = (p2 - p1) / (np.linalg.norm(p2 - p1) + 1e-8)
        to_mid = mid - center
        perp = np.cross(bond_vec, to_mid)
        if np.linalg.norm(perp) < 1e-6:
            perp = np.cross(bond_vec, [1, 0, 0])
            if np.linalg.norm(perp) < 1e-6:
                perp = np.cross(bond_vec, [0, 1, 0])
        perp = perp / (np.linalg.norm(perp) + 1e-8)
        pos_list.append(center + perp * offset)
    return pos_list

def get_pretty_mol_name(iupac_name, synonyms, inp):
    for key in synonyms:
        key = key.lower().replace(" ", "")
        if key in COMMON_NAMES:
            kor, eng = COMMON_NAMES[key]
            return f"{kor}({eng})"
    if iupac_name:
        return f"{iupac_name}({iupac_name})"
    return inp


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
        self.hydrogen_r = HYDROGEN_ORBITAL_RADII[orb_type]
        self.element_r = ELEMENT_ORBITAL_RADII[atom_symbol][orb_type]
        self.scale = self.element_r / self.hydrogen_r
    def radial_wavefunc(self, r):
        a0 = 0.529 
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
                return sph_harm_y(1, 0, theta, phi).real
            elif self.m == 1:
                return np.sqrt(2) * sph_harm_y(1, 1, theta, phi).real
            elif self.m == -1:
                return np.sqrt(2) * sph_harm_y(1, 1, theta, phi).imag
        else:
            return np.zeros_like(theta)
    def wavefunc(self):
        R_part = self.radial_wavefunc(self.R)
        Y_part = self.angular_wavefunc(self.THETA, self.PHI)
        psi = R_part * Y_part
        return psi
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
    def __init__(self):
        super().__init__()
        self.setWindowTitle("3D Molecular Visual Simulator")
        self.setMinimumSize(720, 480)
        self.resize(1280, 720)
        self.StericNumber_info = None
        layout = QHBoxLayout(self)
        self.plotter = QtInteractor(self)
        layout.addWidget(self.plotter, stretch=2)
        right_panel = QVBoxLayout()
        layout.addLayout(right_panel, stretch=1)
        self.output_box = QTextEdit()
        self.output_box.setFont(QFont("Consolas", 14))
        self.output_box.setReadOnly(True)
        self.output_box.setFixedHeight(250)
        right_panel.addWidget(QLabel("출력", font=QFont("Arial", 15)))
        right_panel.addWidget(self.output_box)
        self.mol_entry = QLineEdit()
        self.mol_entry.setFont(QFont("Arial", 14))
        self.mol_entry.setPlaceholderText("분자식/SMILES/CID/분자명 또는 원소기호 여러 개(H Og Se U), All")
        self.gen_button = QPushButton("생성")
        self.gen_button.setFont(QFont("Arial", 14))
        self.gen_button.clicked.connect(self.generate_molecule)
        row = QHBoxLayout()
        row.addWidget(self.mol_entry)
        row.addWidget(self.gen_button)
        right_panel.addLayout(row)
        self.checks = {}
        for label, key in [('결합 길이','show_bond_length'), ('결합 각','show_bond_angle')]:
            cb = QCheckBox(label)
            cb.setChecked(True)
            cb.setFont(QFont("Arial", 13))
            cb.stateChanged.connect(lambda s,k=key: self.toggle_option(k,s))
            right_panel.addWidget(cb)
            self.checks[key] = cb
        right_panel.addSpacing(10)
        self.ao_main_layout = QVBoxLayout()
        ao_row = QHBoxLayout()
        self.ao_checkbox = QCheckBox("AO")
        self.ao_checkbox.setChecked(False)
        self.ao_checkbox.setFont(QFont("Arial", 13))
        self.ao_checkbox.stateChanged.connect(self.ao_toggled)
        ao_row.addWidget(self.ao_checkbox)
        
        self.atom_meshes    = []
        self.bond_meshes    = []
        self.atom_positions = None
        self.atom_symbols   = []
        self.bonds          = []
        self.bond_lengths   = []
        self.bond_angles    = []
        self.s_checkbox = QCheckBox("s")
        self.s_checkbox.setChecked(False)
        self.s_checkbox.setFont(QFont("Arial", 12))
        self.s_checkbox.stateChanged.connect(self.update_ao_sub_visibility)
        self.s_checkbox.stateChanged.connect(self.redraw)
        self.s_checkbox.hide()
        self.p_checkbox = QCheckBox("p")
        self.p_checkbox.setChecked(False)
        self.p_checkbox.setFont(QFont("Arial", 12))
        self.p_checkbox.stateChanged.connect(self.update_ao_sub_visibility)
        self.p_checkbox.hide()
        ao_row.addWidget(self.s_checkbox)
        ao_row.addWidget(self.p_checkbox)
        self.ao_main_layout.addLayout(ao_row)
        self.p_sub_layout = QHBoxLayout()
        self.px_checkbox = QCheckBox("px")
        self.py_checkbox = QCheckBox("py")
        self.pz_checkbox = QCheckBox("pz")
        for cb in (self.px_checkbox, self.py_checkbox, self.pz_checkbox):
            cb.setChecked(False)
            cb.setFont(QFont("Arial", 11))
            cb.stateChanged.connect(self.redraw)
            cb.hide()
            self.p_sub_layout.addWidget(cb)
        self.ao_main_layout.addLayout(self.p_sub_layout)
        right_panel.addLayout(self.ao_main_layout)
        right_panel.addSpacing(10)
        self.atom_list_label = QLabel()
        self.atom_list_label.setFont(QFont("Consolas", 12))
        self.atom_list_label.setAlignment(Qt.AlignTop)
        self.atom_list_label.setFrameStyle(QFrame.Box | QFrame.Plain)
        self.atom_list_label.setStyleSheet("background-color: #23272F; color: #F5F6FA;")
        right_panel.addWidget(self.atom_list_label)
        right_panel.addStretch()
        self.state = {k: True for k in self.checks}
        self.atom_meshes = self.bond_meshes = self.atom_positions = None
        self.bonds = self.bond_centers = self.bond_lengths = self.bond_angles = []
        self.set_output = lambda text: self.output_box.setHtml(text)
        self.view_offset = np.array([0.0, 0.0, 0.0])
        self.selected_atom_idx = None
        self.generate_molecule(default="ALL")
        self.setFocusPolicy(Qt.StrongFocus)
        self.plotter.enable_point_picking(callback=self.on_atom_pick, use_picker=True, show_message=False, left_clicking=True, show_point=False)

    def analyze_StericNumber(self):
        if not hasattr(self, "molecule") or self.selected_atom_idx is None or self.atom_positions is None:
            self.StericNumber_info = None
            return
        atom = self.molecule.atoms[self.selected_atom_idx]
        symbol = atom.symbol
        bond_count = len(atom.neighbors)
        if bond_count == 0:
            self.StericNumber_info = None
            return
        prop = ELEMENT_PROPERTIES.get(symbol)
        group_number = prop[-2] if prop and len(prop) >= 2 else None
        if group_number is not None:
            lone_pairs = get_lone_pair_count(group_number, bond_count)
        else:
            lone_pairs = 0
        SN = bond_count + lone_pairs
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
        self.StericNumber_info = {
        'atom_idx': self.selected_atom_idx,
        'symbol': symbol,
        'bond_count': bond_count,
        'lone_pairs': lone_pairs,
        'SN': SN,
        'structure': structure,
        'hybrid': hybrid
    }


    def update_ao_visibility(self, state):
        checked = state == Qt.Checked
        self.s_checkbox.setVisible(checked)
        self.p_checkbox.setVisible(checked)
        if not checked:
            self.s_checkbox.setChecked(False)
            self.p_checkbox.setChecked(False)
        self.update_ao_sub_visibility()

    def update_ao_sub_visibility(self, state=None):
        p_checked = self.p_checkbox.isChecked()
        for cb in (self.px_checkbox, self.py_checkbox, self.pz_checkbox):
            cb.setVisible(p_checked)
            if not p_checked:
                cb.setChecked(False)

    def ao_toggled(self, state):
        checked = state == Qt.Checked
        self.update_ao_visibility(state)
        if checked:
            if self.atom_positions is not None and len(self.atom_positions) > 0:
                self.selected_atom_idx = 0
            else:
                self.selected_atom_idx = None
        else:
            self.selected_atom_idx = None
        self.redraw()

    def on_atom_pick(self, picked_point, event):
        if not self.ao_checkbox.isChecked():
            return
        if self.atom_positions is None or len(self.atom_positions) == 0:
            return
        picked_pos = np.array(picked_point)
        dists = np.linalg.norm(self.atom_positions + self.view_offset - picked_pos, axis=1)
        idx = int(np.argmin(dists))
        self.selected_atom_idx = idx
        self.analyze_StericNumber()
        self.redraw()


    def toggle_option(self, key, state):
        self.state[key] = bool(state)
        self.redraw()

    def generate_molecule(self, default=None):
        inp = self.mol_entry.text().strip() if not default else default
        if not inp:
            self.set_output("<b>분자식을 입력하세요.</b>")
            self.atom_list_label.setText("")
            return
        if inp.lower() == "all":
            positions = get_periodic_table_positions()
            tokens = [sym for row in PERIODIC_TABLE_GRID for sym in row if sym and sym in ELEMENT_PROPERTIES]
            atom_meshes = []
            atom_infos = []
            centers = [positions[sym] for sym in tokens]
            for t, center in zip(tokens, centers):
                prop = ELEMENT_PROPERTIES[t]
                color = prop[4] if prop and prop[4] else '#9E9E9E'
                r = prop[0]
                atom_meshes.append((pv.Sphere(radius=r, center=center, theta_resolution=32, phi_resolution=32), color))
                atom_infos.append(f"{t}: EN({prop[1]}), R({prop[0]}), IE1({prop[2]}), EA({prop[3]})")
            self.atom_meshes = atom_meshes
            self.bond_meshes = []
            self.atom_positions = np.array(centers)
            self.bonds = []
            self.bond_centers = []
            self.bond_lengths = []
            self.bond_angles = []
            self.view_offset = np.array([0.0, 0.0, 0.0])
            self.selected_atom_idx = 0 if self.ao_checkbox.isChecked() and len(centers) > 0 else None
            self.set_output("<b>주기율표 배치 생성 완료</b>")
            self.atom_list_label.setText('\n'.join(atom_infos))
            self.redraw()
            return
        else:
            tokens = inp.split()
        if all(t in ELEMENT_PROPERTIES for t in tokens) and len(tokens) > 0:
            atom_meshes = []
            atom_infos = []
            spacing = 2.5
            radii = [ELEMENT_PROPERTIES[t][0] for t in tokens]
            total_width = sum(r*2+spacing for r in radii) - spacing
            start = -total_width / 2 + radii[0]
            centers = []
            x = start
            for i, t in enumerate(tokens):
                r = ELEMENT_PROPERTIES[t][0]
                centers.append([x, 0, 0])
                x += r * 2 + spacing
            for t, center in zip(tokens, centers):
                prop = ELEMENT_PROPERTIES[t]
                color = prop[4] if prop and prop[4] else '#9E9E9E'
                r = prop[0]
                atom_meshes.append((pv.Sphere(radius=r, center=center, theta_resolution=32, phi_resolution=32), color))
                atom_infos.append(f"{t}: EN({prop[1]}), R({prop[0]}), IE1({prop[2]}), EA({prop[3]})")
            self.atom_meshes = atom_meshes
            self.bond_meshes = []
            self.atom_positions = np.array(centers)
            self.bonds = []
            self.bond_centers = []
            self.bond_lengths = []
            self.bond_angles = []
            self.view_offset = np.array([0.0, 0.0, 0.0])
            self.selected_atom_idx = 0 if self.ao_checkbox.isChecked() and len(centers) > 0 else None
            self.set_output("<b>단일 원자 공모형 생성 완료</b>")
            self.atom_list_label.setText('\n'.join(atom_infos))
            self.redraw()
            return
        try:
            sdf_text, iupac_name, formula, synonyms, input_type = fetch_3d_sdf_and_iupac_any(inp)
            mol = parse_mol(sdf_text)
            self.molecule = Molecule(mol)
            self.atom_meshes, self.bond_meshes = build_meshes(self.molecule)
            self.atom_positions = self.molecule.get_positions()
            self.atom_symbols = self.molecule.get_symbols()
            self.bonds, self.bond_centers, self.bond_lengths, self.bond_angles = get_bond_info(self.molecule)
            self.view_offset = np.array([0.0, 0.0, 0.0])
            self.selected_atom_idx = 0 if self.ao_checkbox.isChecked() and len(self.atom_positions) > 0 else None
            if input_type == 'name':
                formula_disp = formula if formula else "-"
                self.set_output(f"<b>{formula_disp}</b> 생성 완료")
            else:
                mol_name = get_pretty_mol_name(iupac_name, synonyms, inp)
                self.set_output(f"<b>{mol_name}</b> 생성 완료")
            self.atom_list_label.setText(self.molecule.atom_summary())
            self.redraw()
        except Exception as e:
            self.set_output(f"<b>오류:</b> {e}")
            self.atom_list_label.setText("")

    def redraw(self):
        self.plotter.clear()
        offset = self.view_offset if hasattr(self, "view_offset") else np.array([0.0, 0.0, 0.0])
        ao_on   = self.ao_checkbox.isChecked()
        sel_idx = self.selected_atom_idx
        valid_ao = (
            ao_on
            and sel_idx is not None
            and hasattr(self, "atom_symbols")
            and 0 <= sel_idx < len(self.atom_symbols)
        )
        for i, (mesh, color) in enumerate(self.atom_meshes or []):
            mesh_copy = mesh.copy()
            mesh_copy.translate(offset)
            if valid_ao and i == sel_idx:
                self.plotter.add_mesh(mesh_copy, color=color, opacity=0.2,
                                    specular=0.4, smooth_shading=True)
                center      = mesh_copy.center
                atom_symbol = self.atom_symbols[sel_idx]
                atom_radius = ELEMENT_PROPERTIES.get(atom_symbol, (0.53,))[0]
                nucleus = pv.Sphere(radius=atom_radius/6, center=center,
                                    theta_resolution=32, phi_resolution=32)
                self.plotter.add_mesh(nucleus, color='#FF3333', opacity=1.0,
                                    specular=0.6, smooth_shading=True)
            else:
                self.plotter.add_mesh(mesh_copy, color=color, opacity=1.0,
                                    specular=0.4, smooth_shading=True)
        for mesh, color in self.bond_meshes or []:
            mesh_copy = mesh.copy()
            mesh_copy.translate(offset)
            self.plotter.add_mesh(mesh_copy, color=color, opacity=1.0,
                                smooth_shading=True)
        if self.StericNumber_info:
            info = self.StericNumber_info
            text  = (f"결합수: {info['bond_count']}, 비공유전자쌍: {info['lone_pairs']}, SN: {info['SN']}<br>"
                    f"결합구조: {info['structure']}, 혼성화: {info['hybrid']}")
            self.set_output(text)
        if valid_ao and self.atom_positions is not None:
            atom_symbol = self.atom_symbols[sel_idx]
            atom_center = self.atom_positions[sel_idx] + offset
            checked_orbitals = []
            if self.s_checkbox.isChecked(): checked_orbitals.append('s')
            if self.p_checkbox.isChecked():
                if self.px_checkbox.isChecked(): checked_orbitals.append('px')
                if self.py_checkbox.isChecked(): checked_orbitals.append('py')
                if self.pz_checkbox.isChecked(): checked_orbitals.append('pz')
            for orb_type in checked_orbitals:
                try:
                    ho = HydrogenOrbital(
                        orb_type,
                        HYDROGEN_ORBITAL_RADII,
                        ELEMENT_ORBITAL_RADII,
                        atom_symbol
                    )
                except Exception as e:
                    self.set_output(f"<b>오비탈 생성 오류: {e}</b>")
                    continue
                for surf in ho.generate_isosurfaces():
                    mesh_o = surf['surface'].copy()
                    mesh_o.points += atom_center
                    color   = '#FF3333' if surf['color'] == 'red' else '#1976D2'
                    opacity = 0.7 if surf['type'] == 'outer' else 0.4
                    self.plotter.add_mesh(mesh_o, color=color,
                                        opacity=opacity, smooth_shading=True)
        if valid_ao and self.StericNumber_info:
            sn                = self.StericNumber_info['SN']
            atom_pos          = self.atom_positions[sel_idx] + offset
            neighbor_indices  = list(self.molecule.atoms[sel_idx].neighbors)
            neighbor_positions= [self.atom_positions[i] + offset for i in neighbor_indices]
            add_sn_shape(self.plotter, sn, atom_pos, neighbor_positions)
        if self.state.get('show_bond_length') and self.bond_lengths:
            pos = get_bond_label_pos_perp(self.atom_positions+offset, self.bonds, offset=0.5)
            labels = [f"{l:.2f}Å" for l in self.bond_lengths]
            self.plotter.add_point_labels(pos, labels,
                                        font_size=15, text_color='#A5D6A7',
                                        point_color='#23272F', point_size=18,
                                        name='bond_label')
        if self.state.get('show_bond_angle') and self.bond_angles:
            pos_vals = [c+offset for c, _ in self.bond_angles]
            angle_vals = [f"{a:.1f}°" for _, a in self.bond_angles]
            self.plotter.add_point_labels(pos_vals, angle_vals,
                                        font_size=15, text_color='#FFD54F',
                                        point_color='#23272F', point_size=18,
                                        name='angle_label')
        self.plotter.set_background("#000000")
        self.plotter.add_box_axes(line_width=3,
                                xlabel='X', ylabel='Y', zlabel='Z',
                                edge_color='#393D45', text_scale=0.9,
                                x_color='#90CAF9', y_color='#A5D6A7',
                                z_color='#CE93D8', label_color='#F5F6FA')
        self.plotter.show_bounds(grid='back', location='outer',
                                xtitle='X', ytitle='Y', ztitle='Z',
                                color='#393D45', font_size=14,
                                corner_factor=0.9)
        self.plotter.render()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    pal = QPalette()
    pal.setColor(QPalette.Window, QColor(24, 26, 32))
    pal.setColor(QPalette.WindowText, QColor(245, 246, 250))
    pal.setColor(QPalette.Base, QColor(35, 39, 47))
    pal.setColor(QPalette.Text, QColor(245, 246, 250))
    pal.setColor(QPalette.Button, QColor(35, 39, 47))
    pal.setColor(QPalette.ButtonText, QColor(245, 246, 250))
    pal.setColor(QPalette.Highlight, QColor(25, 118, 210))
    pal.setColor(QPalette.HighlightedText, QColor(245, 246, 250))
    app.setPalette(pal)
    win = MoleculeApp()
    win.show()
    sys.exit(app.exec_())