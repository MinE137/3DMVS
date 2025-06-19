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
    # 1. 결합 원자 좌표를 꼭짓점에 우선 할당 (중심 기준 상대좌표)
    for i in range(min(n,3)):
        verts.append(bonded_atoms[i] - center)
    # 2. 부족한 꼭짓점은 평면상에 정삼각형 형태로 자동 생성
    if n < 3:
        # 평면 결정: 인접 원자가 없으면 z=0, 1개면 그 벡터 기준, 2개면 두 벡터의 법선
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
        else: # n==2
            v1 = verts[0] / np.linalg.norm(verts[0])
            v2 = verts[1] / np.linalg.norm(verts[1])
            normal = np.cross(v1, v2)
            if np.linalg.norm(normal) < 1e-4:
                normal = np.array([0,0,1])
            normal = normal / np.linalg.norm(normal)
            start = (v1 + v2) / 2 * radius
        # 부족한 꼭짓점들을 120도(2π/3) 간격으로 배치
        for i in range(3-n):
            angle = 2*np.pi/3 * (i+1)
            rot = rotation_matrix(normal, angle)
            pt = np.dot(rot, start)
            verts.append(pt)
    verts = np.array(verts)
    # 3. 무게중심을 원점으로 이동 후 center로 평행이동
    centroid = verts.mean(axis=0)
    verts = verts - centroid + center
    return verts

def make_regular_tetrahedron(center, neighbors, radius=1.5):
    n = len(neighbors)
    verts = []

    if n == 2:
        a, b = neighbors[0], neighbors[1]
        edge = np.linalg.norm(a - b)
        # 표준 정사면체(무게중심 원점, 한 변 edge)
        v0 = np.array([edge/2, 0, -edge/(2*np.sqrt(3))])
        v1 = np.array([-edge/2, 0, -edge/(2*np.sqrt(3))])
        h = edge * np.sqrt(2/3)
        v2 = np.array([0, edge/2, h - edge/(2*np.sqrt(3))])
        v3 = np.array([0, -edge/2, h - edge/(2*np.sqrt(3))])
        base = np.array([v0, v1, v2, v3])
        # 모든 꼭짓점 쌍을 순회하며 rigid transform 적용
        best_verts = None
        min_error = float('inf')
        for i, j in combinations(range(4), 2):
            for perm in permutations([a, b]):
                # std_vec: base[j] - base[i], tgt_vec: perm[1] - perm[0]
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
        # 기존 방식: 세 결합원자와 선택 원자 기준으로 정사면체 생성
        verts = []
        for i in range(3):
            verts.append(neighbors[i] - center)
        # 마지막 꼭짓점 임시 추가
        verts = np.array(verts)
        verts = np.vstack([verts, np.array([0,0,-radius])])
        centroid = verts.mean(axis=0)
        verts = verts - centroid + center
        # 면을 기준으로 네 번째 꼭짓점 재조정(기존 코드 유지)
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
        # 네 결합형성원자를 꼭짓점으로 하는 정사면체, 무게중심이 center
        verts = [neighbors[i] - center for i in range(4)]
        verts = np.array(verts)
        centroid = verts.mean(axis=0)
        verts = verts - centroid + center
        return verts

    else:
        # 일반적인 경우(3개 미만): 기존 방식 사용
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