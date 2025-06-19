import common
import _1402
import _1403
import _1401_cod
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

def get_periodic_table_positions():
    # 프랑슘 반지름(Å) × 2.2
    fr_radius = ELEMENT_PROPERTIES['Fr'][0]
    spacing = fr_radius * 2.2  # ≈ 7.656 Å

    positions = {}
    # 란타넘족/악티늄족의 y오프셋
    lanthanide_row = 7
    actinide_row = 8
    # 7주기(y=7), 란타넘족(y=8), 악티늄족(y=9)로 배치
    for row, line in enumerate(PERIODIC_TABLE_GRID):
        # 란타넘족과 악티늄족은 7주기보다 더 아래에 위치
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

class MoleculeApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("3D Molecular Visual Simulator")
        self.setMinimumSize(720, 480)
        self.resize(1280, 720)
        self.hybridization_info = None
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
        self.analyze_hybridization()  # 혼성화 정보 판정 추가
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
        # 2D 격자 순서대로 원소 나열
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
        # 화면 초기화
        self.plotter.clear()
        offset = self.view_offset if hasattr(self, "view_offset") else np.array([0.0, 0.0, 0.0])
        # AO 토글 상태와 선택 인덱스 가져오기
        ao_on   = self.ao_checkbox.isChecked()
        sel_idx = self.selected_atom_idx
        # AO 그릴 수 있는 유효 조건: 토글 켜짐, 인덱스 존재, 리스트 범위 내
        valid_ao = (
            ao_on
            and sel_idx is not None
            and hasattr(self, "atom_symbols")
            and 0 <= sel_idx < len(self.atom_symbols)
        )
        # 원자 메시 그리기 (AO도 여기서 처리)
        for i, (mesh, color) in enumerate(self.atom_meshes or []):
            mesh_copy = mesh.copy()
            mesh_copy.translate(offset)

            if valid_ao and i == sel_idx:
                # 선택된 원자는 반투명으로 그리고 핵 표시
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
                # 일반 원자
                self.plotter.add_mesh(mesh_copy, color=color, opacity=1.0,
                                    specular=0.4, smooth_shading=True)

        # 결합 메시 그리기
        for mesh, color in self.bond_meshes or []:
            mesh_copy = mesh.copy()
            mesh_copy.translate(offset)
            self.plotter.add_mesh(mesh_copy, color=color, opacity=1.0,
                                smooth_shading=True)

        # 혼성화 정보 텍스트
        if self.hybridization_info:
            info = self.hybridization_info
            text  = (f"결합수: {info['bond_count']}, 비공유전자쌍: {info['lone_pairs']}, SN: {info['SN']}<br>"
                    f"결합구조: {info['structure']}, 혼성화: {info['hybrid']}")
            self.set_output(text)

        # AO 오비탈 그리기 (s, p, px/py/pz)
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

        # SN 형태 표시
        if valid_ao and self.hybridization_info:
            sn                = self.hybridization_info['SN']
            atom_pos          = self.atom_positions[sel_idx] + offset
            neighbor_indices  = list(self.molecule.atoms[sel_idx].neighbors)
            neighbor_positions= [self.atom_positions[i] + offset for i in neighbor_indices]
            add_sn_shape(self.plotter, sn, atom_pos, neighbor_positions)

        # 결합 길이 레이블
        if self.state.get('show_bond_length') and self.bond_lengths:
            pos = get_bond_label_pos_perp(self.atom_positions+offset, self.bonds, offset=0.5)
            labels = [f"{l:.2f}Å" for l in self.bond_lengths]
            self.plotter.add_point_labels(pos, labels,
                                        font_size=15, text_color='#A5D6A7',
                                        point_color='#23272F', point_size=18,
                                        name='bond_label')

        # 결합 각도 레이블
        if self.state.get('show_bond_angle') and self.bond_angles:
            pos_vals = [c+offset for c, _ in self.bond_angles]
            angle_vals = [f"{a:.1f}°" for _, a in self.bond_angles]
            self.plotter.add_point_labels(pos_vals, angle_vals,
                                        font_size=15, text_color='#FFD54F',
                                        point_color='#23272F', point_size=18,
                                        name='angle_label')

        # 배경·축·경계 설정
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

        # 변경사항 화면에 반영
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
