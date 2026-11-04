"""URDF forward kinematics, position Jacobian, CoM, and diagonal mass.

Maps the 18-DOF MPC vector onto named revolute joints; all other joints stay 0.
Inertial blocks feed p_com, J_com, and diag(M) without Pinocchio.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from xml.etree import ElementTree as ET

import numpy as np

from kmpc.types import JOINT_NAMES, N_DOF

# Wrist J5–J7 on both arms: scale λ so these M_ii map onto r_arm.
WRIST_INDICES = (8, 9, 10, 15, 16, 17)


def _local(tag: str) -> str:
    return tag.split('}', 1)[-1]


def _vec3(text: Optional[str], default: Tuple[float, float, float] = (0.0, 0.0, 0.0)) -> np.ndarray:
    if not text:
        return np.array(default, dtype=float)
    parts = [float(x) for x in text.split()]
    if len(parts) != 3:
        raise ValueError(f'expected 3 floats, got {text!r}')
    return np.array(parts, dtype=float)


def rpy_matrix(roll: float, pitch: float, yaw: float) -> np.ndarray:
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    rx = np.array([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]])
    ry = np.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]])
    rz = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]])
    return rz @ ry @ rx


def origin_matrix(xyz: np.ndarray, rpy: np.ndarray) -> np.ndarray:
    t = np.eye(4)
    t[:3, :3] = rpy_matrix(float(rpy[0]), float(rpy[1]), float(rpy[2]))
    t[:3, 3] = xyz
    return t


def axis_rotation(axis: np.ndarray, angle: float) -> np.ndarray:
    axis = axis / (np.linalg.norm(axis) + 1e-15)
    x, y, z = axis
    c, s = np.cos(angle), np.sin(angle)
    t = 1.0 - c
    r = np.array([
        [t * x * x + c, t * x * y - s * z, t * x * z + s * y],
        [t * x * y + s * z, t * y * y + c, t * y * z - s * x],
        [t * x * z - s * y, t * y * z + s * x, t * z * z + c],
    ])
    out = np.eye(4)
    out[:3, :3] = r
    return out


def axis_translation(axis: np.ndarray, distance: float) -> np.ndarray:
    axis = axis / (np.linalg.norm(axis) + 1e-15)
    out = np.eye(4)
    out[:3, 3] = axis * distance
    return out


def _inertia_matrix(el: ET.Element) -> np.ndarray:
    ixx = float(el.attrib.get('ixx', 0.0))
    ixy = float(el.attrib.get('ixy', 0.0))
    ixz = float(el.attrib.get('ixz', 0.0))
    iyy = float(el.attrib.get('iyy', 0.0))
    iyz = float(el.attrib.get('iyz', 0.0))
    izz = float(el.attrib.get('izz', 0.0))
    return np.array(
        [[ixx, ixy, ixz], [ixy, iyy, iyz], [ixz, iyz, izz]], dtype=float)


@dataclass
class UrdfJoint:
    name: str
    joint_type: str
    parent: str
    child: str
    xyz: np.ndarray
    rpy: np.ndarray
    axis: np.ndarray


@dataclass
class UrdfLinkInertial:
    name: str
    mass: float
    com: np.ndarray
    rpy: np.ndarray
    I_link: np.ndarray


class UrdfKinematics:
    def __init__(self, urdf_xml: str) -> None:
        root = ET.fromstring(urdf_xml)
        self.joints: Dict[str, UrdfJoint] = {}
        self._by_child: Dict[str, UrdfJoint] = {}
        self.links: Dict[str, UrdfLinkInertial] = {}
        for node in root.iter():
            tag = _local(node.tag)
            if tag == 'joint':
                self._parse_joint(node)
            elif tag == 'link':
                self._parse_link(node)
        self._children: Dict[str, List[UrdfJoint]] = {}
        for joint in self.joints.values():
            self._children.setdefault(joint.parent, []).append(joint)
        self.total_mass = float(sum(link.mass for link in self.links.values()))
        self._name_index = {name: i for i, name in enumerate(JOINT_NAMES)}

    def _parse_joint(self, node: ET.Element) -> None:
        name = node.attrib['name']
        jtype = node.attrib.get('type', 'fixed')
        parent = child = ''
        xyz = np.zeros(3)
        rpy = np.zeros(3)
        axis = np.array([0.0, 0.0, 1.0])
        for child_el in node:
            tag = _local(child_el.tag)
            if tag == 'parent':
                parent = child_el.attrib['link']
            elif tag == 'child':
                child = child_el.attrib['link']
            elif tag == 'origin':
                xyz = _vec3(child_el.attrib.get('xyz'))
                rpy = _vec3(child_el.attrib.get('rpy'))
            elif tag == 'axis':
                axis = _vec3(child_el.attrib.get('xyz'), (0.0, 0.0, 1.0))
        joint = UrdfJoint(name, jtype, parent, child, xyz, rpy, axis)
        self.joints[name] = joint
        self._by_child[child] = joint

    def _parse_link(self, node: ET.Element) -> None:
        name = node.attrib['name']
        mass = 0.0
        com = np.zeros(3)
        rpy = np.zeros(3)
        inertia = np.zeros((3, 3))
        for child_el in node:
            if _local(child_el.tag) != 'inertial':
                continue
            for item in child_el:
                tag = _local(item.tag)
                if tag == 'mass':
                    mass = float(item.attrib.get('value', 0.0))
                elif tag == 'origin':
                    com = _vec3(item.attrib.get('xyz'))
                    rpy = _vec3(item.attrib.get('rpy'))
                elif tag == 'inertia':
                    inertia = _inertia_matrix(item)
        self.links[name] = UrdfLinkInertial(name, mass, com, rpy, inertia)

    def chain(self, base_link: str, tip_link: str) -> List[UrdfJoint]:
        out: List[UrdfJoint] = []
        link = tip_link
        seen = set()
        while link != base_link:
            if link in seen:
                raise ValueError(f'cycle while walking {tip_link} → {base_link}')
            seen.add(link)
            joint = self._by_child.get(link)
            if joint is None:
                raise KeyError(f'no parent joint for link {link}')
            out.append(joint)
            link = joint.parent
        out.reverse()
        return out

    def q_map(self, q18: np.ndarray) -> Dict[str, float]:
        q18 = np.asarray(q18, dtype=float).reshape(N_DOF)
        return {name: float(q18[i]) for i, name in enumerate(JOINT_NAMES)}

    def fk(self, q18: np.ndarray, tip_link: str, base_link: str = 'base_link') -> np.ndarray:
        qmap = self.q_map(q18)
        t = np.eye(4)
        for joint in self.chain(base_link, tip_link):
            t = t @ origin_matrix(joint.xyz, joint.rpy)
            q = qmap.get(joint.name, 0.0)
            if joint.joint_type in ('revolute', 'continuous'):
                t = t @ axis_rotation(joint.axis, q)
            elif joint.joint_type == 'prismatic':
                t = t @ axis_translation(joint.axis, q)
        return t

    def fk_pos(self, q18: np.ndarray, tip_link: str, base_link: str = 'base_link') -> np.ndarray:
        return self.fk(q18, tip_link, base_link)[:3, 3].copy()

    def fk_point(
        self,
        q18: np.ndarray,
        link: str,
        p_local: np.ndarray,
        base_link: str = 'base_link',
    ) -> np.ndarray:
        """World position of a point fixed in `link` coordinates."""
        t = self.fk(q18, link, base_link)
        p_local = np.asarray(p_local, dtype=float).reshape(3)
        return t[:3, :3] @ p_local + t[:3, 3]

    def jacobian_pos(
        self,
        q18: np.ndarray,
        tip_link: str,
        base_link: str = 'base_link',
    ) -> np.ndarray:
        """Geometric linear Jacobian (3×18) in base_link: ṗ = J Δq/dt."""
        return self.jacobian_pos_point(q18, tip_link, np.zeros(3), base_link)

    def jacobian_pos_point(
        self,
        q18: np.ndarray,
        link: str,
        p_local: np.ndarray,
        base_link: str = 'base_link',
    ) -> np.ndarray:
        """Linear Jacobian of a point fixed in `link` (3×18, base_link)."""
        return self.jacobian_spatial(q18, link, p_local, base_link)[:3].copy()

    def jacobian_spatial(
        self,
        q18: np.ndarray,
        link: str,
        p_local: Optional[np.ndarray] = None,
        base_link: str = 'base_link',
    ) -> np.ndarray:
        """Geometric Jacobian 6×18 in base_link: [Jv; Jω] of a point in `link`.

        p_local defaults to the link origin. Jv and Jω share this p_ee.
        """
        qmap = self.q_map(q18)
        jac = np.zeros((6, N_DOF))
        t = np.eye(4)
        local = np.zeros(3) if p_local is None else p_local
        p_ee = self.fk_point(q18, link, local, base_link)
        for joint in self.chain(base_link, link):
            t = t @ origin_matrix(joint.xyz, joint.rpy)
            z = t[:3, :3] @ (joint.axis / (np.linalg.norm(joint.axis) + 1e-15))
            p_j = t[:3, 3]
            q = qmap.get(joint.name, 0.0)
            if joint.joint_type in ('revolute', 'continuous'):
                jv = np.cross(z, p_ee - p_j)
                jw = z
                t = t @ axis_rotation(joint.axis, q)
            elif joint.joint_type == 'prismatic':
                jv = z
                jw = np.zeros(3)
                t = t @ axis_translation(joint.axis, q)
            else:
                continue
            idx = self._name_index.get(joint.name)
            if idx is not None:
                jac[:3, idx] = jv
                jac[3:, idx] = jw
        return jac

    def _fk_tree(
        self,
        q18: np.ndarray,
        base_link: str = 'base_link',
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, Tuple[np.ndarray, np.ndarray, str, str]], List[str]]:
        """One walk: link poses, joint axes, parent-first link order."""
        qmap = self.q_map(q18)
        t_link: Dict[str, np.ndarray] = {base_link: np.eye(4)}
        joint_pose: Dict[str, Tuple[np.ndarray, np.ndarray, str, str]] = {}
        order: List[str] = []
        stack = [base_link]
        while stack:
            link = stack.pop()
            order.append(link)
            for joint in self._children.get(link, ()):
                t = t_link[link] @ origin_matrix(joint.xyz, joint.rpy)
                axis = joint.axis / (np.linalg.norm(joint.axis) + 1e-15)
                z = t[:3, :3] @ axis
                p_j = t[:3, 3].copy()
                q = qmap.get(joint.name, 0.0)
                if joint.joint_type in ('revolute', 'continuous'):
                    t = t @ axis_rotation(joint.axis, q)
                elif joint.joint_type == 'prismatic':
                    t = t @ axis_translation(joint.axis, q)
                t_link[joint.child] = t
                joint_pose[joint.name] = (z, p_j, joint.joint_type, joint.child)
                stack.append(joint.child)
        return t_link, joint_pose, order

    def inertial_quantities(
        self,
        q18: np.ndarray,
        base_link: str = 'base_link',
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """p_com (3,), J_com (3×18), diag(M) (18,) from one tree FK."""
        t_link, joint_pose, order = self._fk_tree(q18, base_link)
        p_com_link: Dict[str, np.ndarray] = {}
        i_world: Dict[str, np.ndarray] = {}
        mass_link: Dict[str, float] = {}
        for link, t in t_link.items():
            inert = self.links.get(link)
            if inert is None or inert.mass <= 0.0:
                continue
            r_in = t[:3, :3] @ rpy_matrix(
                float(inert.rpy[0]), float(inert.rpy[1]), float(inert.rpy[2]))
            p_com_link[link] = t[:3, :3] @ inert.com + t[:3, 3]
            i_world[link] = r_in @ inert.I_link @ r_in.T
            mass_link[link] = float(inert.mass)

        m_tot = self.total_mass
        if m_tot <= 1e-12:
            return np.zeros(3), np.zeros((3, N_DOF)), np.zeros(N_DOF)

        first = np.zeros(3)
        for link, mass in mass_link.items():
            first = first + mass * p_com_link[link]
        p_com = first / m_tot

        m_sub: Dict[str, float] = {}
        s_sub: Dict[str, np.ndarray] = {}
        for link in reversed(order):
            mass = mass_link.get(link, 0.0)
            s = mass * p_com_link[link] if mass > 0.0 else np.zeros(3)
            for joint in self._children.get(link, ()):
                mass = mass + m_sub.get(joint.child, 0.0)
                s = s + s_sub.get(joint.child, np.zeros(3))
            m_sub[link] = mass
            s_sub[link] = s

        jac = np.zeros((3, N_DOF))
        for name, idx in self._name_index.items():
            pose = joint_pose.get(name)
            if pose is None:
                continue
            z, p_j, jtype, child = pose
            m_c = m_sub.get(child, 0.0)
            if m_c <= 1e-12:
                continue
            p_c = s_sub[child] / m_c
            if jtype in ('revolute', 'continuous'):
                jac[:, idx] = (m_c / m_tot) * np.cross(z, p_c - p_j)
            elif jtype == 'prismatic':
                jac[:, idx] = (m_c / m_tot) * z

        mdiag = np.zeros(N_DOF)
        for link, mass in mass_link.items():
            p_i = p_com_link[link]
            i_w = i_world[link]
            current = link
            seen = set()
            while current != base_link and current not in seen:
                seen.add(current)
                joint = self._by_child.get(current)
                if joint is None:
                    break
                idx = self._name_index.get(joint.name)
                pose = joint_pose.get(joint.name)
                if idx is not None and pose is not None:
                    z, p_j, jtype, _child = pose
                    if jtype in ('revolute', 'continuous'):
                        r = p_i - p_j
                        zr = np.cross(z, r)
                        mdiag[idx] += mass * float(zr @ zr)
                        mdiag[idx] += float(z @ i_w @ z)
                    elif jtype == 'prismatic':
                        mdiag[idx] += mass
                current = joint.parent
        return p_com, jac, np.maximum(mdiag, 0.0)

    def com(self, q18: np.ndarray, base_link: str = 'base_link') -> np.ndarray:
        p_com, _jac, _mdiag = self.inertial_quantities(q18, base_link)
        return p_com

    def jacobian_com(self, q18: np.ndarray, base_link: str = 'base_link') -> np.ndarray:
        _p, jac, _mdiag = self.inertial_quantities(q18, base_link)
        return jac

    def mass_matrix_diag(self, q18: np.ndarray, base_link: str = 'base_link') -> np.ndarray:
        _p, _jac, mdiag = self.inertial_quantities(q18, base_link)
        return mdiag

    def inertia_lambda(
        self,
        q18: np.ndarray,
        r_arm: float,
        base_link: str = 'base_link',
    ) -> float:
        """λ such that mean wrist M_ii * λ ≈ r_arm."""
        mdiag = self.mass_matrix_diag(q18, base_link)
        wrist = mdiag[list(WRIST_INDICES)]
        mean = float(np.mean(wrist))
        if mean <= 1e-12:
            return 1.0
        return float(r_arm) / mean
