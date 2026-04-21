"""
SSR3D-FRONT 数据集批量场景生成器

自动遍历 dataset-ssr3dfront/scenes/ 目录下的所有 JSON 文件，
加载对应的模型并生成 GLB 场景到 output 目录。

使用方法:
    python generate_room_ssr3dfront.py
"""

import os
import json
import numpy as np
from pathlib import Path

import trimesh
from scipy.spatial.transform import Rotation as R

# ========== 配置路径 ==========
# 请确保以下路径正确
BASE_DIR = Path(r"d:/3D-Dataset")
SCENES_DIR = BASE_DIR / "dataset-ssr3dfront" / "scenes"
FUTURE_MODEL_PATH = BASE_DIR / "3D-FUTURE-model"
OUTPUT_DIR = BASE_DIR / "dataset-ssr3dfront" / "output"
# ==============================


class SSR3DFrontGenerator:
    """SSR3D-FRONT 单个房间场景生成器"""

    def __init__(self, json_path, output_path):
        self.json_path = json_path
        self.output_path = output_path
        self.scene_data = None
        self.scene = trimesh.Scene()

    def load_scene(self):
        """加载场景 JSON 数据"""
        with open(self.json_path, 'r', encoding='utf-8') as f:
            self.scene_data = json.load(f)
        return self.scene_data

    def _triangulate_polygon(self, n):
        """将多边形三角化（扇形）"""
        faces = []
        for i in range(1, n - 1):
            faces.append([0, i, i + 1])
        return np.array(faces)

    def create_room_geometry(self):
        """创建房间墙壁、地板、天花板"""
        bounds_top = self.scene_data.get('bounds_top', [])
        bounds_bottom = self.scene_data.get('bounds_bottom', [])

        if not bounds_top or not bounds_bottom:
            return

        # 地板
        floor_v = np.array(bounds_bottom)
        floor_f = self._triangulate_polygon(len(bounds_bottom))
        self.scene.add_geometry(
            trimesh.Trimesh(vertices=floor_v, faces=floor_f, process=False),
            geom_name="floor"
        )

        # 天花板
        ceil_v = np.array(bounds_top)
        ceil_f = self._triangulate_polygon(len(bounds_top))
        self.scene.add_geometry(
            trimesh.Trimesh(vertices=ceil_v, faces=ceil_f, process=False),
            geom_name="ceiling"
        )

        # 墙壁
        n = len(bounds_top)
        for i in range(n):
            j = (i + 1) % n
            wall_v = np.array([bounds_bottom[i], bounds_bottom[j], bounds_top[j], bounds_top[i]])
            # 双面
            wall_f = np.array([[0, 1, 2], [0, 2, 3], [0, 2, 1], [0, 3, 2]])
            self.scene.add_geometry(
                trimesh.Trimesh(vertices=wall_v, faces=wall_f, process=False),
                geom_name=f"wall_{i}"
            )

    def parse_jid(self, jid):
        """解析 jid 获取模型 ID 和缩放"""
        parts = jid.split('-(')
        if len(parts) == 1:
            return jid, 1.0, 1.0, 1.0
        model_id = parts[0]
        sx = float(parts[1].rstrip(')'))
        sy = float(parts[2].rstrip(')'))
        sz = float(parts[3].rstrip(')'))
        return model_id, sx, sy, sz

    def find_model_path(self, model_id):
        """查找模型文件"""
        model_dir = FUTURE_MODEL_PATH / model_id
        if (model_dir / "normalized_model.glb").exists():
            return model_dir / "normalized_model.glb"
        if (model_dir / "raw_model.glb").exists():
            return model_dir / "raw_model.glb"
        glbs = list(model_dir.glob("*.glb"))
        if glbs:
            return glbs[0]
        if (model_dir / "raw_model.obj").exists():
            return model_dir / "raw_model.obj"
        return None

    def load_furniture(self):
        """加载所有家具"""
        objects = self.scene_data.get('objects', [])
        for obj_data in objects:
            jid = obj_data.get('jid', '')
            pos = obj_data.get('pos', [0, 0, 0])
            rot = obj_data.get('rot', [0, 0, 0, 1])
            size = obj_data.get('size', [1, 1, 1])

            model_id, sx, sy, sz = self.parse_jid(jid)
            model_path = self.find_model_path(model_id)
            if not model_path or not model_path.exists():
                continue

            try:
                loaded = trimesh.load(model_path, force='scene')
                if isinstance(loaded, trimesh.Scene):
                    meshes = [g for g in loaded.geometry.values() if isinstance(g, trimesh.Trimesh)]
                    if not meshes:
                        continue
                    mesh = trimesh.util.concatenate(meshes)
                else:
                    mesh = loaded

                # 缩放到期望尺寸
                extents = mesh.extents
                if np.all(extents > 0):
                    scale = np.mean(np.array(size) / extents)
                    mesh.apply_scale(scale)
                    mesh.apply_scale([sx, sy, sz])

                # 旋转
                if len(rot) == 4:
                    if not np.allclose(rot, [0, 0, 0, 1]) and not np.allclose(rot, [1, 0, 0, 0]):
                        R_mat = np.eye(4)
                        R_mat[:3, :3] = Rotation.from_quat(rot).as_matrix()
                        mesh.apply_transform(R_mat)

                # 平移
                T = np.eye(4)
                T[:3, 3] = pos
                mesh.apply_transform(T)

                self.scene.add_geometry(mesh, geom_name=f"obj_{jid}")
            except Exception:
                continue

    def generate(self):
        """执行生成"""
        try:
            self.load_scene()
            self.create_room_geometry()
            self.load_furniture()

            if not self.scene.geometry:
                return False

            os.makedirs(self.output_path.parent, exist_ok=True)
            self.scene.export(str(self.output_path))
            return True
        except Exception:
            return False


def main():
    if not SCENES_DIR.exists():
        print(f"错误: 场景目录不存在: {SCENES_DIR}")
        return
    if not FUTURE_MODEL_PATH.exists():
        print(f"错误: 模型目录不存在: {FUTURE_MODEL_PATH}")
        return

    json_files = sorted(SCENES_DIR.glob("*.json"))
    total = len(json_files)
    print(f"找到 {total} 个场景文件")
    
    success = 0
    failed = 0

    for i, json_path in enumerate(json_files, 1):
        output_path = OUTPUT_DIR / f"{json_path.stem}.glb"
        print(f"[{i}/{total}] 处理: {json_path.name}", end=" ... ")
        
        gen = SSR3DFrontGenerator(json_path, output_path)
        if gen.generate():
            print("成功")
            success += 1
        else:
            print("跳过/失败")
            failed += 1

    print(f"\n完成! 成功: {success}, 失败: {failed}")


if __name__ == "__main__":
    main()
