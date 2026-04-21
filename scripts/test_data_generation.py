"""
测试 generate_training_data.py 的功能

验证:
1. 场景加载和解析
2. 热力图生成（高斯分布 + 碰撞检测）
3. 数据增强（随机移动/旋转/缩放）
4. 样本保存和 annotations.json 格式
5. 碰撞检测逻辑

使用方法:
    python scripts/test_data_generation.py
"""

import sys
import json
import tempfile
from pathlib import Path

import numpy as np
import trimesh

# 添加项目根目录到 Python 路径
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from pretreatment.generate_training_data import TrainingDataGenerator, ObjectInfo


def test_parse_jid():
    """测试 jid 解析"""
    gen = TrainingDataGenerator(
        scene_dir=Path("dummy"),
        model_dir=Path("dummy"),
        output_dir=Path("dummy"),
    )
    # 标准格式
    model_id, sx, sy, sz = gen.parse_jid("chair_001-(2.0)-(1.5)-(1.0)")
    assert model_id == "chair_001"
    assert sx == 2.0 and sy == 1.5 and sz == 1.0

    # 无缩放格式
    model_id, sx, sy, sz = gen.parse_jid("table_002")
    assert model_id == "table_002"
    assert sx == 1.0 and sy == 1.0 and sz == 1.0

    print(f"[PASS] test_parse_jid")


def test_heatmap_generation():
    """测试热力图生成"""
    gen = TrainingDataGenerator(
        scene_dir=Path("dummy"),
        model_dir=Path("dummy"),
        output_dir=Path("dummy"),
    )

    # 创建带边界的场景
    scene = trimesh.Scene()
    floor = trimesh.creation.box(extents=[10, 10, 0.1])
    floor.apply_translation([0, 0, -0.05])
    scene.add_geometry(floor, geom_name="floor")

    # 生成热力图，中心在 (0, 0)
    heatmap = gen.generate_heatmap(scene, target_pos=[0, 0, 0], image_size=256, sigma=10)

    # 验证形状
    assert heatmap.shape == (256, 256)
    # 验证值范围
    assert heatmap.min() >= 0.0
    assert heatmap.max() <= 1.0
    # 中心点应该是最大值（或接近最大值）
    center_val = heatmap[128, 128]
    assert center_val > 0.5, f"中心值 {center_val} 应该 > 0.5"
    # 角落应该接近 0
    corner_val = heatmap[0, 0]
    assert corner_val < 0.1, f"角落值 {corner_val} 应该 < 0.1"

    print("[PASS] test_heatmap_generation")


def test_heatmap_collision():
    """测试热力图碰撞检测"""
    gen = TrainingDataGenerator(
        scene_dir=Path("dummy"),
        model_dir=Path("dummy"),
        output_dir=Path("dummy"),
    )

    # 创建带障碍物的场景
    scene = trimesh.Scene()
    # 添加地板确保有 bounds
    floor = trimesh.creation.box(extents=[10, 10, 0.1])
    floor.apply_translation([0, 0, -0.05])
    scene.add_geometry(floor, geom_name="floor")
    
    # 添加一个大障碍物覆盖中心区域
    cube = trimesh.creation.box(extents=[2, 2, 1])
    cube.apply_translation([0, 0, 0.5])
    scene.add_geometry(cube, geom_name="obstacle")

    heatmap = gen.generate_heatmap(scene, target_pos=[0, 0, 0], image_size=128, sigma=5)

    # 碰撞区域（中心附近）应该为 0 或显著低于最大值
    center_val = heatmap[64, 64]
    max_val = heatmap.max()
    # 中心值应该远小于最大值（因为有障碍物）
    assert center_val < max_val * 0.5, f"碰撞中心值 {center_val} 应该显著小于最大值 {max_val}"

    print("[PASS] test_heatmap_collision")


def test_world_to_image_conversion():
    """测试世界坐标到图像坐标的转换"""
    gen = TrainingDataGenerator(
        scene_dir=Path("dummy"),
        model_dir=Path("dummy"),
        output_dir=Path("dummy"),
    )

    # 模拟场景边界
    bounds = np.array([[-5, -5, 0], [5, 5, 3]])

    x, y = gen._world_to_image([0, 0, 0], bounds, 256)
    assert abs(x - 128) < 1, f"x={x} 应该接近 128"
    assert abs(y - 128) < 1, f"y={y} 应该接近 128"

    x, y = gen._world_to_image([-5, -5, 0], bounds, 256)
    assert x < 5 and y < 5, f"左下角应该映射到图像左下角: x={x}, y={y}"

    x, y = gen._world_to_image([5, 5, 0], bounds, 256)
    assert x > 250 and y > 250, f"右上角应该映射到图像右上角: x={x}, y={y}"

    print("[PASS] test_world_to_image_conversion")


def test_text_generation():
    """测试文本 prompt 和 response 生成"""
    gen = TrainingDataGenerator(
        scene_dir=Path("dummy"),
        model_dir=Path("dummy"),
        output_dir=Path("dummy"),
    )

    prompt = gen.generate_text_prompt("椅子")
    assert "<image>" in prompt
    assert "椅子" in prompt

    response = gen.generate_response("椅子")
    assert "<SEG>" in response
    assert "椅子" in response

    print("[PASS] test_text_generation")


def test_rotation_6d_conversion():
    """测试四元数到 6D 旋转的转换"""
    gen = TrainingDataGenerator(
        scene_dir=Path("dummy"),
        model_dir=Path("dummy"),
        output_dir=Path("dummy"),
    )

    from scipy.spatial.transform import Rotation as R

    # 无旋转
    quat = [0, 0, 0, 1]
    rot_6d = gen.rotation_6d_from_quat(quat)
    assert len(rot_6d) == 6
    np.testing.assert_allclose(rot_6d, [1, 0, 0, 0, 1, 0], atol=1e-6)

    # Z 轴旋转 90 度
    rot_z = R.from_euler('z', 90, degrees=True)
    quat = rot_z.as_quat()
    rot_6d = gen.rotation_6d_from_quat(quat.tolist())
    R_mat = np.eye(3)
    R_mat[:, 0] = rot_6d[:3]
    R_mat[:, 1] = rot_6d[3:]
    R_reconstructed = R.from_matrix(R_mat)
    angle_diff = (R_reconstructed.inv() * rot_z).magnitude()
    assert angle_diff < 0.01, f"旋转重建误差: {angle_diff}"

    print("[PASS] test_rotation_6d_conversion")


def test_full_pipeline():
    """测试完整流程（使用模拟数据）"""
    gen = TrainingDataGenerator(
        scene_dir=Path("dummy"),
        model_dir=Path("dummy"),
        output_dir=Path("dummy"),
    )

    # 模拟一个物体
    obj_mesh = trimesh.creation.box(extents=[1, 1, 1])
    obj = ObjectInfo(
        jid="test_001",
        model_id="box_001",
        name="测试物体",
        pos=[0, 0, 0],
        rot=[0, 0, 0, 1],
        size=[1, 1, 1],
        scale_jid=(1.0, 1.0, 1.0),
        mesh=obj_mesh,
        is_on_floor=True,
        is_on_wall=False,
    )

    # 验证物体信息
    assert obj.name == "测试物体"
    assert obj.is_on_floor
    assert not obj.is_on_wall

    # 验证热力图生成
    scene = trimesh.Scene()
    heatmap = gen.generate_heatmap(scene, target_pos=[0, 0, 0], image_size=256)
    assert heatmap.shape == (256, 256)

    # 验证旋转 6D
    rot_6d = gen.rotation_6d_from_quat(obj.rot)
    assert len(rot_6d) == 6

    print("[PASS] test_full_pipeline")


def test_augmentation():
    """测试数据增强功能"""
    gen = TrainingDataGenerator(
        scene_dir=Path("dummy"),
        model_dir=Path("dummy"),
        output_dir=Path("dummy"),
        augmentation=True,
        aug_ratio=0.5,
    )

    obj_mesh = trimesh.creation.box(extents=[1, 1, 1])
    obj = ObjectInfo(
        jid="test_002",
        model_id="box_002",
        name="增强物体",
        pos=[0, 0, 0],
        rot=[0, 0, 0, 1],
        size=[1, 1, 1],
        scale_jid=(1.0, 1.0, 1.0),
        mesh=obj_mesh,
        is_on_floor=True,
        is_on_wall=False,
    )

    scene = trimesh.Scene()

    aug_obj, aug_scene = gen.augmentation_object(obj, scene)
    if aug_obj is not None:
        # 验证增强后的物体位置不同于原始位置
        assert aug_obj.pos != obj.pos, "增强后的位置应该改变"
        # 验证增强后的物体仍然在地面上
        assert aug_obj.is_on_floor, "增强后的物体应该在地面上"

    print("[PASS] test_augmentation")


def test_annotations_format():
    """测试 annotations.json 格式"""
    with tempfile.TemporaryDirectory() as tmpdir:
        gen = TrainingDataGenerator(
            scene_dir=Path(tmpdir),
            model_dir=Path(tmpdir),
            output_dir=Path(tmpdir),
        )

        # 添加多个模拟样本（确保 train/val/test 划分有效）
        gen.annotations = []
        for i in range(10):
            gen.annotations.append({
                "scene_id": f"scene_{i:06d}",
                "split": "train",
                "plane_image_path": f"plane_images/scene_{i:06d}.png",
                "images_path": [
                    f"plane_images/scene_{i:06d}.png",
                    f"object_images/scene_{i:06d}.png",
                ],
                "mask_path": f"masks/scene_{i:06d}.png",
                "text_prompt": "<image>\n<image>\n请把测试物体放回原来的位置",
                "response": "好的，我会把测试物体放回原来的位置。<SEG>",
                "rotation_6d": [1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                "scale": 1.0,
            })
        gen.save_annotations()

        # 读取并验证
        with open(Path(tmpdir) / "annotations.json", 'r', encoding='utf-8') as f:
            annotations = json.load(f)

        assert len(annotations) == 10
        ann = annotations[0]
        required_fields = [
            "scene_id", "split", "plane_image_path",
            "images_path", "mask_path", "text_prompt",
            "response", "rotation_6d", "scale",
        ]
        for field in required_fields:
            assert field in ann, f"缺少字段: {field}"

        assert len(ann["rotation_6d"]) == 6
        assert isinstance(ann["scale"], float)

        # 验证 split 划分
        splits = [a["split"] for a in annotations]
        assert "train" in splits
        assert "val" in splits or "test" in splits

    print(f"\n[PASS] test_annotations_format")


def main():
    print("=" * 60)
    print("运行 generate_training_data.py 测试")
    print("=" * 60)

    tests = [
        test_parse_jid,
        test_world_to_image_conversion,
        test_heatmap_generation,
        test_heatmap_collision,
        test_text_generation,
        test_rotation_6d_conversion,
        test_full_pipeline,
        test_augmentation,
        test_annotations_format,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"[FAIL] {test.__name__}: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    print("\n" + "=" * 60)
    print(f"测试完成: {passed} 通过, {failed} 失败")
    print("=" * 60)

    if failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
