import argparse
import open3d as o3d
import numpy as np
def project_point_cloud(pcd, plane):
    """
    Project a point cloud onto a specified plane.
    :param pcd: open3d.geometry.PointCloud, the input point cloud.
    :param plane: str, the plane to project onto ('xy', 'yz', 'xz').
    :return: open3d.geometry.PointCloud, the projected point cloud.
    """
    # 获取点云的numpy数组表示
    points = np.asarray(pcd.points)

    # 创建一个空的数组来存储投影点
    projected_points = np.zeros_like(points)

    if plane == 'xy':
        # 投影到xy平面，z坐标设为0
        projected_points[:, 2] = 0
        projected_points[:, 0] = points[:, 0]
        projected_points[:, 1] = points[:, 1]

    elif plane == 'yz':
        # 投影到yz平面，x坐标设为0
        projected_points[:, 0] = 0
        projected_points[:, 2] = points[:, 2]
        projected_points[:, 1] = points[:, 1]

    elif plane == 'xz':
        # 投影到xz平面，y坐标设为0
        projected_points[:, 1] = 0
        projected_points[:, 0] = points[:, 0]
        projected_points[:, 2] = points[:, 2]

    else:
        raise ValueError("Invalid plane. Choose 'xy', 'yz', or 'xz'.")

    # 将原始点和投影点的对应坐标复制到投影点数组
    
    # 将投影点转换回open3d PointCloud格式
    projected_pcd = o3d.geometry.PointCloud()
    projected_pcd.points = o3d.utility.Vector3dVector(projected_points)

    return projected_pcd


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pcd_file',default='output\\after20241202\\3D\\image_array.pcd', help='Path to the input PCD file.')
    parser.add_argument('--plane', default='xy', help='Plane to project onto (xy, yz, or xz).')
    args = parser.parse_args()
    # 使用示例
    pcd_file = args.pcd_file  # PCD文件路径
    plane = args.plane  # 投影平面

    # 读取PCD文件
    pcd = o3d.io.read_point_cloud(pcd_file)

    # 投影点云
    projected_pcd = project_point_cloud(pcd, plane)

    # 可视化结果
    o3d.visualization.draw_geometries([projected_pcd])