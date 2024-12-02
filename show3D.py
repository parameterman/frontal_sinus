import open3d as o3d
import argparse
# 读取PCD文件

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="visualize point cloud data")
    parser.add_argument("pcd_file", help="path to pcd file")
    args = parser.parse_args()
    
    # 解析命令行参数

    pcd = o3d.io.read_point_cloud(args.pcd_file)

    # 可视化点云数据
    o3d.visualization.draw_geometries([pcd])