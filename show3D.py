import open3d as o3d
import argparse
# 读取PCD文件

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="visualize point cloud data")
    parser.add_argument("--pcd_file",default="output\\20241130\\3D\\image_array.pcd", help="path to pcd file")
    args = parser.parse_args()
    
    # 解析命令行参数

    if not args.pcd_file:
        print("No pcd file specified, use default file output\\20241130\\3D\\image_array.pcd")
        args.pcd_file = "output\\20241130\\3D\\image_array.pcd"
        if not o3d.io.read_point_cloud(args.pcd_file):
            print("pcd file not exist, please check the path")
            exit()
    pcd = o3d.io.read_point_cloud(args.pcd_file)

    # 可视化点云数据
    o3d.visualization.draw_geometries([pcd])