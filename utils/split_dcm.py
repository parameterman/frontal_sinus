import os
import argparse
from pathlib import Path
import pydicom
import numpy as np
import matplotlib.pyplot as plt
from volume import print_progress
# 读取DICOM系列，并将它们堆叠成3D体积数据
def load_dicom_series(dicom_folder):
    dicom_files = [os.path.join(dicom_folder, f) for f in os.listdir(dicom_folder) if f.endswith('.dcm')]
    dicom_files.sort()  # 保证文件按照顺序排列
    slices = [pydicom.dcmread(f).pixel_array for f in dicom_files]
    return np.stack(slices, axis=0)

# 读取单个DICOM文件
def load_dicom_file(dicom_file):
    # 使用pydicom读取DICOM文件
    ds = pydicom.dcmread(dicom_file, force=True)
    # for elem in ds:
    #     print(f"Tag: {elem.tag}")
    #     print(f"Description: {elem}")
    #     print(f"Value: {elem.value}")
    #     print(f"VR: {elem.VR}")
    #     print('------')
    
    # 提取图像数据（pixel_array 是包含影像数据的numpy数组）
    pixel_array = ds.pixel_array
    
    return pixel_array

# 保存切片，方向可以是 'axial', 'coronal', 或 'sagittal'
def save_slices(volume, step, plane, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    if plane == 'axial':
        for i in range(0, volume.shape[0], step):
            print_progress(i+1, volume.shape[0], prefix='Progress:', suffix='Complete', length=50)
            slice = volume[i, :, :]
            # print(type(slice))
            plt.imsave(output_dir / f"axial_slice_{i}.png", slice, cmap='gray')
    elif plane == 'coronal':
        for i in range(0, volume.shape[1], step):
            print_progress(i+1, volume.shape[1], prefix='Progress:', suffix='Complete', length=50)
            slice = volume[:, i, :]
            # print(slice.dtype)
            plt.imsave(output_dir / f"coronal_slice_{i}.png", slice, cmap='gray')
    elif plane == 'sagittal':
        for i in range(0, volume.shape[2], step):
            print_progress(i+1, volume.shape[2], prefix='Progress:', suffix='Complete', length=50)
            slice = volume[:, :, i]
            plt.imsave(output_dir / f"sagittal_slice_{i}.png", slice, cmap='gray')
    else:
        raise ValueError("plane must be 'axial', 'coronal', or 'sagittal'")

# 主函数
def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="DICOM CT切片工具")
    parser.add_argument('-f', '--file', type=str, help="单个DICOM文件路径")
    parser.add_argument('-d', '--dir', type=str, help="包含多个DICOM文件的文件夹路径")
    parser.add_argument('-o', '--output', type=str, help="输出文件夹路径")
    parser.add_argument('-s', '--step', type=int, default=5, help="切片步长 (默认是5)")
    parser.add_argument('-p', '--plane', type=str, default='coronal', choices=['axial', 'coronal', 'sagittal'], help="切片方向 (默认是coronal)")
    
    args = parser.parse_args()

    if not args.output:
        # 使用传输的文件名或者文件夹名作为输出文件夹名
        if args.file:
            output = Path(args.file).stem
        elif args.dir:
            output = Path(args.dir).stem
        else:
            raise ValueError("必须提供DICOM文件路径或文件夹路径!")
        output += "_slices"
    else:
        output = args.output


    # 加载DICOM文件或文件夹
    if args.file:
        volume = load_dicom_file(args.file)
    elif args.dir:
        volume = load_dicom_series(args.dir)
    else:
        print("必须提供DICOM文件路径或文件夹路径!")
        return

    # 打印体积数据的形状
    print(f"Volume shape: {volume.shape}")

    # 保存切片
    save_slices(volume, args.step, args.plane, output)
    
if __name__ == "__main__":
    main()