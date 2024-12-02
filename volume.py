import argparse
import glob
import open3d
import open3d.visualization
import pydicom
import trimesh
from scipy import ndimage
import numpy as np
import torch
import os
import cv2
from unet.unet_build import UNet
# unet_build import UNet
from PIL import Image
from stl import mesh
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import dicom2stl
from utils import split_dcm
import sys
import time

def print_progress(iteration, total, prefix='', suffix='', decimals=1, length=50, fill='█'):
    """
    在命令行打印进度条
    :param iteration: 当前迭代次数
    :param total: 总迭代次数
    :param prefix: 进度条前的字符串
    :param suffix: 进度条后的字符串
    :param decimals: 显示的小数位数
    :param length: 进度条的长度
    :param fill: 填充进度条的字符
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filled_length = int(length * iteration // total)
    bar = fill * filled_length + '-' * (length - filled_length)
    sys.stdout.write('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix))
    sys.stdout.flush()
    # 在结束时打印一个新行
    if iteration == total:
        print()

# 示例：模拟一个循环
# total_iterations = 100
# for i in range(total_iterations):
#     # 模拟一些工作
#     time.sleep(0.01)
#     print_progress(i + 1, total_iterations, prefix='Progress:', suffix='Complete', length=50)

def nd2pcld(nd_array, output):
    if nd_array.ndim != 3:
        raise ValueError("nd_array should have 3 dimensions")
    height, width, depth = nd_array.shape
    pixel_num = np.sum(nd_array > 0)
    points = np.zeros((pixel_num+100, 3))
    colors = np.zeros((pixel_num+100, 3))
    # points = np.array([[0,0,0]])
    print("points shape:",points.shape)
    # colors = np.array([])
    num = 0
    for i in range(height):
        # print(i)
        print_progress(i+1, height, prefix='Progress:', suffix='Complete', length=50)
        for j in range(width):
            for k in range(depth):
                if num >= 100+pixel_num:
                    break
                if nd_array[i,j,k] > 0:
                    points[num] = [i,j,k]
                    num += 1
                    colors[num] = [0,nd_array[i,j,k],0]
                    
    print("points shape:",points.shape)
   
    pcd = open3d.geometry.PointCloud()
    pcd.points = open3d.utility.Vector3dVector(points)
    # pcd.colors = open3d.utility.Vector3dVector(colors/255)
    open3d.io.write_point_cloud(output, pcd)
    open3d.visualization.draw_geometries([pcd])

def reconstruct_dcm(dcm_path,image_data):
    ds = pydicom.dcmread(dcm_path)
    ds.PixelData =  image_data.tobytes()
    return ds

def init_net(model_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 加载网络，图片单通道，分类为1。
    net = UNet(n_channels=1, n_classes=1)
    # 将网络拷贝到deivce中
    net.to(device=device)
    # 加载模型参数
    net.load_state_dict(torch.load(model_path, map_location=device,weights_only=True))
    # 测试模式
    net.eval()
    return net,device

def predict(net, device, tests_path,output_dir,pixel_spacing):
    
    volume = 0
    ids = 0
    save_root = output_dir
    if not os.path.exists(save_root):
        os.makedirs(save_root)
    for test_path in tests_path:
        print_progress(ids+1, len(tests_path), prefix='Progress:', suffix='Complete', length=50)
        # 读取图片

        img = cv2.imread(test_path)
        # label = cv2.imread(label_path)
        # 转为灰度图
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        # label = cv2.cvtColor(label, cv2.COLOR_RGB2GRAY)
        # _, label = cv2.threshold(label, 0, 255, cv2.THRESH_BINARY)
        # 转为batch为1，通道为1，大小为512*512的数组
        # img = cv2.resize(img, (534, 534))
        img = img.reshape(1, 1, img.shape[0], img.shape[1])
        # label = label.reshape(1, 1, label.shape[0], label.shape[1])
        
        # 转为tensor
        img_tensor = torch.from_numpy(img)
        # label_tensor = torch.from_numpy(label)
        # 将tensor拷贝到device中，只用cpu就是拷贝到cpu中，用cuda就是拷贝到cuda中。
        img_tensor = img_tensor.to(device=device, dtype=torch.float32)
        
        
        # 预测
        pred = net(img_tensor)
        # 提取结果
        pred = np.array(pred.data.cpu()[0])[0]
        # label = np.array(label_tensor.data.cpu()[0])[0]
        pred[pred > 0.2] = 255
        pred[pred <= 0.2] = 0
        pred_area = np.sum(pred == 255)
        save_path = os.path.join(save_root,os.path.basename(test_path))
        cv2.imwrite(save_path,pred)
        print(test_path,pred_area,"pixels")
        pixel_volume = pixel_spacing[0]*pixel_spacing[1]*pixel_spacing[2]
        volume += pred_area*pixel_volume
        ids += 1
        # label_area = np.sum(label == 255)*pixel_size*pixel_size
    return volume

def predict_nosave(net, device, tests_path,output_dir,pixel_spacing):
    volume = 0
    ids = 0
    first_array = np.array(cv2.imread(tests_path[0]))
    depth = len(tests_path)
    image_array = np.zeros((first_array.shape[0],first_array.shape[1],depth),dtype=np.uint16)
    save_root = output_dir
    if not os.path.exists(save_root):
        os.makedirs(save_root)
    for test_path in tests_path:
        print_progress(ids+1, len(tests_path), prefix="Progress:{}".format(ids+1), suffix='Complete', length=50)
        # 读取图片
        img = cv2.imread(test_path)
        # label = cv2.imread(label_path)
        # 转为灰度图
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        # label = cv2.cvtColor(label, cv2.COLOR_RGB2GRAY)
        # _, label = cv2.threshold(label, 0, 255, cv2.THRESH_BINARY)
        # 转为batch为1，通道为1，大小为512*512的数组
        # img = cv2.resize(img, (534, 534))
        img = img.reshape(1, 1, img.shape[0], img.shape[1])
        # label = label.reshape(1, 1, label.shape[0], label.shape[1])
        
        # 转为tensor
        img_tensor = torch.from_numpy(img)
        # label_tensor = torch.from_numpy(label)
        # 将tensor拷贝到device中，只用cpu就是拷贝到cpu中，用cuda就是拷贝到cuda中。
        img_tensor = img_tensor.to(device=device, dtype=torch.float32)
        
        
        # 预测
        pred = net(img_tensor)
        # 提取结果
        pred = np.array(pred.data.cpu()[0])[0]
        # label = np.array(label_tensor.data.cpu()[0])[0]
        pred[pred > 0.2] = 255
        pred[pred <= 0.2] = 0
        image_array[:, tests_path.index(test_path),:] = pred
        pred_area = np.sum(pred == 255)
        # save_path = os.path.join(save_root,os.path.basename(test_path))
        # cv2.imwrite(save_path,pred)
        # print(test_path,pred_area,"pixels")
        pixel_volume = pixel_spacing[0]*pixel_spacing[1]*pixel_spacing[2]
        volume += pred_area*pixel_volume
        ids += 1
    save_path = os.path.join(output_dir,'image_array.npy')
    np.save(save_path,image_array)
    print("3D image saved to",save_path)
        # label_area = np.sum(label == 255)*pixel_size*pixel_size
    return volume

def Generate3D(image_dir,output_dir):
    # 读取png文件
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    png_files = [f for f in os.listdir(image_dir) if f.endswith('.png')]
    if len(png_files) == 0:
        print("No png files found in the directory.")
        return
    first_img = Image.open(os.path.join(image_dir,png_files[0]))
    # width, height = first_img.size
    first_array = np.array(first_img)
    depth = len(png_files)

    volume = np.zeros((first_array.shape[0],first_array.shape[1],depth),dtype=np.uint16)
    for i,png_file in enumerate(png_files):
        img = Image.open(os.path.join(image_dir,png_file))
        # img.convert("L")
        img_array = np.array(img,dtype=np.uint16)
        volume[:,i,:] = img_array

    save_path = os.path.join(output_dir,'image_array.npy')
    np.save(save_path,volume)
    print("3D image saved to",save_path)
    

    # print("3D image saved to",save_path)
    # return save_path

def numpy2stl(volume,output_dir):
    m = mesh.Mesh(np.zeros((volume.shape[0]*volume.shape[1]*volume.shape[2],3), dtype=mesh.Mesh.dtype))

    idx = 0
    for i in range(volume.shape[0]):
        print("processing slice",i)
        for j in range(volume.shape[1]):
            for k in range(volume.shape[2]):
                m.vectors[idx] = [i,j,k]
                idx += 1
    m.save(os.path.join(output_dir,'mesh.stl'))

def STL_show(stl_path):
    your_mesh = trimesh.load_mesh(stl_path)
    if your_mesh.is_empty:
        print("Empty mesh")
        return
    # 显示3D模型
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    for face in your_mesh.faces:
        vertices = your_mesh.vertices[face]
        ax.add_collection3d(plt.Polygon(vertices, closed=True, color='b',alpha=0.1))

    ax.set_aspect('auto')
    plt.show()

def main():
    paser = argparse.ArgumentParser()
    paser.add_argument('-m','--model_path', type=str, default='models\\MishUNet_model_latest_2.pth', help='模型路径')
    paser.add_argument('-d','--dcm_path', type=str, default='dcms\\20241130.dcm', help='dcm文件路径')
    paser.add_argument('-s','--slices_path', type=str, default='slices', help='切片文件路径')
    paser.add_argument('-o','--output_dir', type=str, default='output', help='输出路径')
    paser.add_argument('-p','--pixel_spacing', type=tuple, default=(0.015,0.015,0.015),  help='像素间距')
    paser.add_argument('--need_mask', type=bool, default=False, help='是否需要保存预测结果图片')
    args = paser.parse_args()
    model_path = args.model_path
    dcm_path = args.dcm_path
    output_dir = args.output_dir
    pixel_spacing = args.pixel_spacing
    slices_path = args.slices_path
    need_mask = args.need_mask

    if not os.path.exists(model_path):
        print("模型不存在")
        if not os.path.exists(os.path.dirname(model_path)):
            os.makedirs(os.path.dirname(model_path))
        print("请将模型放在模型目录:",os.path.dirname(model_path))
        return
    
    if not os.path.exists(slices_path):
        os.makedirs(slices_path)
    slices_dir = os.path.basename(dcm_path).split('.')[0]
    slices_path = os.path.join(slices_path,slices_dir)
    
    if not os.path.exists(slices_path):
        if not os.path.exists(dcm_path):
            print("dcm文件不存在")
            return
        else:
            print("切片文件不存在，开始切片")
            dcm = split_dcm.load_dicom_file(dcm_path)
            
            if not os.path.exists(slices_path):
                os.makedirs(slices_path)
            split_dcm.save_slices(dcm, 1, 'coronal', slices_path)
            print("切片文件保存到",slices_path)
    print("dcm文件路径:",dcm_path)
    print("切片文件路径:",slices_path)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    basename = os.path.basename(slices_path)
    output_dir = os.path.join(output_dir,basename) # 输出路径 output/20241130
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 加载网络
    net,device = init_net(model_path)
    tests_path = glob.glob(slices_path + '/*.png')
# 保存3D数组
    save_path = os.path.join(output_dir,'image_array.npy')  # 三维数组输出路径 output/20241130/image_array.npy
    
    if need_mask:
        # 预测并保存预测结果图片
        pred_path = os.path.join(output_dir,'pred')  #预测结果路径 output/20241130/pred
    # 预测
        if not os.path.exists(pred_path):
            os.makedirs(pred_path)
        volume = predict(net,device, tests_path,pred_path,pixel_spacing)
        print("开始保存3D数组")
        Generate3D(pred_path,output_dir=output_dir)
        
    else:
        # 预测并保存3D数组
        volume = predict_nosave(net,device, tests_path,output_dir,pixel_spacing)
    
    print("体积:",volume,"cm^3")
    volumn = np.load(save_path)
    
       #点云生成
    print("开始生成点云")
    output_dir3D = os.path.join(output_dir,'3D')    # 点云输出路径 output/20241130/3D
    if not os.path.exists(output_dir3D):
        os.makedirs(output_dir3D)
    nd2pcld(volumn,output=os.path.join(output_dir3D,'image_array.pcd'))
    # print("总体体积:",volume,"cm^3")


if __name__ == "__main__":
    # 选择设备，有cuda用cuda，没有就用cpu
    main()