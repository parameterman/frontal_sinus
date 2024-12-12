import os
import open3d
import open3d.visualization
import numpy as np
import argparse
import sys
import subprocess
from multiprocessing import Process
def run_script(pcd_file):
    os.system('python show3D.py --pcd_file '+pcd_file)



def nd2pcld(nd_array, output):
    if nd_array.ndim != 3:
        raise ValueError("nd_array should have 3 dimensions")
    height, width, depth = nd_array.shape
    pixel_num = np.sum(nd_array > 0)
    points = np.zeros((pixel_num+1, 3))
    colors = np.zeros((pixel_num+1, 3))
    # points = np.array([[0,0,0]])
    print("points shape:",points.shape)
    # colors = np.array([])
    num = 0
    for i in range(height):
        # print(i)
        print_progress(i+1, height, prefix='Progress:', suffix='Complete', length=50)
        for j in range(width):
            for k in range(depth):
                if num >= pixel_num:
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

    
def projection(pcd):#求出在三个方向上的投影 输入为
    
    #读取pcd文件
    #计算三个方向的投影 proj_xy，proj_yz，proj_xz，shape=(n,3)
    proj_xy = np.dot(pcd.points, np.array([[1,0,0],[0,1,0],[0,0,0]]))
    proj_yz = np.dot(pcd.points, np.array([[0,0,0],[0,1,0],[0,0,1]]))
    proj_xz = np.dot(pcd.points, np.array([[1,0,0],[0,0,0],[0,0,1]]))

    print('projection done')
    print('proj_xy shape:',proj_xy.shape)
    print('proj_yz shape:',proj_yz.shape)
    print('proj_xz shape:',proj_xz.shape)
    #去除投影上重复的点，只保留一个，将所有的坐标以数组保存  proj_xy_arr，proj_yz_arr，proj_xz_arr，shape=(m,3)
    proj_xy_arr = np.unique(proj_xy, axis=0)
    proj_yz_arr = np.unique(proj_yz, axis=0)
    proj_xz_arr = np.unique(proj_xz, axis=0)

    print('projection unique done')
    print('proj_xy_arr shape:',proj_xy_arr.shape)
    print('proj_yz_arr shape:',proj_yz_arr.shape)
    print('proj_xz_arr shape:',proj_xz_arr.shape)
    #返回三个方向的投影数组
    return proj_xy_arr, proj_yz_arr, proj_xz_arr

def projDiff(arr1, arr2,plane):
    #输入同一个方向投影的两个数组 数组元素为点在投影平面上的坐标

    #求出两个数组的并集-交集
    #并集
    map1 = np.zeros((534,534))
    map2 = np.zeros((534,534))
    if plane == 'xy':
        for i in range(len(arr1)):
            map1[int(arr1[i][0]),int(arr1[i][1])] = 1
        for i in range(len(arr2)):
            map2[int(arr2[i][0]),int(arr2[i][1])] = 1
    elif plane == 'yz':
        for i in range(len(arr1)):
            map1[int(arr1[i][1]),int(arr1[i][2])] = 1
        for i in range(len(arr2)):
            map2[int(arr2[i][1]),int(arr2[i][2])] = 1
    else:
        for i in range(len(arr1)):
            map1[int(arr1[i][0]),int(arr1[i][2])] = 1
        for i in range(len(arr2)):
            map2[int(arr2[i][0]),int(arr2[i][2])] = 1
        #交集
    union_arr = np.where(map1+map2>=1)
    print('union_arr shape:',union_arr[0].shape)
    #将x和y坐标合并
    
    union_arr = np.vstack((union_arr[0],union_arr[1])).T
    u_map = np.zeros((534,534))
    for i in range(len(union_arr)):
        u_map[int(union_arr[i][0]),int(union_arr[i][1])] = 1
    
    print('union_arr shape:',union_arr.shape)
    print('union_arr:',union_arr)
    #交集
    
    intersect_arr = np.where(map1+map2==2)
    print('intersect_arr shape:',intersect_arr[0].shape)
    i_map = np.zeros((534,534))
    intersect_arr = np.vstack((intersect_arr[0],intersect_arr[1])).T
    for i in range(len(intersect_arr)):
        i_map[int(intersect_arr[i][0]),int(intersect_arr[i][1])] = 1
    print('intersect_arr shape:',intersect_arr.shape)
    print('intersect_arr:',intersect_arr)
    # #求出差集
    
    # diff_arr = np.where(map1+map2==1)
    diff_arr = np.where(u_map+i_map==1)
    diff_arr = np.vstack((diff_arr[0],diff_arr[1])).T
    print('diff_arr shape:',diff_arr.shape)
    #展示3D点云
    # pcd = o3d.geometry.PointCloud()
    # pcd.points = o3d.utility.Vector3dVector(diff_arr)
    # o3d.visualization.draw_geometries([pcd])
    # #返回差集数组
    return diff_arr
    # passwsddd

def deleteDiff(cloud_map, diff_arr,plane):#从点云中删除差集对应的点
    #输入原始点云数组，差集数组 差集数组只包含(x,y)，需要去除原始点云数组中所有(x,y)坐标相同的点
    #原始点云数组元素为(x,y,z)
    #将点云转换为numpy数组(534,534,534)
    
    #删除原始点云数据中与差集数组中(x,y)坐标相同的点
    for i in range(len(diff_arr)):
        # print(len(diff_arr),diff_arr.shape)
        print_progress(i, len(diff_arr))
        dim1 = diff_arr[i][0]
        dim2 = diff_arr[i][1]
            
        if plane == 'xy':
            #将cloud——map中所有(x,y)坐标相同的点置为0
            cloud_map[int(dim1),int(dim2),:] = 0
        
        elif plane == 'yz':
            cloud_map[:,int(dim1),int(dim2)] = 0
        
        elif plane == 'xz':
            cloud_map[int(dim1),:,int(dim2)] = 0
        else:
            print('plane error')
            break
        #将(534,534,534)的numpy数组转换为点云数组
    
        #找到原始点云数组中所有(x,y)坐标相同的点
        #删除原始点云数组中所有(x,y)坐标相同的点
        # origin_arr = np.delete(origin_arr, index, axis=0)
        #返回删除差集后的点云数组
        
    return cloud_map
    #返回删除差集后的点云数组    


def main():
    #输入需要对比的两个原始点云文件，pcd_before.pcd和pcd_after.pcd文件
    parser = argparse.ArgumentParser(description='setDiff')
    parser.add_argument('--before', default='output\\before20241202\\3D\\image_array.pcd', help='before pcd file')
    parser.add_argument('--after', default='output\\after20241202\\3D\\image_array.pcd', help='after pcd file')
    parser.add_argument('--pixel_size', default=0.015, help='pixel size')
    #求出两个文件对应的三个方向的投影 proj_xy_before，proj_yz_before，proj_xz_before，proj_xy_after，proj_yz_after，proj_xz_after
    args = parser.parse_args()
    output_dir_before = os.path.dirname(args.before)
    output_dir_after = os.path.dirname(args.after)
    pixel_size = float(args.pixel_size)
    pcd_before = open3d.io.read_point_cloud(args.before)
    pcd_after = open3d.io.read_point_cloud(args.after)
    proj_xy_before, proj_yz_before, proj_xz_before = projection(pcd_before)
    proj_xy_after, proj_yz_after, proj_xz_after = projection(pcd_after)

    #求出三个方向的差集 proj_xy_diff，proj_yz_diff，proj_xz_diff
    proj_xy_diff = projDiff(proj_xy_before, proj_xy_after, 'xy')
    proj_yz_diff = projDiff(proj_yz_before, proj_yz_after, 'yz')
    proj_xz_diff = projDiff(proj_xz_before, proj_xz_after, 'xz')

    #删除原始点云文件中三个方向的差集对应的点
    

    pcd_before_arr = np.array(pcd_before.points)
    pcd_after_arr = np.array(pcd_after.points)
    print('pcd_before_arr shape:',pcd_before_arr.shape)
    print('pcd_after_arr shape:',pcd_after_arr.shape)
    #将原始点云数组转换为(534,534,534)的numpy数组
    cloud_map_before = np.zeros((534,534,534))
    for i in range(len(pcd_before_arr)):
        cloud_map_before[int(pcd_before_arr[i][0]),int(pcd_before_arr[i][1]),int(pcd_before_arr[i][2])] = 1
    
    cloud_map_after = np.zeros((534,534,534))
    for i in range(len(pcd_after_arr)):
        cloud_map_after[int(pcd_after_arr[i][0]),int(pcd_after_arr[i][1]),int(pcd_after_arr[i][2])] = 1

    print('cloud_map_before shape:',cloud_map_before.shape)
    print('cloud_map_after shape:',cloud_map_after.shape)
    volume_before = len(pcd_before_arr)-len(pcd_after_arr)
    volume_before = volume_before*pixel_size*pixel_size*pixel_size
    
    cloud_map_before = deleteDiff(cloud_map_before, proj_xy_diff, 'xy')
    cloud_map_before = deleteDiff(cloud_map_before, proj_yz_diff, 'yz')
    cloud_map_before = deleteDiff(cloud_map_before, proj_xz_diff, 'xz')
    cloud_map_after = deleteDiff(cloud_map_after, proj_xy_diff, 'xy')
    cloud_map_after = deleteDiff(cloud_map_after, proj_yz_diff, 'yz')
    cloud_map_after = deleteDiff(cloud_map_after, proj_xz_diff, 'xz')
    
    #将(534,534,534)的numpy数组转换为点云数组
    sum_before = np.sum(cloud_map_before==1)
    sum_after = np.sum(cloud_map_after==1)
    print('sum_before:',sum_before)
    print('sum_after:',sum_after)
    # cloud_map_before = np.where(cloud_map_before==1)
    # print('cloud_map_before shape:',cloud_map_before[0].shape,cloud_map_before[1].shape,cloud_map_before[2].shape)
    # cloud_map_after = np.where(cloud_map_after==1)
    # print('cloud_map_after shape:',cloud_map_after[0].shape,cloud_map_after[1].shape,cloud_map_after[2].shape)
    # print('cloud_map_after shape:',cloud_map_after.shape)
    #将点云数组转换为open3d点云对象
    nd2pcld(cloud_map_before, os.path.join(output_dir_before, 'pcd_before_diff.pcd'))
    nd2pcld(cloud_map_after, os.path.join(output_dir_after, 'pcd_after_diff.pcd'))

    before_save_path = os.path.join(output_dir_before, 'pcd_before_diff.pcd')
    after_save_path = os.path.join(output_dir_after, 'pcd_after_diff.pcd')
    # subprocess.run('python show3D.py --pcd_file '+os.path.join(output_dir_before, 'pcd_before_diff.pcd'))
    # subprocess.run('python show3D.py --pcd_file '+os.path.join(output_dir_after, 'pcd_after_diff.pcd'))

    volume = sum_before - sum_after
    volume = volume*pixel_size*pixel_size*pixel_size
    print('volume_before:',volume_before,'cm^3')
    print('volume:',volume,'cm^3')
    return  before_save_path, after_save_path
    #保存删除差集后的点云文件 pcd_before_diff.pcd，pcd_after_diff.pcd
if __name__ == '__main__':
    before_save_path, after_save_path = main()
    p1 = Process(target=run_script, args=(before_save_path,))
    p1.start()
    # p.join()
    p2 = Process(target=run_script, args=(after_save_path,))
    p2.start()
