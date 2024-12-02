# frontal_sinus
##上额窦CT影响测量

首先，请确保下载了conda，创建conda虚拟环境：

```
conda create -n frontal_sinus python=3.9

```

激活虚拟环境：
```
conda activate frontal_sinus
```


安装依赖：
```
pip install -r requirements.txt
```


执行git clone命令：

```
git clone https://github.com/yuanming-hu/frontal_sinus.git
```

然后cd进入到frontal_sinus目录下，运行命令：
```
python init.py
```
请将需要测量的dcm文件放在/dcms文件夹下；模型放在/models文件夹下，然后运行如以下命令

linux：
```
python volume.py -m models/XXX.pth -d dcms/XXX.dcm -o output 
```

windows：
```
python volume.py -m models\XXX.pth -d dcms\XXX.dcm -o output 
```

查看已经生成的3D点云文件请输入以下命令
3D点云*.pcd文件默认存放在 output/XXX/3D/XXX.pcd 'XXX'是dcm文件名
```
python show3D.py --pcd_file pcd文件的所在路径 
```