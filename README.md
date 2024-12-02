# frontal_sinus

首先，请确保下载了conda，创建conda虚拟环境：

conda create -n frontal_sinus python=3.9

激活虚拟环境：

conda activate frontal_sinus

安装依赖：

pip install -r requirements.txt

执行git clone命令：
git clone https://github.com/yuanming-hu/frontal_sinus.git


然后cd进入到frontal_sinus目录下，运行命令：

python volume.py

首次运行后会在当前工作目录下创建多个文件夹，请将需要测量的dcm文件放在/dcms文件夹下；模型放在/models文件夹下，