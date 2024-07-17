import os
import shutil
import sys
import re

def create_and_move_images(source_folder):
    # 获取源文件夹中所有文件
    files = os.listdir(source_folder)
    
    # 遍历每个文件
    for file in files:
        # 提取文件名中的中文人名
        person_name = ''.join(re.findall(r'[\u4e00-\u9fff]', file))
        
        if person_name:
            # 创建目标文件夹路径
            target_folder = os.path.join(source_folder, person_name)
            
            # 如果目标文件夹不存在，则创建它
            if not os.path.exists(target_folder):
                os.makedirs(target_folder)
            
            # 构造源文件和目标文件的完整路径
            source_file = os.path.join(source_folder, file)
            target_file = os.path.join(target_folder, file)
            
            # 移动文件到目标文件夹
            shutil.move(source_file, target_file)
            print(f'Moved {file} to {target_folder}')


source_folder = sys.argv[1]
create_and_move_images(source_folder)