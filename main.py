import numpy as np
import cv2
import os



npy_folder = "/home/zhang//onnx/person/"

def get_aspect_scaled_ratio(src_w, src_h, dst_w, dst_h):
    r_w = dst_w / src_w
    r_h = dst_h / src_h
    isAlignWidth = r_h > r_w
    if isAlignWidth:
        ratio = r_w
    else:
        ratio = r_h
    return ratio, isAlignWidth

    
def images_to_npy(image_folder, npy_folder, txt_path):
    # 创建.npy文件夹
    if not os.path.exists(npy_folder):
        os.makedirs(npy_folder)

    # 获取图片文件夹中所有图片的文件名
    image_files = [f for f in os.listdir(image_folder) if f.endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
    
    # 打开一个文件以保存图片的绝对路径
    with open(txt_path, 'w') as f:
        for image_file in image_files:
            # 读取图片
            img = cv2.imread(os.path.join(image_folder, image_file))

            if img is None:
                print(f"Error: Unable to read image from {image_file}")
                continue


            isAlignWidth = False
            m_net_h = 640
            m_net_w =640
            ratio, isAlignWidth = get_aspect_scaled_ratio(img.shape[1], img.shape[0], m_net_w, m_net_h)
            temp_img = np.full((m_net_h, m_net_w, img.shape[2]), 114, dtype=np.uint8)
            
            if isAlignWidth:
                new_height = int(img.shape[0] * ratio)
                ty1 = (m_net_h - new_height) // 2
                img = cv2.resize(img, (m_net_w, new_height))
                temp_img[ty1:ty1+new_height, :] = img
            else:
                new_width = int(img.shape[1] * ratio)
                tx1 = (m_net_w - new_width) // 2
                img = cv2.resize(img, (new_width, m_net_h))
                temp_img[:, tx1:tx1+new_width] = img
                
                
            # 将图片转换为numpy数组并转换为float32
            img_array = np.array(temp_img, dtype=np.float32) / 255.0

            # 调整数据形状
            img_array = img_array.transpose((2, 0, 1))  # 将形状从 HWC 转换为 CHW

            # 保存为.npy文件
            npy_path = os.path.join(npy_folder, os.path.splitext(image_file)[0] + '.npy')
            np.save(npy_path, img_array)
            
            corrected_path = npy_path.replace("\\", "/")
            print(f"Image saved as {npy_path}")

            # 写入图片的绝对路径到txt文件
            f.write(npy_folder + corrected_path + '\n')

if __name__ == "__main__":
    image_folder = "img"  # 替换为您的图片文件夹路径
    npy_folder = "npy"  # 替换为您希望保存.npy文件的文件夹路径
    txt_path = "paths.txt"  # 存储图片绝对路径的txt文件路径

    images_to_npy(image_folder, npy_folder, txt_path)
    
