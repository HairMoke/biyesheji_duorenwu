import os
import glob


def delete_files_in_folders(base_dir, file_pattern):
    # 遍历S01到S64的文件夹
    for i in range(1, 65):
        folder_name = f"S{i:02d}"  # 格式化文件夹名为S01到S64
        folder_path = os.path.join(base_dir, folder_name)

        if os.path.exists(folder_path) and os.path.isdir(folder_path):
            # 使用glob查找匹配的文件
            file_paths = glob.glob(os.path.join(folder_path, file_pattern))

            for file_path in file_paths:
                try:
                    os.remove(file_path)
                    print(f"Deleted: {file_path}")
                except Exception as e:
                    print(f"Error deleting {file_path}: {e}")
        else:
            print(f"Folder not found or not a directory: {folder_path}")


if __name__ == "__main__":
    base_dir = "THU"  # 基础目录
    file_pattern = "y_ftr_train.npy"  # 要删除的文件名或模式

    delete_files_in_folders(base_dir, file_pattern)
