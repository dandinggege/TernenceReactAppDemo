import os
from pathlib import Path

def split_file(file_path, chunk_size=20*1024*1024):
    """
    将文件拆分为指定大小的块

    Args:
        file_path: 要拆分的文件路径
        chunk_size: 每个块的大小，默认为20MB (20*1024*1024 bytes)

    Returns:
        list: 拆分后的文件路径列表
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"文件 {file_path} 不存在")

    chunk_files = []
    with open(file_path, 'rb') as f:
        file_size = os.path.getsize(file_path)
        num_chunks = (file_size + chunk_size - 1) // chunk_size  # 向上取整

        print(f"开始拆分文件 {file_path}, 总大小: {file_size} 字节, 共需 {num_chunks} 个块")

        for i in range(num_chunks):
            chunk_data = f.read(chunk_size)
            if not chunk_data:  # 如果没有更多数据了，跳出循环
                break

            chunk_filename = f"{file_path.stem}.{i:03d}.part"
            chunk_filepath = file_path.parent / chunk_filename

            with open(chunk_filepath, 'wb') as chunk_f:
                chunk_f.write(chunk_data)

            chunk_files.append(str(chunk_filepath))
            print(f"已创建分块文件: {chunk_filepath} ({len(chunk_data)} 字节)")

    return chunk_files


def combine_files_from_directory(directory_path, output_path=None):
    """
    从指定目录中读取拆分后的文件并合并

    Args:
        directory_path: 包含分块文件的目录路径
        output_path: 输出文件路径，如果不指定则自动确定

    Returns:
        str: 合并后的文件路径
    """
    dir_path = Path(directory_path)

    if not dir_path.exists() or not dir_path.is_dir():
        raise FileNotFoundError(f"目录 {directory_path} 不存在或不是一个目录")

    # 获取所有 .part 文件
    part_files = list(dir_path.glob("*.part"))

    if not part_files:
        raise FileNotFoundError(f"在目录 {directory_path} 中未找到 .part 分块文件")

    # 按文件名排序以确保正确的合并顺序
    part_files.sort(key=lambda x: int(x.stem.split('.')[-1]))

    # 如果没有指定输出路径，则尝试从第一个分块文件推断原始文件名
    if output_path is None:
        first_part = part_files[0].stem
        original_name = '.'.join(first_part.split('.')[:-1])
        output_path = dir_path / f"{original_name}.safetensors"
    else:
        output_path = Path(output_path)

    print(f"找到 {len(part_files)} 个分块文件，开始合并...")

    with open(output_path, 'wb') as output_f:
        for i, part_file in enumerate(part_files):
            print(f"正在合并第 {i + 1}/{len(part_files)} 个文件: {part_file}")
            with open(part_file, 'rb') as part_f:
                output_f.write(part_f.read())

    print(f"合并完成! 输出文件: {output_path}")
    return str(output_path)


if __name__ == "__main__":
    # split_file("/Users/mac/Desktop/model.safetensors")
    combine_files_from_directory("/Users/mac/Desktop/models")