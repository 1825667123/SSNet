import os


def print_directory_tree(root_path, prefix=""):
    """
    打印指定路径的文件夹结构树（排除.inp结尾的文件）

    参数:
        root_path: 要查看的根文件夹路径
        prefix: 用于格式化输出的前缀字符串
    """
    # 检查路径是否存在
    if not os.path.exists(root_path):
        print(f"错误: 路径 '{root_path}' 不存在")
        return

    # 检查是否是目录
    if not os.path.isdir(root_path):

        print(f"错误: '{root_path}' 不是一个目录")
        return

    # 获取目录下的所有项目（文件和子目录），并过滤掉.inp结尾的文件
    items = []
    for item in os.listdir(root_path):

        # 排除.inp结尾的文件（目录不受影响）
        # item_path = os.path.join(root_path, item)
        # # if os.path.isfile(item_path) and item.endswith(".inp") or item.endswith(".dat"):
        # #     continue  # 跳过.inp文件
        items.append(item)

    # 排序：目录在前，文件在后，均按名称排序
    items.sort(key=lambda x: (not os.path.isdir(os.path.join(root_path, x)), x.lower()))

    for i, item in enumerate(items):
        item_path = os.path.join(root_path, item)
        # 判断是否是最后一个项目
        is_last = i == len(items) - 1

        # 确定前缀符号
        if is_last:
            line_prefix = prefix + "└── "
            next_prefix = prefix + "    "
        else:
            line_prefix = prefix + "├── "
            next_prefix = prefix + "│   "

        # 打印当前项目
        print(line_prefix + item)

        # 如果是目录，递归处理
        if os.path.isdir(item_path):
            print_directory_tree(item_path, next_prefix)


if __name__ == "__main__":
    # 可以直接修改这里的路径，或通过输入获取
    target_path = input("请输入要查看的文件夹路径 (直接回车查看当前目录): ").strip()

    # 如果未输入路径，使用当前工作目录
    if not target_path:
        target_path = os.getcwd()

    print(f"\n文件夹结构树: {target_path}")
    print("-" * (len(target_path) + 12))  # 打印分隔线
    print_directory_tree(target_path)