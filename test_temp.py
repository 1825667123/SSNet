import matplotlib.pyplot as plt
from matplotlib.font_manager import FontManager

# 打印字体扫描优先级和实际读取的字体路径
fm = FontManager()
# 查找msyh字体的实际路径
font_path = fm.findfont("msyh")
print(f"当前使用的msyh字体路径：{font_path}")
# 判断是否是个人目录下的字体
if "/home/guanxin/.fonts" in font_path:
    print("✅ 正在使用你的个人字体！")
else:
    print("❌ 使用的是系统全局字体（或未找到）")