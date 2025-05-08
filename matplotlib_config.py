import matplotlib
import matplotlib.pyplot as plt
import platform


def configure():
    # 解决负号显示问题
    matplotlib.rcParams['axes.unicode_minus'] = False

    # 配置中文字体支持
    system = platform.system()
    chinese_font_available = False

    if system == 'Windows':
        # Windows系统
        try:
            plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial']
            chinese_font_available = True
        except:
            try:
                plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial']
                chinese_font_available = True
            except:
                pass
    elif system == 'Darwin':
        # macOS系统
        try:
            plt.rcParams['font.sans-serif'] = ['PingFang SC', 'Heiti SC', 'Arial']
            chinese_font_available = True
        except:
            pass
    else:
        # Linux系统
        try:
            plt.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei', 'Noto Sans CJK SC', 'Arial']
            chinese_font_available = True
        except:
            pass

    # 备用方案：使用matplotlib的默认字体
    if not chinese_font_available:
        print("Cannot find suitable Chinese font. Using English instead.")

    return chinese_font_available