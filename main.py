import streamlit as st
from streamlit_echarts import st_echarts
import os
import re
import csv
import json
import base64
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import tempfile
import time
from integrated_analysis import IntegratedAnalysis
from docx import Document
import fitz  # PyMuPDF
from PyPDF2 import PdfReader

# ==================== 可靠性分析核心代码 ====================
# 可靠性关键词库
ASSURANCE = {
    "鉴证声明": r"第三方鉴证|独立鉴证|外部审验|鉴证[报告意见]|审验|评级|专家点评",
    "鉴证机构": r"质量认证中心|CQC|SGS|必维|BSI|TÜV|Bureau Veritas|DNV",
    "鉴证标准": r"AA1000|ISO\d{4,}|ISAE\s?\d+|鉴证标准",
    "签署信息": r"授权人签名|签字[：:]|签署[：:]|Signature"
}
STAKEHOLDER = {
    "反馈渠道": r"读者意见反馈|意见反馈|意见征集|听取意见|反馈|建议|意见反馈表"
}
COMMITMENT = {
    "承诺主体": r"董事会|管理层|本公司|本企业",
    "承诺动词": r"保证|承诺|声明|确保",
    "承诺内容": r"真实|不虚假|无重大遗漏|准确完整",
    "责任表述": r"承担.*责任|法律责任|连带责任"
}

def _reliability_hits(patts: dict, text: str) -> int:
    """返回 patts 中匹配到的类别数"""
    return sum(1 for k, v in patts.items() if re.search(v, text, flags=re.I))

def reliability_score(text: str) -> dict:
    """可靠性评分主函数"""
    text = text.strip()
    tail_3k = text[-8000:]
    tail_1k = text[-5000:]
    E = 1 if _reliability_hits(ASSURANCE, tail_3k) >= 2 else 0
    S = 1 if _reliability_hits(STAKEHOLDER, tail_1k) >= 1 else 0
    A = 1 if _reliability_hits(COMMITMENT, text) >= 3 else 0
    R = (E + S + A) / 3
    return {"外部鉴证": E, "利益相关方": S, "真实性承诺": A, "可靠性R": round(R, 2)}

def reliability_process_files(input_dir: str, progress_callback=None) -> pd.DataFrame:
    """批量处理目录下的所有txt文件进行可靠性分析"""
    dir_path = Path(input_dir)
    results = []
    txt_files = list(dir_path.glob('*.txt'))
    total_files = len(txt_files)
    
    print(f"可靠性分析: 找到 {total_files} 个文件于 {input_dir}")
    
    for i, file_path in enumerate(txt_files):
        print(f"  处理可靠性: {file_path.name}")
        if progress_callback:
            progress = (i + 1) / total_files
            progress_callback(progress, file_path.name)
        
        try:
            text = file_path.read_text(encoding='utf8')
            score = reliability_score(text)
            results.append({
                '文件名': file_path.name,
                '外部鉴证(E)': score['外部鉴证'],
                '利益相关方(S)': score['利益相关方'],
                '真实性承诺(A)': score['真实性承诺'],
                '综合可靠性(R)': score['可靠性R']
            })
        except Exception as e:
            print(f"处理文件 {file_path.name} 时出错: {e}")
            results.append({
                '文件名': file_path.name,
                '外部鉴证(E)': 0,
                '利益相关方(S)': 0,
                '真实性承诺(A)': 0,
                '综合可靠性(R)': 0
            })
    
    return pd.DataFrame(results)

# ==================== 可读性分析核心代码 ====================
# 可读性规则正则
TOC = {
    "目录声明": r"目录|目次|CONTENTS|报告结构",
    "章节标题": r"^\s*[第（]?\d+[章部分节][）]?\s+.+$",
    "页码标注": r"\d+[页Pp]?$|\.\.\.\s*\d+$|^\s*\d+\s*$|第\d+页"
}
FIGURE = {
    "图表标题": r"[图表] *\d+[\-–—]\d+|图\s*\d+|表\s*\d+|示意图|统计图",
    "图表引用": r"见图|如表|如下图|如下表|参见图|详见附表"
}
TERM = {
    "术语定义": r"是指|即|简称|英文全称|以下简称|缩写为",
    "术语表": r"术语表|词汇表|附录[一1]?\s*[：:]?\s*关键术语|附录[一1]?\s*[：:]?\s*名词解释"
}

# 视觉模型常量（用于统计PDF中的图片和表格）
VISION_SYS_PROMPT = """
你是一名「报表视觉解析机器人」你能清楚分辨图片和表格。
任务是：
1：统计图片和表格数目：
    a.统计真正的「照片/效果图/实景图」数量
    b.排除小 logo、流程箭头、文本框、纯图标。  
    c.文本框里面有图片要按图片计算（计入图片数目），文本框里面没有图片要按文本段落计算（不计入）。
    d.图片尺寸大于50x50才进行计数
2. 统计「数据表」数量，必须同时满足下列所有条件：  
   a. 矩形网格，≥2 行×≥2 列（含表头）；  
   b. 存在横向+纵向对齐线（实线、虚线或隐形对齐线均可）；  
   c. 不是流程图、甘特图、组织架构图、纯排版线、页眉页脚线；  
   d. 排除“文本段+外框”式排版框（即仅用于美化而非数据展示的框）。
对每页图片逐页思考后，给出一行 JSON：
{"page_no":<int>,"photos":<int>,"tables":<int>}
最后额外输出一行汇总：
{"total_pages":<int>,"total_photos":<int>,"total_tables":<int>}
除上述 JSON 外，不要有任何解释、标题、引号。
"""
VISION_MODEL = "moonshot-v1-32k-vision-preview"
KIMI_API_KEY = "sk-Pk59EU0pxAQzR20oosWfRYNE3dxjHwt2mAiAeal8IgcosmBX"
VISION_BASE_URL = "https://api.moonshot.cn/v1"

def _readability_search(patts: dict, text: str) -> int:
    return sum(1 for v in patts.values() if re.search(v, text, flags=re.M | re.I))

def readability_score(text: str) -> dict:
    """可读性评分主函数"""
    text = text.strip()
    head_5k = text[:8000]
    toc_hit = _readability_search({"目录声明": TOC["目录声明"]}, head_5k) >= 1
    heading = len(re.findall(TOC["章节标题"], head_5k, re.M))
    page_num = _readability_search({"页码标注": TOC["页码标注"]}, head_5k) >= 1
    C = 1 if toc_hit or heading >= 1 or page_num else 0
    fig_caption = len(re.findall(FIGURE["图表标题"], text))
    fig_ref = len(re.findall(FIGURE["图表引用"], text))
    V = 1 if fig_caption + fig_ref >= 2 else 0
    has_term = re.search(r"[A-Z]{2,}|[^\x00-\x7F]{2,}.*简称", text)
    explain = _readability_search(TERM, text) >= 1
    T = 1 if (not has_term) or explain else 0
    R_read = round((C + V + T) / 3, 2)
    return {"目录及排版": C, "图表使用": V, "术语解释": T, "可读性R": R_read}

def read_pdf_text_for_readability(file_path: Path) -> str:
    """读取PDF文本"""
    try:
        reader = PdfReader(file_path)
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    except Exception as e:
        print(f"PDF读取失败: {e}")
        return ""

def count_pdf_visual_elements(pdf_path: Path) -> tuple:
    """调用视觉模型API统计PDF中的图片和表格数量"""
    try:
        from openai import OpenAI
        doc = fitz.open(pdf_path)
        client = OpenAI(api_key=KIMI_API_KEY, base_url=VISION_BASE_URL)
        total_p = total_t = 0
        
        for page_no, page in enumerate(doc, 1):
            try:
                pix = page.get_pixmap(dpi=50)
                img_bytes = pix.tobytes("png")
                b64_str = base64.b64encode(img_bytes).decode()
                url = f"data:image/png;base64,{b64_str}"

                messages = [
                    {"role": "system", "content": VISION_SYS_PROMPT},
                    {"role": "user", "content": [
                        {"type": "text", "text": f"第 {page_no} 页"},
                        {"type": "image_url", "image_url": {"url": url}}
                    ]}
                ]

                resp = client.chat.completions.create(
                    model=VISION_MODEL,
                    messages=messages,
                    temperature=0
                )

                raw = resp.choices[0].message.content
                for line in reversed(raw.splitlines()):
                    line = line.strip().strip("```json").strip("```")
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                        if "photos" in data and "tables" in data:
                            total_p += data["photos"]
                            total_t += data["tables"]
                            break
                    except Exception:
                        continue
            except Exception as e:
                print(f"页面 {page_no} 视觉分析失败: {e}")
                continue
        
        doc.close()
        return total_p, total_t
    except Exception as e:
        print(f"视觉模型分析失败，将使用默认值: {e}")
        return 0, 0

def readability_process_folder(folder: str, save_csv: str, out_root: str, progress_callback=None):
    """批量处理目录下的PDF文件进行可读性分析"""
    root = Path(folder)
    csv_path = Path(save_csv) / "readability.csv"
    
    results = []
    files = list(root.glob("*.pdf"))
    total_files = len(files)
    
    if total_files == 0:
        print(f"警告: {folder} 目录下没有找到PDF文件")
        return results
    
    for i, file in enumerate(files):
        print(f"处理可读性: {file.name}")
        
        if progress_callback:
            progress = (i + 1) / total_files
            progress_callback(progress, file.name)
        
        try:
            text = read_pdf_text_for_readability(file)
            if not text:
                print(f"警告: {file.name} 文本提取为空")
            
            # 调用视觉API统计图片和表格数量
            img_cnt, tbl_cnt = count_pdf_visual_elements(file)
            print(f"  视觉分析: 图片={img_cnt}, 表格={tbl_cnt}")
            
            score = readability_score(text)
            
            # 如果图片数量超过5张，调整图表使用评分
            if img_cnt > 5:
                score["图表使用"] = 1
                score["可读性R"] = round((score["目录及排版"] + 1 + score["术语解释"]) / 3, 2)
            
            # 文件名转换：PDF后缀改为TXT以便匹配
            file_name_txt = file.stem + ".txt"
            
            res = {
                "文件名": file_name_txt,
                "C": score["目录及排版"],
                "V": score["图表使用"],
                "T": score["术语解释"],
                "图片数量": img_cnt,
                "表格数量": tbl_cnt,
                "R_read": score["可读性R"]
            }
            results.append(res)
            print(f"  结果: C={res['C']}, V={res['V']}, T={res['T']}, R_read={res['R_read']}")
        except Exception as e:
            print(f"处理 {file.name} 出错: {e}")
    
    if results:
        print(f"可读性分析完成，共 {len(results)} 条结果")
        print(f"保存到: {csv_path}")
        with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)
        print(f"可读性CSV已保存: {csv_path}")
    else:
        print("可读性分析无结果")
    
    return results

# 设置页面配置
st.set_page_config(
    page_title="建筑业ESG报告披露质量评估系统",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# 创建临时文件夹用于存放上传的文件
TEMP_DIR = tempfile.mkdtemp()
OUTPUT_DIR = Path("综合评价结果")
OUTPUT_DIR.mkdir(exist_ok=True)

# 历史记录目录
HISTORY_DIR = Path("历史分析记录")
HISTORY_DIR.mkdir(exist_ok=True)

def get_history_list():
    """获取历史记录列表"""
    history_files = list(HISTORY_DIR.glob("*.json"))
    # 按修改时间排序，最新的在前面
    history_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    return history_files

def save_analysis_history(results: dict, name: str = None):
    """保存分析结果到历史记录"""
    timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
    if name:
        filename = f"{name}_{timestamp}.json"
    else:
        filename = f"分析记录_{timestamp}.json"
    
    # 将DataFrame转换为字典
    save_data = {
        "timestamp": timestamp,
        "name": name or f"分析记录_{timestamp}",
        "results": {}
    }
    
    for key, value in results.items():
        if isinstance(value, pd.DataFrame):
            save_data["results"][key] = value.to_dict(orient='records')
        elif value is not None:
            save_data["results"][key] = value
    
    with open(HISTORY_DIR / filename, 'w', encoding='utf-8') as f:
        json.dump(save_data, f, ensure_ascii=False, indent=2)
    
    return filename

def load_analysis_history(filepath: Path) -> dict:
    """加载历史分析记录"""
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 将字典转换回DataFrame
    results = {}
    for key, value in data.get("results", {}).items():
        if isinstance(value, list):
            results[key] = pd.DataFrame(value)
        else:
            results[key] = value
    
    # 重新计算综合评分（使用最新的计算方式）
    if 'combined' in results and 'readability' in results and 'reliability' in results:
        # 创建映射字典
        readability_map = dict(zip(results['readability']['文件名'], results['readability']['R_read']))
        reliability_map = dict(zip(results['reliability']['文件名'], results['reliability']['综合可靠性(R)']))
        
        # 添加可读性评分
        results['combined']['可读性评分'] = results['combined']['文件名'].map(readability_map).fillna(0)
        
        # 添加可靠性评分
        results['combined']['可靠性评分'] = results['combined']['文件名'].map(reliability_map).fillna(0)
        
        # 重新计算综合评分（6个维度的平均值）
        # 注意：完整性评分和实质性评分的范围是0-2，需要除以2转换为0-1范围，以保证各维度权重相等
        results['combined']['综合评分'] = (results['combined']['情感评分'] + 
                                         results['combined']['完整性评分'] / 2 + 
                                         results['combined']['实质性评分'] / 2 + 
                                         results['combined']['可比性评分'] + 
                                         results['combined']['可读性评分'] + 
                                         results['combined']['可靠性评分']) / 6
    
    return {
        "timestamp": data.get("timestamp"),
        "name": data.get("name"),
        "results": results
    }

def delete_history_record(filepath: Path):
    """删除历史记录"""
    try:
        filepath.unlink()
        return True
    except Exception as e:
        print(f"删除失败: {e}")
        return False

# 初始化分析器（不预加载模型，延迟加载）
try:
    analyzer = IntegratedAnalysis()
    print("分析器初始化成功！")
except Exception as e:
    st.error(f"分析器初始化失败: {e}")
    analyzer = None

# 加载分析结果函数（需要在侧边栏代码之前定义）
def load_analysis_results():
    results = {}
    
    # 加载综合分析结果
    combined_path = os.path.join(OUTPUT_DIR, "combined_analysis_results.csv")
    if os.path.exists(combined_path):
        results['combined'] = pd.read_csv(combined_path, encoding='utf-8-sig')
    
    # 加载完整性分析结果
    integrity_path = os.path.join(OUTPUT_DIR, "integrity_analysis_results.csv")
    if os.path.exists(integrity_path):
        results['integrity'] = pd.read_csv(integrity_path, encoding='utf-8-sig')
    
    # 加载实质性分析结果
    substantive_path = os.path.join(OUTPUT_DIR, "substantive_analysis_results.csv")
    if os.path.exists(substantive_path):
        results['substantive'] = pd.read_csv(substantive_path, encoding='utf-8-sig')
    
    # 加载可比性分析结果
    comparability_path = os.path.join(OUTPUT_DIR, "comparability_results.csv")
    if os.path.exists(comparability_path):
        results['comparability'] = pd.read_csv(comparability_path, encoding='utf-8-sig')
    
    # 加载可读性分析结果
    readability_path = os.path.join(OUTPUT_DIR, "readability.csv")
    if os.path.exists(readability_path):
        results['readability'] = pd.read_csv(readability_path, encoding='utf-8-sig')
    
    # 加载可靠性分析结果
    reliability_path = os.path.join(OUTPUT_DIR, "reliability.csv")
    if os.path.exists(reliability_path):
        results['reliability'] = pd.read_csv(reliability_path, encoding='utf-8-sig')
    
    # 加载情感分析（平衡性分析）结果
    sentiment_path = os.path.join(OUTPUT_DIR, "sentiment_analysis_results.csv")
    if os.path.exists(sentiment_path):
        results['sentiment'] = pd.read_csv(sentiment_path, encoding='utf-8-sig')
    
    # 将可读性和可靠性评分整合到综合评分结果中
    if 'combined' in results and 'readability' in results and 'reliability' in results:
        # 创建映射字典
        readability_map = dict(zip(results['readability']['文件名'], results['readability']['R_read']))
        reliability_map = dict(zip(results['reliability']['文件名'], results['reliability']['综合可靠性(R)']))
        
        # 添加可读性评分
        results['combined']['可读性评分'] = results['combined']['文件名'].map(readability_map).fillna(0)
        
        # 添加可靠性评分
        results['combined']['可靠性评分'] = results['combined']['文件名'].map(reliability_map).fillna(0)
        
        # 重新计算综合评分（6个维度的平均值）
        # 注意：完整性评分和实质性评分的范围是0-2，需要除以2转换为0-1范围，以保证各维度权重相等
        results['combined']['综合评分'] = (results['combined']['情感评分'] + 
                                         results['combined']['完整性评分'] / 2 + 
                                         results['combined']['实质性评分'] / 2 + 
                                         results['combined']['可比性评分'] + 
                                         results['combined']['可读性评分'] + 
                                         results['combined']['可靠性评分']) / 6
    
    return results

# 创建侧边栏
st.sidebar.title("📁 文件上传")

# 文件上传区域
st.sidebar.markdown("### PDF文件上传")
pdf_files = st.sidebar.file_uploader(
    "选择PDF格式的ESG报告",
    type=["pdf"],
    accept_multiple_files=True,
    key="pdf_uploader"
)

st.sidebar.markdown("### TXT文件上传")
txt_files = st.sidebar.file_uploader(
    "选择TXT格式的ESG报告",
    type=["txt"],
    accept_multiple_files=True,
    key="txt_uploader"
)

# 自动保存上传的文件到汇总文件夹（上传时自动替换原有文件）
PDF_DIR = Path("汇总1")
TXT_DIR = Path("汇总")
PDF_DIR.mkdir(exist_ok=True)
TXT_DIR.mkdir(exist_ok=True)

# 保存PDF文件到汇总1文件夹（先清空原有文件）
if pdf_files:
    # 清空汇总1文件夹中的所有PDF文件
    for old_file in PDF_DIR.glob("*.pdf"):
        try:
            old_file.unlink()
        except Exception as e:
            st.sidebar.warning(f"无法删除旧文件 {old_file.name}: {e}")
    
    # 保存新上传的文件
    for pdf_file in pdf_files:
        pdf_save_path = PDF_DIR / pdf_file.name
        with open(pdf_save_path, "wb") as f:
            f.write(pdf_file.getbuffer())
    st.sidebar.success(f"✅ 已替换并保存 {len(pdf_files)} 个PDF文件到汇总1文件夹")

# 保存TXT文件到汇总文件夹（先清空原有文件）
if txt_files:
    # 清空汇总文件夹中的所有TXT文件
    for old_file in TXT_DIR.glob("*.txt"):
        try:
            old_file.unlink()
        except Exception as e:
            st.sidebar.warning(f"无法删除旧文件 {old_file.name}: {e}")
    
    # 保存新上传的文件
    for txt_file in txt_files:
        txt_save_path = TXT_DIR / txt_file.name
        with open(txt_save_path, "wb") as f:
            f.write(txt_file.getbuffer())
    st.sidebar.success(f"✅ 已替换并保存 {len(txt_files)} 个TXT文件到汇总文件夹")

# 权重设置选项
st.sidebar.markdown("### 分析权重设置")
use_custom_weights = st.sidebar.checkbox("使用自定义权重", value=False)

# 如果选择自定义权重，显示权重设置界面
if use_custom_weights:
    st.sidebar.markdown("#### 完整性分析权重")
    summary_weights = {}
    for label, name in analyzer.summary_label_map.items():
        summary_weights[label] = st.sidebar.slider(
            name,
            min_value=0.0,
            max_value=2.0,
            value=1.0,
            step=0.1,
            key=f"summary_{label}"
        )
    
    st.sidebar.markdown("#### 实质性分析权重")
    substantive_weights = {}
    for dimension in analyzer.substantive_dimensions:
        substantive_weights[dimension] = st.sidebar.slider(
            dimension,
            min_value=0.0,
            max_value=2.0,
            value=1.0,
            step=0.1,
            key=f"substantive_{dimension}"
        )
else:
    summary_weights = None
    substantive_weights = None

# 维度选择选项
st.sidebar.markdown("### 分析维度选择")
analysis_dimensions = st.sidebar.multiselect(
    "选择要进行的分析维度",
    options=[
        "完整性分析",
        "实质性分析",
        "可比性分析",
        "可读性分析",
        "可靠性分析",
        "平衡性分析"
    ],
    default=[
        "完整性分析",
        "实质性分析",
        "可比性分析",
        "可读性分析",
        "可靠性分析",
        "平衡性分析"
    ],
    help="勾选你要运行的分析维度，不勾选的维度将跳过"
)

# 初始化session_state保存已分析的维度（用于图表显示）
if 'displayed_dimensions' not in st.session_state:
    # 默认显示所有维度（显示已有结果）
    st.session_state.displayed_dimensions = [
        "完整性分析",
        "实质性分析",
        "可比性分析",
        "可读性分析",
        "可靠性分析",
        "平衡性分析"
    ]

# 分析按钮
analyze_button = st.sidebar.button("开始分析", type="primary", key="analyze_button")

# 数据导出功能（直接在侧边栏显示下载按钮）
st.sidebar.markdown("### 📥 数据导出")

# 初始化导出结果为None
export_results = None
export_file_name = None

# 根据当前状态决定导出哪个结果
# 这个导出按钮会在后面根据历史记录的加载情况进行更新
if 'export_data' in st.session_state:
    export_results = st.session_state.export_data
    export_file_name = st.session_state.export_file_name
else:
    # 默认导出当前分析结果
    try:
        export_results = load_analysis_results()
        export_file_name = f"ESG评估结果_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
    except Exception as e:
        print(f"加载当前结果失败: {e}")

# 显示导出按钮
try:
    if export_results:
        from io import BytesIO
        export_output = BytesIO()
        
        with pd.ExcelWriter(export_output, engine='openpyxl') as writer:
            if 'combined' in export_results and export_results['combined'] is not None:
                export_results['combined'].to_excel(writer, sheet_name='综合评分', index=False)
            if 'integrity' in export_results and export_results['integrity'] is not None:
                export_results['integrity'].to_excel(writer, sheet_name='完整性分析', index=False)
            if 'substantive' in export_results and export_results['substantive'] is not None:
                export_results['substantive'].to_excel(writer, sheet_name='实质性分析', index=False)
            if 'comparability' in export_results and export_results['comparability'] is not None:
                export_results['comparability'].to_excel(writer, sheet_name='可比性分析', index=False)
            if 'readability' in export_results and export_results['readability'] is not None:
                export_results['readability'].to_excel(writer, sheet_name='可读性分析', index=False)
            if 'reliability' in export_results and export_results['reliability'] is not None:
                export_results['reliability'].to_excel(writer, sheet_name='可靠性分析', index=False)
            if 'sentiment' in export_results and export_results['sentiment'] is not None:
                export_results['sentiment'].to_excel(writer, sheet_name='平衡性分析', index=False)
        
        export_output.seek(0)
        
        st.sidebar.download_button(
            label="⬇️ 下载 ESG评估结果 (Excel)",
            data=export_output.getvalue(),
            file_name=export_file_name,
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True
        )
    else:
        st.sidebar.info("暂无分析结果可导出")
except Exception as e:
    st.sidebar.warning(f"导出准备失败: {e}")

# 历史记录区域
st.sidebar.markdown("### 📁 历史分析记录")
history_files = get_history_list()

if history_files:
    # 创建历史记录选项
    history_options = ["当前分析"] + [f.stem for f in history_files]
    selected_history = st.sidebar.selectbox(
        "选择查看的分析记录",
        options=history_options,
        index=0,
        key="history_selector"
    )
    
    # 如果选择了历史记录，显示删除按钮
    if selected_history != "当前分析":
        col1, col2 = st.sidebar.columns(2)
        with col1:
            load_history_btn = st.button("🔍 查看", key="load_history")
        with col2:
            delete_history_btn = st.button("🗑️ 删除", key="delete_history")
    else:
        load_history_btn = False
        delete_history_btn = False
else:
    selected_history = "当前分析"
    st.sidebar.info("暂无历史记录")
    load_history_btn = False
    delete_history_btn = False

# 保存当前结果功能
st.sidebar.markdown("---")
st.sidebar.markdown("#### 💾 保存当前结果")

# 初始化保存状态
if 'show_save_input' not in st.session_state:
    st.session_state.show_save_input = False
if 'save_success_msg' not in st.session_state:
    st.session_state.save_success_msg = None

# 显示保存成功消息
if st.session_state.save_success_msg:
    st.sidebar.success(st.session_state.save_success_msg)
    st.session_state.save_success_msg = None

# 点击按钮显示输入框
if st.sidebar.button("💾 保存当前结果", key="show_save_dialog"):
    st.session_state.show_save_input = True

# 显示输入框和确认按钮
if st.session_state.show_save_input:
    save_name = st.sidebar.text_input(
        "请输入保存名称",
        placeholder="例如：2024年度ESG分析",
        key="save_name_input"
    )
    
    col_save, col_cancel = st.sidebar.columns(2)
    with col_save:
        if st.button("✅ 确认保存", key="confirm_save"):
            if save_name.strip():
                try:
                    # 加载当前结果并保存
                    current_results = load_analysis_results()
                    if current_results:
                        history_filename = save_analysis_history(current_results, save_name.strip())
                        st.session_state.save_success_msg = f"✅ 已保存为: {save_name.strip()}"
                        st.session_state.show_save_input = False
                        st.rerun()
                    else:
                        st.sidebar.error("没有可保存的分析结果")
                except Exception as e:
                    st.sidebar.error(f"保存失败: {e}")
            else:
                st.sidebar.warning("请输入保存名称")
    
    with col_cancel:
        if st.button("❌ 取消", key="cancel_save"):
            st.session_state.show_save_input = False
            st.rerun()

# 使用说明
# 使用说明和系统功能折叠面板
with st.sidebar.expander("📖 使用说明与系统功能", expanded=False):
    st.markdown("### 📖 使用说明")
    st.markdown("1. **上传文件**：上传的文件会自动保存到对应文件夹")
    st.markdown("   - PDF文件 → 汇总1/ 文件夹（用于可读性分析）")
    st.markdown("   - TXT文件 → 汇总/ 文件夹（用于其他分析）")
    st.markdown("2. 可选：在左侧边栏设置分析权重")
    st.markdown("3. 点击'开始分析'按钮进行分析")
    st.markdown("4. 查看分析结果图表和综合评分")
    
    st.markdown("\n---\n")
    
    st.markdown("### 🎯 系统功能")
    st.markdown("- **完整性分析**：评估ESG报告内容的全面性")
    st.markdown("- **实质性分析**：评估ESG报告内容的重要性")
    st.markdown("- **可比性分析**：评估不同年份报告的一致性")
    st.markdown("- **可读性分析**：评估ESG报告的易读性")
    st.markdown("- **可靠性分析**：评估ESG报告的可信性")
    st.markdown("- **平衡性分析**：评估ESG报告的情感平衡")

# 主页面标题
st.title("🏗️ 建筑业ESG报告披露质量评估系统")

# 显示处理进度的占位符
progress_bar = st.progress(0)
status_text = st.empty()

# 图表显示区域
charts_section = st.expander("📊 分析结果图表", expanded=True)

# 综合评分表
scores_section = st.expander("📈 综合评分结果", expanded=True)

# 保存上传的文件到临时目录
def save_uploaded_files(files, file_type):
    saved_paths = []
    if files:
        for file in files:
            # 创建文件保存路径
            if file_type == "pdf":
                save_path = os.path.join(TEMP_DIR, "pdf", file.name)
            else:
                save_path = os.path.join(TEMP_DIR, "txt", file.name)
            
            # 确保目录存在
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            
            # 保存文件
            with open(save_path, "wb") as f:
                f.write(file.getbuffer())
            
            saved_paths.append(save_path)
    return saved_paths

# 分析PDF文件（可读性分析）
def analyze_pdf_files(pdf_paths, progress_callback=None):
    if not pdf_paths:
        return None
    
    try:
        # 使用汇总1文件夹作为PDF来源
        pdf_source_dir = "汇总1"
        temp_output_dir = os.path.join(TEMP_DIR, "output")
        
        print(f"\n=== 开始可读性分析 ===")
        print(f"PDF源目录: {pdf_source_dir}")
        print(f"输出目录: {OUTPUT_DIR}")
        
        # 调用可读性分析并传入进度回调
        readability_results = readability_process_folder(
            pdf_source_dir,
            str(OUTPUT_DIR),
            temp_output_dir,
            progress_callback
        )
        print(f"可读性分析完成，返回结果数: {len(readability_results) if readability_results else 0}")
        return readability_results
    except Exception as e:
        st.error(f"可读性分析失败: {e}")
        print(f"可读性分析异常: {e}")
        import traceback
        traceback.print_exc()
        return None

# 分析TXT文件（其他分析）
def analyze_txt_files(txt_paths, progress_callback=None):
    if not txt_paths:
        return None
    
    try:
        # 使用汇总文件夹作为TXT来源
        txt_source_dir = "汇总"
        
        # 调用综合分析并传入进度回调
        analyzer.analyze_all_files(
            txt_source_dir,
            summary_weights=summary_weights,
            substantive_weights=substantive_weights,
            progress_callback=progress_callback
        )
        return True
    except Exception as e:
        st.error(f"TXT文件分析失败: {e}")
        return False

# 分析可靠性
def analyze_reliability(txt_paths, progress_callback=None):
    if not txt_paths:
        return None
    
    try:
        # 使用汇总文件夹作为TXT来源
        txt_source_dir = "汇总"
        
        reliability_results = reliability_process_files(txt_source_dir, progress_callback)
        # 保存可靠性分析结果
        reliability_output_path = os.path.join(OUTPUT_DIR, "reliability.csv")
        print(f"  保存可靠性结果到: {reliability_output_path}")
        print(f"  结果数据行数: {len(reliability_results) if reliability_results is not None else 0}")
        reliability_results.to_csv(reliability_output_path, index=False, encoding='utf-8-sig')
        print(f"  可靠性分析结果已保存")
        return reliability_results
    except Exception as e:
        st.error(f"可靠性分析失败: {e}")
        print(f"可靠性分析异常: {e}")
        return None

# 绘制综合评分柱状图
def plot_combined_scores(combined_df):
    if combined_df is None or combined_df.empty:
        return
    
    # 按文件名排序
    combined_df = combined_df.sort_values('文件名')
    
    # 获取文件名和各维度评分
    file_names = combined_df['文件名'].tolist()
    balance_scores = [round(score, 2) for score in combined_df['情感评分'].tolist()]  # 情感评分改为平衡性评分
    summary_scores = [round(score, 2) for score in combined_df['完整性评分'].tolist()]
    substantive_scores = [round(score, 2) for score in combined_df['实质性评分'].tolist()]
    comparability_scores = [round(score, 2) for score in combined_df['可比性评分'].tolist()]  # 可比性改为可比性评分
    readability_scores = [round(score, 2) for score in combined_df.get('可读性评分', [0]*len(file_names)).tolist()]
    reliability_scores = [round(score, 2) for score in combined_df.get('可靠性评分', [0]*len(file_names)).tolist()]
    
    # 使用已计算好的综合评分
    comprehensive_scores = [round(score, 2) for score in combined_df['综合评分'].tolist()]
    
    # 准备ECharts折线图数据
    line_option = {
        'title': {
            'text': 'ESG报告综合评分对比',
            'left': 'center',
            'textStyle': {
                'fontSize': 16
            }
        },
        'tooltip': {
            'trigger': 'axis',
            'axisPointer': {
                'type': 'cross',
                'animation': False
            },
            'formatter': '{b}<br/>{a}: {c:.2f}'
        },
        'legend': {
            'data': ['平衡性评分', '完整性评分', '实质性评分', '可比性评分', '可读性评分', '可靠性评分', '综合评分'],
            'bottom': '0'
        },
        'grid': {
            'left': '3%',
            'right': '4%',
            'bottom': '15%',
            'containLabel': True
        },
        'xAxis': {
            'type': 'category',
            'data': file_names,
            'axisLabel': {
                'rotate': 45,
                'interval': 0,
                'fontSize': 12
            }
        },
        'yAxis': {
            'type': 'value',
            'name': '评分',
            'min': 0,
            'max': 2,
            'interval': 0.2,
            'axisLabel': {
                'formatter': '{value}'
            }
        },
        'series': [
            {
                'name': '平衡性评分',
                'type': 'line',
                'data': balance_scores,
                'itemStyle': {
                    'color': '#5470c6'
                },
                'lineStyle': {
                    'width': 2
                },
                'symbol': 'circle',
                'symbolSize': 6,
                'label': {
                    'show': True,
                    'position': 'top',
                    'fontSize': 10,
                    'formatter': '{@[1]:.2f}'
                }
            },
            {
                'name': '完整性评分',
                'type': 'line',
                'data': summary_scores,
                'itemStyle': {
                    'color': '#91cc75'
                },
                'lineStyle': {
                    'width': 2
                },
                'symbol': 'circle',
                'symbolSize': 6,
                'label': {
                    'show': True,
                    'position': 'top',
                    'fontSize': 10,
                    'formatter': '{@[1]:.2f}'
                }
            },
            {
                'name': '实质性评分',
                'type': 'line',
                'data': substantive_scores,
                'itemStyle': {
                    'color': '#fac858'
                },
                'lineStyle': {
                    'width': 2
                },
                'symbol': 'circle',
                'symbolSize': 6,
                'label': {
                    'show': True,
                    'position': 'top',
                    'fontSize': 10,
                    'formatter': '{@[1]:.2f}'
                }
            },
            {
                'name': '可比性评分',
                'type': 'line',
                'data': comparability_scores,
                'itemStyle': {
                    'color': '#ee6666'
                },
                'lineStyle': {
                    'width': 2
                },
                'symbol': 'circle',
                'symbolSize': 6,
                'label': {
                    'show': True,
                    'position': 'top',
                    'fontSize': 10,
                    'formatter': '{@[1]:.2f}'
                }
            },
            {
                'name': '可读性评分',
                'type': 'line',
                'data': readability_scores,
                'itemStyle': {
                    'color': '#73c0de'
                },
                'lineStyle': {
                    'width': 2
                },
                'symbol': 'circle',
                'symbolSize': 6,
                'label': {
                    'show': True,
                    'position': 'top',
                    'fontSize': 10,
                    'formatter': '{@[1]:.2f}'
                }
            },
            {
                'name': '可靠性评分',
                'type': 'line',
                'data': reliability_scores,
                'itemStyle': {
                    'color': '#3ba272'
                },
                'lineStyle': {
                    'width': 2
                },
                'symbol': 'circle',
                'symbolSize': 6,
                'label': {
                    'show': True,
                    'position': 'top',
                    'fontSize': 10,
                    'formatter': '{@[1]:.2f}'
                }
            },
            {
                'name': '综合评分',
                'type': 'line',
                'data': comprehensive_scores,
                'itemStyle': {
                    'color': '#9a60b4'
                },
                'lineStyle': {
                    'width': 3,
                    'type': 'dashed'
                },
                'symbol': 'diamond',
                'symbolSize': 8,
                'label': {
                    'show': True,
                    'position': 'top',
                    'fontSize': 10,
                    'fontWeight': 'bold',
                    'formatter': '{@[1]:.2f}'
                }
            }
        ]
    }
    
    # 在Streamlit中显示ECharts图表
    st_echarts(options=line_option, height='600px', width='100%')

# 绘制各维度评分雷达图
def plot_radar_chart(combined_df):
    if combined_df is None or combined_df.empty:
        return
    
    # 选择要展示的维度（显示名称）
    display_dimensions = ['平衡性评分', '完整性评分', '实质性评分', '可比性评分', '可读性评分', '可靠性评分']
    # 对应的实际列名
    actual_columns = ['情感评分', '完整性评分', '实质性评分', '可比性评分', '可读性评分', '可靠性评分']
    
    # 添加交互功能：让用户选择要显示的文件
    all_files = combined_df['文件名'].tolist()
    selected_files = st.multiselect(
        '选择要显示的报告文件',
        options=all_files,
        default=all_files,
        key='radar_file_selector'
    )
    
    # 过滤数据
    filtered_df = combined_df[combined_df['文件名'].isin(selected_files)]
    
    if filtered_df.empty:
        st.write('未选择任何文件')
        return
    
    # 计算各维度的最大值，确保雷达图范围足够大
    max_values = []
    for col in actual_columns:
        col_max = filtered_df[col].max()
        # 向上取整到最接近的0.5或1，确保有足够的空间
        if col_max <= 1:
            max_values.append(1)
        elif col_max <= 1.5:
            max_values.append(1.5)
        elif col_max <= 2:
            max_values.append(2)
        else:
            # 如果有更大的值，向上取整到整数
            max_values.append(int(col_max) + 1)
    
    # 准备Echarts雷达图的数据
    radar_option = {
        'title': {
            'text': 'ESG报告各维度评分雷达图',
            'left': 'center',
            'textStyle': {
                'fontSize': 18
            }
        },
        'tooltip': {
            'trigger': 'item',
            'formatter': '{b}: {c:.2f}'
        },
        'legend': {
            'orient': 'vertical',
            'right': 10,
            'top': 'center',
            'type': 'scroll',
            'textStyle': {
                'fontSize': 10
            }
        },
        'radar': {
            'indicator': [{'name': dim, 'max': max_val, 'nameTextStyle': {'color': '#000000'}} for dim, max_val in zip(display_dimensions, max_values)],
            'radius': '65%',  # 减小半径，确保图形在范围内
            'center': ['50%', '50%'],  # 确保雷达图居中
            'splitNumber': 5,
            'axisLine': {
                'lineStyle': {
                    'width': 1
                }
            },
            'splitLine': {
                'lineStyle': {
                    'width': 1,
                    'type': 'dashed'
                }
            },
            'splitArea': {
                'show': True,
                'areaStyle': {
                    'color': ['rgba(250, 250, 250, 0.3)', 'rgba(200, 200, 200, 0.3)']
                }
            }
        },
        'series': [
            {
                'name': 'ESG评分',
                'type': 'radar',
                'data': [],
                'symbol': 'circle',
                'symbolSize': 6,
                'lineStyle': {
                    'width': 2
                },
                'areaStyle': {
                    'opacity': 0.1
                }
            }
        ]
    }
    
    # 为每个选择的文件添加数据
    colors = ['#5470c6', '#91cc75', '#fac858', '#ee6666', '#73c0de', '#3ba272', '#fc8452', '#9a60b4', '#ea7ccc', '#666666']
    for idx, (i, row) in enumerate(filtered_df.iterrows()):
        # 直接将数值格式化为2位小数
        values = [round(val, 2) for val in row[actual_columns].tolist()]
        radar_option['series'][0]['data'].append({
            'value': values,
            'name': row['文件名'],
            'symbol': 'circle',
            'symbolSize': 8,
            'lineStyle': {
                'width': 2
            },
            'areaStyle': {
                'opacity': 0.1
            },
            'itemStyle': {
                'color': colors[idx % len(colors)]
            },
            'label': {
                'show': True,
                'fontSize': 10
            }
        })
    
    # 在Streamlit中显示Echarts雷达图
    st_echarts(options=radar_option, height='600px', width='100%')

# 绘制可比性趋势图
def plot_comparability_trend(comparability_df):
    if comparability_df is None or comparability_df.empty:
        return
    
    # 获取年份对和可比性数据，保留2位小数
    year_pairs = comparability_df['年份对'].tolist()
    comparability_values = [round(float(val), 2) for val in comparability_df['可比性'].tolist()]
    similarity_values = [round(float(val), 2) for val in comparability_df['相似度'].tolist()]
    
    # 可比性评分折线图
    line_option = {
        'title': {
            'text': '可比性评分趋势',
            'left': 'center',
            'textStyle': {
                'fontSize': 14
            }
        },
        'tooltip': {
            'trigger': 'axis',
            'axisPointer': {
                'type': 'cross'
            }
        },
        'xAxis': {
            'type': 'category',
            'data': year_pairs,
            'axisLabel': {
                'rotate': 45,
                'interval': 0,
                'fontSize': 11
            }
        },
        'yAxis': {
            'type': 'value',
            'name': '评分',
            'min': 0,
            'max': 1,
            'interval': 0.2
        },
        'series': [
            {
                'name': '可比性评分',
                'type': 'line',
                'data': comparability_values,
                'smooth': True,
                'itemStyle': {
                    'color': '#27ae60'
                },
                'lineStyle': {
                    'width': 2
                },
                'areaStyle': {
                    'opacity': 0.1
                },
                'label': {
                    'show': True,
                    'position': 'top',
                    'formatter': '{c}',
                    'fontSize': 10
                }
            }
        ]
    }
    
    # 相似度折线图
    similarity_option = {
        'title': {
            'text': '相似度趋势',
            'left': 'center',
            'textStyle': {
                'fontSize': 14
            }
        },
        'tooltip': {
            'trigger': 'axis',
            'axisPointer': {
                'type': 'cross'
            }
        },
        'xAxis': {
            'type': 'category',
            'data': year_pairs,
            'axisLabel': {
                'rotate': 45,
                'interval': 0,
                'fontSize': 11
            }
        },
        'yAxis': {
            'type': 'value',
            'name': '相似度',
            'min': 0,
            'max': 1,
            'interval': 0.2
        },
        'series': [
            {
                'name': '相似度',
                'type': 'line',
                'data': similarity_values,
                'smooth': True,
                'itemStyle': {
                    'color': '#3498db'
                },
                'lineStyle': {
                    'width': 2
                },
                'areaStyle': {
                    'opacity': 0.1
                },
                'label': {
                    'show': True,
                    'position': 'top',
                    'formatter': '{c}',
                    'fontSize': 10
                }
            }
        ]
    }
    
    # 在Streamlit中并排显示两个ECharts图表
    col1, col2 = st.columns(2)
    with col1:
        st_echarts(options=line_option, height='400px', width='100%')
    with col2:
        st_echarts(options=similarity_option, height='400px', width='100%')

# 绘制可读性分析结果图
def plot_readability_results(readability_df):
    if readability_df is None or readability_df.empty:
        return
    
    # 按文件名排序
    readability_df = readability_df.sort_values('文件名')
    
    # 获取文件名和数据，保留2位小数
    file_names = readability_df['文件名'].tolist()
    r_read_scores = [round(float(x), 2) for x in readability_df['R_read'].tolist()]
    c_scores = [round(float(x), 2) for x in readability_df['C'].tolist()]
    v_scores = [round(float(x), 2) for x in readability_df['V'].tolist()]
    t_scores = [round(float(x), 2) for x in readability_df['T'].tolist()]
    
    # 综合可读性评分折线图
    line_option = {
        'title': {
            'text': '可读性评分趋势',
            'left': 'center',
            'textStyle': {
                'fontSize': 14
            }
        },
        'tooltip': {
            'trigger': 'axis',
            'axisPointer': {
                'type': 'cross'
            }
        },
        'xAxis': {
            'type': 'category',
            'data': file_names,
            'axisLabel': {
                'rotate': 45,
                'interval': 0,
                'fontSize': 11
            }
        },
        'yAxis': {
            'type': 'value',
            'name': '评分',
            'min': 0,
            'max': 1,
            'interval': 0.2
        },
        'series': [
            {
                'name': '可读性评分',
                'type': 'line',
                'data': r_read_scores,
                'smooth': True,
                'itemStyle': {
                    'color': '#9b59b6'
                },
                'lineStyle': {
                    'width': 2
                },
                'areaStyle': {
                    'opacity': 0.1
                },
                'label': {
                    'show': True,
                    'position': 'top',
                    'formatter': '{c}',
                    'fontSize': 10
                }
            }
        ]
    }
    
    # 获取图片数量和表格数量（如果存在）
    img_counts = []
    tbl_counts = []
    if '图片数量' in readability_df.columns:
        img_counts = [int(x) for x in readability_df['图片数量'].tolist()]
    if '表格数量' in readability_df.columns:
        tbl_counts = [int(x) for x in readability_df['表格数量'].tolist()]
    
    # 可读性各维度得分折线图（包含图片和表格数量，双 Y 轴）
    legend_data = ['C (目录及排版)', 'V (图表使用)', 'T (术语解释)']
    series_data = [
        {
            'name': 'C (目录及排版)',
            'type': 'line',
            'data': c_scores,
            'smooth': True,
            'yAxisIndex': 0,
            'itemStyle': {
                'color': '#5470c6'
            },
            'lineStyle': {
                'width': 2
            }
        },
        {
            'name': 'V (图表使用)',
            'type': 'line',
            'data': v_scores,
            'smooth': True,
            'yAxisIndex': 0,
            'itemStyle': {
                'color': '#91cc75'
            },
            'lineStyle': {
                'width': 2
            }
        },
        {
            'name': 'T (术语解释)',
            'type': 'line',
            'data': t_scores,
            'smooth': True,
            'yAxisIndex': 0,
            'itemStyle': {
                'color': '#fac858'
            },
            'lineStyle': {
                'width': 2
            }
        }
    ]
    
    # 如果有图片和表格数量，添加到图表中
    if img_counts and tbl_counts:
        legend_data.extend(['图片数量', '表格数量'])
        series_data.extend([
            {
                'name': '图片数量',
                'type': 'line',
                'data': img_counts,
                'smooth': True,
                'yAxisIndex': 1,
                'itemStyle': {
                    'color': '#3498db'
                },
                'lineStyle': {
                    'width': 2,
                    'type': 'dashed'
                },
                'symbol': 'diamond',
                'symbolSize': 8
            },
            {
                'name': '表格数量',
                'type': 'line',
                'data': tbl_counts,
                'smooth': True,
                'yAxisIndex': 1,
                'itemStyle': {
                    'color': '#e74c3c'
                },
                'lineStyle': {
                    'width': 2,
                    'type': 'dashed'
                },
                'symbol': 'triangle',
                'symbolSize': 8
            }
        ])
    
    # 配置Y轴
    y_axis_config = [
        {
            'type': 'value',
            'name': '得分',
            'min': 0,
            'max': 1,
            'interval': 0.2,
            'position': 'left'
        }
    ]
    
    # 如果有图片表格数据，添加第二Y轴
    if img_counts and tbl_counts:
        max_count = max(max(img_counts), max(tbl_counts)) if img_counts and tbl_counts else 100
        y_axis_config.append({
            'type': 'value',
            'name': '数量',
            'min': 0,
            'max': int(max_count * 1.2),
            'position': 'right',
            'axisLine': {
                'lineStyle': {
                    'color': '#3498db'
                }
            }
        })
    
    dims_line_option = {
        'title': {
            'text': '各维度得分趋势',
            'left': 'center',
            'textStyle': {
                'fontSize': 14
            }
        },
        'tooltip': {
            'trigger': 'axis',
            'axisPointer': {
                'type': 'cross'
            }
        },
        'legend': {
            'data': legend_data,
            'top': 30
        },
        'grid': {
            'top': 80,
            'right': 60
        },
        'xAxis': {
            'type': 'category',
            'data': file_names,
            'axisLabel': {
                'rotate': 45,
                'interval': 0,
                'fontSize': 11
            }
        },
        'yAxis': y_axis_config,
        'series': series_data
    }
    
    # 在Streamlit中并排显示两个ECharts图表
    col1, col2 = st.columns(2)
    with col1:
        st_echarts(options=line_option, height='400px', width='100%')
    with col2:
        st_echarts(options=dims_line_option, height='400px', width='100%')

# 绘制可靠性分析结果图
def plot_reliability_results(reliability_df):
    if reliability_df is None or reliability_df.empty:
        return
    
    # 按文件名排序
    reliability_df = reliability_df.sort_values('文件名')
    
    # 获取文件名和数据，保留2位小数
    file_names = reliability_df['文件名'].tolist()
    r_scores = [round(float(x), 2) for x in reliability_df['综合可靠性(R)'].tolist()]
    e_scores = [round(float(x), 2) for x in reliability_df['外部鉴证(E)'].tolist()]
    s_scores = [round(float(x), 2) for x in reliability_df['利益相关方(S)'].tolist()]
    a_scores = [round(float(x), 2) for x in reliability_df['真实性承诺(A)'].tolist()]
    
    # 综合可靠性评分折线图
    line_option = {
        'title': {
            'text': '可靠性评分趋势',
            'left': 'center',
            'textStyle': {
                'fontSize': 14
            }
        },
        'tooltip': {
            'trigger': 'axis',
            'axisPointer': {
                'type': 'cross'
            }
        },
        'xAxis': {
            'type': 'category',
            'data': file_names,
            'axisLabel': {
                'rotate': 45,
                'interval': 0,
                'fontSize': 11
            }
        },
        'yAxis': {
            'type': 'value',
            'name': '评分',
            'min': 0,
            'max': 1,
            'interval': 0.2
        },
        'series': [
            {
                'name': '可靠性评分',
                'type': 'line',
                'data': r_scores,
                'smooth': True,
                'itemStyle': {
                    'color': '#e67e22'
                },
                'lineStyle': {
                    'width': 2
                },
                'areaStyle': {
                    'opacity': 0.1
                },
                'label': {
                    'show': True,
                    'position': 'top',
                    'formatter': '{c}',
                    'fontSize': 10
                }
            }
        ]
    }
    
    # 可靠性各维度得分折线图
    dims_line_option = {
        'title': {
            'text': '各维度得分趋势',
            'left': 'center',
            'textStyle': {
                'fontSize': 14
            }
        },
        'tooltip': {
            'trigger': 'axis',
            'axisPointer': {
                'type': 'cross'
            }
        },
        'legend': {
            'data': ['E (外部鉴证)', 'S (利益相关方)', 'A (真实性承诺)'],
            'top': 30
        },
        'grid': {
            'top': 80
        },
        'xAxis': {
            'type': 'category',
            'data': file_names,
            'axisLabel': {
                'rotate': 45,
                'interval': 0,
                'fontSize': 11
            }
        },
        'yAxis': {
            'type': 'value',
            'name': '得分',
            'min': 0,
            'max': 1,
            'interval': 0.2
        },
        'series': [
            {
                'name': 'E (外部鉴证)',
                'type': 'line',
                'data': e_scores,
                'smooth': True,
                'itemStyle': {
                    'color': '#ee6666'
                },
                'lineStyle': {
                    'width': 2
                }
            },
            {
                'name': 'S (利益相关方)',
                'type': 'line',
                'data': s_scores,
                'smooth': True,
                'itemStyle': {
                    'color': '#3ba272'
                },
                'lineStyle': {
                    'width': 2
                }
            },
            {
                'name': 'A (真实性承诺)',
                'type': 'line',
                'data': a_scores,
                'smooth': True,
                'itemStyle': {
                    'color': '#fac858'
                },
                'lineStyle': {
                    'width': 2
                }
            }
        ]
    }
    
    # 在Streamlit中并排显示两个ECharts图表
    col1, col2 = st.columns(2)
    with col1:
        st_echarts(options=line_option, height='400px', width='100%')
    with col2:
        st_echarts(options=dims_line_option, height='400px', width='100%')

# 绘制完整性分析结果图
def plot_integrity_results(integrity_df):
    if integrity_df is None or integrity_df.empty:
        return
    
    # 按文件名排序
    integrity_df = integrity_df.sort_values('文件名')
    
    # 获取文件名和完整性评分
    file_names = integrity_df['文件名'].tolist()
    integrity_scores = [round(float(x), 2) for x in integrity_df['完整性评分'].tolist()]
    
    # 完整性评分折线图
    line_option = {
        'title': {
            'text': '完整性评分趋势',
            'left': 'center',
            'textStyle': {
                'fontSize': 14
            }
        },
        'tooltip': {
            'trigger': 'axis',
            'axisPointer': {
                'type': 'cross'
            }
        },
        'xAxis': {
            'type': 'category',
            'data': file_names,
            'axisLabel': {
                'rotate': 45,
                'interval': 0,
                'fontSize': 11
            }
        },
        'yAxis': {
            'type': 'value',
            'name': '评分',
            'min': 0,
            'max': 2,
            'interval': 0.5
        },
        'series': [
            {
                'name': '完整性评分',
                'type': 'line',
                'data': integrity_scores,
                'smooth': True,
                'itemStyle': {
                    'color': '#91cc75'
                },
                'lineStyle': {
                    'width': 2
                },
                'areaStyle': {
                    'opacity': 0.1
                },
                'label': {
                    'show': True,
                    'position': 'top',
                    'formatter': '{c}',
                    'fontSize': 10
                }
            }
        ]
    }
    
    # 获取各维度列名（排除文件名和完整性评分）
    dimension_cols = [col for col in integrity_df.columns if col not in ['文件名', '完整性评分']]
    
    # 取前6个维度做折线图
    selected_dims = dimension_cols[:6] if len(dimension_cols) > 6 else dimension_cols
    
    # 准备各维度折线图数据
    series_data = []
    colors = ['#5470c6', '#91cc75', '#fac858', '#ee6666', '#73c0de', '#3ba272']
    for i, dim in enumerate(selected_dims):
        dim_scores = [round(float(x), 2) for x in integrity_df[dim].tolist()]
        series_data.append({
            'name': dim[:8] + '...' if len(dim) > 8 else dim,
            'type': 'line',
            'data': dim_scores,
            'smooth': True,
            'itemStyle': {
                'color': colors[i % len(colors)]
            },
            'lineStyle': {
                'width': 2
            }
        })
    
    dims_line_option = {
        'title': {
            'text': '各维度得分趋势',
            'left': 'center',
            'textStyle': {
                'fontSize': 14
            }
        },
        'tooltip': {
            'trigger': 'axis',
            'axisPointer': {
                'type': 'cross'
            }
        },
        'legend': {
            'data': [dim[:8] + '...' if len(dim) > 8 else dim for dim in selected_dims],
            'top': 30,
            'type': 'scroll'
        },
        'grid': {
            'top': 80
        },
        'xAxis': {
            'type': 'category',
            'data': file_names,
            'axisLabel': {
                'rotate': 45,
                'interval': 0,
                'fontSize': 11
            }
        },
        'yAxis': {
            'type': 'value',
            'name': '得分',
            'min': 0,
            'max': 2,
            'interval': 0.5
        },
        'series': series_data
    }
    
    # 在Streamlit中并排显示两个ECharts图表
    col1, col2 = st.columns(2)
    with col1:
        st_echarts(options=line_option, height='400px', width='100%')
    with col2:
        st_echarts(options=dims_line_option, height='400px', width='100%')

# 绘制实质性分析结果图
def plot_substantive_results(substantive_df):
    if substantive_df is None or substantive_df.empty:
        return
    
    # 按文件名排序
    substantive_df = substantive_df.sort_values('文件名')
    
    # 获取文件名和实质性评分
    file_names = substantive_df['文件名'].tolist()
    substantive_scores = [round(float(x), 2) for x in substantive_df['实质性评分'].tolist()]
    
    # 实质性评分折线图
    line_option = {
        'title': {
            'text': '实质性评分趋势',
            'left': 'center',
            'textStyle': {
                'fontSize': 14
            }
        },
        'tooltip': {
            'trigger': 'axis',
            'axisPointer': {
                'type': 'cross'
            }
        },
        'xAxis': {
            'type': 'category',
            'data': file_names,
            'axisLabel': {
                'rotate': 45,
                'interval': 0,
                'fontSize': 11
            }
        },
        'yAxis': {
            'type': 'value',
            'name': '评分',
            'min': 0,
            'max': 2,
            'interval': 0.5
        },
        'series': [
            {
                'name': '实质性评分',
                'type': 'line',
                'data': substantive_scores,
                'smooth': True,
                'itemStyle': {
                    'color': '#ee6666'
                },
                'lineStyle': {
                    'width': 2
                },
                'areaStyle': {
                    'opacity': 0.1
                },
                'label': {
                    'show': True,
                    'position': 'top',
                    'formatter': '{c}',
                    'fontSize': 10
                }
            }
        ]
    }
    
    # 获取各维度列名（排除文件名和实质性评分）
    dimension_cols = [col for col in substantive_df.columns if col not in ['文件名', '实质性评分']]
    
    # 取前6个维度做折线图
    selected_dims = dimension_cols[:6] if len(dimension_cols) > 6 else dimension_cols
    
    # 准备各维度折线图数据
    series_data = []
    colors = ['#5470c6', '#91cc75', '#fac858', '#ee6666', '#73c0de', '#3ba272']
    for i, dim in enumerate(selected_dims):
        dim_scores = [round(float(x), 2) for x in substantive_df[dim].tolist()]
        series_data.append({
            'name': dim[:8] + '...' if len(dim) > 8 else dim,
            'type': 'line',
            'data': dim_scores,
            'smooth': True,
            'itemStyle': {
                'color': colors[i % len(colors)]
            },
            'lineStyle': {
                'width': 2
            }
        })
    
    dims_line_option = {
        'title': {
            'text': '各维度得分趋势',
            'left': 'center',
            'textStyle': {
                'fontSize': 14
            }
        },
        'tooltip': {
            'trigger': 'axis',
            'axisPointer': {
                'type': 'cross'
            }
        },
        'legend': {
            'data': [dim[:8] + '...' if len(dim) > 8 else dim for dim in selected_dims],
            'top': 30,
            'type': 'scroll'
        },
        'grid': {
            'top': 80
        },
        'xAxis': {
            'type': 'category',
            'data': file_names,
            'axisLabel': {
                'rotate': 45,
                'interval': 0,
                'fontSize': 11
            }
        },
        'yAxis': {
            'type': 'value',
            'name': '得分',
            'min': 0,
            'max': 2,
            'interval': 0.5
        },
        'series': series_data
    }
    
    # 在Streamlit中并排显示两个ECharts图表
    col1, col2 = st.columns(2)
    with col1:
        st_echarts(options=line_option, height='400px', width='100%')
    with col2:
        st_echarts(options=dims_line_option, height='400px', width='100%')

# 绘制平衡性分析结果图（折线图）
def plot_sentiment_balance_results(sentiment_df):
    if sentiment_df is None or sentiment_df.empty:
        return
    
    # 兼容新旧两种格式的列名
    # 新格式：文件名, 积极比例, 消极比例, 中立比例, 平衡性评分
    # 旧格式：file_name, positive_ratio, negative_ratio, neutral_ratio, sentiment_score
    
    # 判断是哪种格式
    if '文件名' in sentiment_df.columns:
        file_col = '文件名'
        balance_col = '平衡性评分'
        positive_col = '积极比例'
        negative_col = '消极比例'
        neutral_col = '中立比例'
    else:
        file_col = 'file_name'
        balance_col = 'sentiment_score'
        positive_col = 'positive_ratio'
        negative_col = 'negative_ratio'
        neutral_col = 'neutral_ratio'
    
    # 按文件名排序
    sentiment_df = sentiment_df.sort_values(file_col)
    
    # 获取文件名和数据
    file_names = sentiment_df[file_col].tolist()
    balance_scores = [round(float(x), 2) for x in sentiment_df[balance_col].tolist()]
    positive_ratios = [round(float(x), 2) for x in sentiment_df[positive_col].tolist()]
    negative_ratios = [round(float(x), 2) for x in sentiment_df[negative_col].tolist()]
    neutral_ratios = [round(float(x), 2) for x in sentiment_df[neutral_col].tolist()]
    
    # 平衡性评分折线图
    line_option = {
        'title': {
            'text': '平衡性评分',
            'left': 'center',
            'textStyle': {
                'fontSize': 14
            }
        },
        'tooltip': {
            'trigger': 'axis',
            'axisPointer': {
                'type': 'cross'
            }
        },
        'legend': {
            'data': ['平衡性评分'],
            'top': 30
        },
        'xAxis': {
            'type': 'category',
            'data': file_names,
            'axisLabel': {
                'rotate': 45,
                'interval': 0,
                'fontSize': 11
            }
        },
        'yAxis': {
            'type': 'value',
            'name': '评分',
            'min': 0,
            'max': 1,
            'interval': 0.2
        },
        'series': [
            {
                'name': '平衡性评分',
                'type': 'line',
                'data': balance_scores,
                'smooth': True,
                'itemStyle': {
                    'color': '#5470c6'
                },
                'lineStyle': {
                    'width': 2
                },
                'areaStyle': {
                    'opacity': 0.1
                }
            }
        ]
    }
    
    # 情感占比折线图
    ratio_line_option = {
        'title': {
            'text': '情感占比',
            'left': 'center',
            'textStyle': {
                'fontSize': 14
            }
        },
        'tooltip': {
            'trigger': 'axis',
            'axisPointer': {
                'type': 'cross'
            }
        },
        'legend': {
            'data': ['积极比例', '消极比例', '中立比例'],
            'top': 30
        },
        'xAxis': {
            'type': 'category',
            'data': file_names,
            'axisLabel': {
                'rotate': 45,
                'interval': 0,
                'fontSize': 11
            }
        },
        'yAxis': {
            'type': 'value',
            'name': '比例',
            'min': 0,
            'max': 1,
            'interval': 0.2
        },
        'series': [
            {
                'name': '积极比例',
                'type': 'line',
                'data': positive_ratios,
                'smooth': True,
                'itemStyle': {
                    'color': '#3ba272'
                },
                'lineStyle': {
                    'width': 2
                }
            },
            {
                'name': '消极比例',
                'type': 'line',
                'data': negative_ratios,
                'smooth': True,
                'itemStyle': {
                    'color': '#ee6666'
                },
                'lineStyle': {
                    'width': 2
                }
            },
            {
                'name': '中立比例',
                'type': 'line',
                'data': neutral_ratios,
                'smooth': True,
                'itemStyle': {
                    'color': '#fac858'
                },
                'lineStyle': {
                    'width': 2
                }
            }
        ]
    }
    
    # 在Streamlit中并排显示两个ECharts图表
    col1, col2 = st.columns(2)
    with col1:
        st_echarts(options=line_option, height='400px', width='100%')
    with col2:
        st_echarts(options=ratio_line_option, height='400px', width='100%')

# 数据导出函数
def export_analysis_results():
    """导出所有分析结果到Excel文件，每个市场为一个平子"""
    try:
        # 加载所有结果
        results = load_analysis_results()
        
        if not results:
            st.error("没有或简少成成分析结果，无法导出")
            return
        
        # 使用BytesIO创建内存中的Excel文件
        from io import BytesIO
        output = BytesIO()
        
        # 创建 ExcelWriter
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            # 综合评分结果
            if 'combined' in results:
                results['combined'].to_excel(writer, sheet_name='综合评分', index=False)
            
            # 完整性分析结果
            if 'integrity' in results:
                results['integrity'].to_excel(writer, sheet_name='完整性分析', index=False)
            
            # 实质性分析结果
            if 'substantive' in results:
                results['substantive'].to_excel(writer, sheet_name='实质性分析', index=False)
            
            # 可比性分析结果
            if 'comparability' in results:
                results['comparability'].to_excel(writer, sheet_name='可比性分析', index=False)
            
            # 可读性分析结果
            if 'readability' in results:
                results['readability'].to_excel(writer, sheet_name='可读性分析', index=False)
            
            # 可靠性分析结果
            if 'reliability' in results:
                results['reliability'].to_excel(writer, sheet_name='可靠性分析', index=False)
        
        output.seek(0)
        
        # 批次下载或一个超级文件
        st.success("数据打包完成，准备下载")
        st.download_button(
            label="⬇️ 下载 ESG评估结果 (Excel)",
            data=output.getvalue(),
            file_name=f"ESG评估结果_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
        
        return True
    except Exception as e:
        st.error(f"数据导出失败: {e}")
        return False

# 主分析流程

# 处理历史记录删除
if history_files and selected_history != "当前分析":
    if 'delete_history' in dir() and delete_history_btn:
        # 找到对应的历史记录文件
        for hf in history_files:
            if hf.stem == selected_history:
                if delete_history_record(hf):
                    st.sidebar.success(f"✅ 已删除记录: {selected_history}")
                    st.rerun()
                break

# 处理历史记录加载
if load_history_btn and selected_history != "当前分析":
    # 找到对应的历史记录文件
    history_data = None
    for hf in history_files:
        if hf.stem == selected_history:
            history_data = load_analysis_history(hf)
            break
    
    if history_data:
        st.info(f"📅 正在查看历史记录: {history_data['name']}")
        results = history_data['results']
        
        # 更新导出数据为当前查看的历史记录
        st.session_state.export_data = results
        st.session_state.export_file_name = f"{history_data['name']}_导出结果.xlsx"
        
        # 显示历史记录的分析结果图表
        with charts_section:
            st.subheader("综合评分分析")
            if 'combined' in results and results['combined'] is not None and len(results['combined']) > 0:
                plot_combined_scores(results['combined'])
                plot_radar_chart(results['combined'])
            else:
                st.write("❌ 没有综合评分结果")
            
            st.subheader("完整性分析")
            if 'integrity' in results and results['integrity'] is not None and len(results['integrity']) > 0:
                plot_integrity_results(results['integrity'])
            else:
                st.write("❌ 没有完整性分析结果")

            st.subheader("实质性分析")
            if 'substantive' in results and results['substantive'] is not None and len(results['substantive']) > 0:
                plot_substantive_results(results['substantive'])
            else:
                st.write("❌ 没有实质性分析结果")

            st.subheader("可比性分析")
            if 'comparability' in results and results['comparability'] is not None and len(results['comparability']) > 0:
                plot_comparability_trend(results['comparability'])
            else:
                st.write("❌ 没有可比性分析结果")

            st.subheader("可读性分析")
            if 'readability' in results and results['readability'] is not None and len(results['readability']) > 0:
                plot_readability_results(results['readability'])
            else:
                st.write("❌ 没有可读性分析结果")

            st.subheader("可靠性分析")
            if 'reliability' in results and results['reliability'] is not None and len(results['reliability']) > 0:
                plot_reliability_results(results['reliability'])
            else:
                st.write("❌ 没有可靠性分析结果")

            st.subheader("平衡性分析")
            if 'sentiment' in results and results['sentiment'] is not None and len(results['sentiment']) > 0:
                plot_sentiment_balance_results(results['sentiment'])
            else:
                st.write("❌ 没有平衡性分析结果")
        
        # 显示综合评分结果
        with scores_section:
            st.subheader("综合评分结果")
            if 'combined' in results and results['combined'] is not None and len(results['combined']) > 0:
                st.dataframe(results['combined'])
            else:
                st.write("❌ 没有综合评分结果")

            st.subheader("完整性分析结果")
            if 'integrity' in results and results['integrity'] is not None and len(results['integrity']) > 0:
                st.dataframe(results['integrity'])
            else:
                st.write("❌ 没有完整性分析结果")

            st.subheader("实质性分析结果")
            if 'substantive' in results and results['substantive'] is not None and len(results['substantive']) > 0:
                st.dataframe(results['substantive'])
            else:
                st.write("❌ 没有实质性分析结果")

            st.subheader("可读性分析结果")
            if 'readability' in results and results['readability'] is not None and len(results['readability']) > 0:
                st.dataframe(results['readability'])
            else:
                st.write("❌ 没有可读性分析结果")

            st.subheader("可靠性分析结果")
            if 'reliability' in results and results['reliability'] is not None and len(results['reliability']) > 0:
                st.dataframe(results['reliability'])
            else:
                st.write("❌ 没有可靠性分析结果")

            st.subheader("可比性分析结果")
            if 'comparability' in results and results['comparability'] is not None and len(results['comparability']) > 0:
                st.dataframe(results['comparability'])
            else:
                st.write("❌ 没有可比性分析结果")

            st.subheader("平衡性分析结果")
            if 'sentiment' in results and results['sentiment'] is not None and len(results['sentiment']) > 0:
                st.dataframe(results['sentiment'])
            else:
                st.write("❌ 没有平衡性分析结果")

# 处理当前分析
elif analyze_button:
    # 处理分析按钮点击
    # 不需要检查文件上传，直接使用汇总和汇总1文件夹
    
    # 创建进度状态
    progress_state = {
        "current": 0,
        "total": 3,  # 三个主要分析任务
        "current_file": ""
    }
    
    # 创建文件信息显示
    file_info_text = st.empty()
    
    # 更新处理状态
    def update_progress(task_progress, current_file=""):
        # 计算总体进度（task_progress 已经是 0-1 的分数，直接转换为百分比）
        overall_progress = int(min(task_progress * 100, 100))  # 确保不超过100
        progress_bar.progress(overall_progress)
        
        # 更新当前处理的文件
        if current_file:
            progress_state["current_file"] = current_file
            file_info_text.text(f"正在分析文件: {current_file}")
    
    # 初始化昺示
    status_text.text("正在进行分析，请稍伪...")
    file_info_text.text("正在准备分析...")
    progress_bar.progress(0)
        
    # 根据选择的维度进行恶性需的分析
    analysis_steps = []
    if "完整性分析" in analysis_dimensions or "实质性分析" in analysis_dimensions or "可比性分析" in analysis_dimensions or "平衡性分析" in analysis_dimensions:
        analysis_steps.append(("txt", "综合分析儫完整性、实质性、可比性、情感）"))
    if "可靠性分析" in analysis_dimensions:
        analysis_steps.append(("reliability", "可靠性分析"))
    if "可读性分析" in analysis_dimensions:
        analysis_steps.append(("readability", "可读性分析"))
        
    # 如果没有选择任何维度，提示用户
    if not analysis_steps:
        st.warning("⚠️ 请至少选择一个分析维度")
    else:
        # 加载模型（延迟加载）
        if analyzer is not None:
            try:
                with st.spinner("正在加载分析模型..."):
                    analyzer.load_all_models()
                st.success("所有模型加载完成！")
            except Exception as e:
                st.error(f"模型加载失败: {e}")
                st.stop()
        else:
            st.error("分析器未初始化，无法进行分析")
            st.stop()
        
        # 预计主任务数
        progress_state["total"] = len(analysis_steps)
            
        # 顺序执行分析（改为顺序执行以便调试）
        step_count = 0
        total_steps = len(analysis_steps)
                
        if any(t[0] == "txt" for t in analysis_steps):
            status_text.text("正在执行综合分析（完整性、实质性、可比性、情感）...")
            print("\n=== 开始综合分析 ===")
            base_progress = step_count / total_steps
            integrated_results = analyze_txt_files(["汇总"], lambda p, f: update_progress(base_progress + p/total_steps, f))
            print("综合分析完成")
            step_count += 1
                    
        if any(t[0] == "reliability" for t in analysis_steps):
            status_text.text("正在执行可靠性分析...")
            print("\n=== 开始可靠性分析 ===")
            base_progress = step_count / total_steps
            reliability_results = analyze_reliability(["汇总"], lambda p, f: update_progress(base_progress + p/total_steps, f))
            print("可靠性分析完成")
            step_count += 1
                    
        if any(t[0] == "readability" for t in analysis_steps):
            status_text.text("正在执行可读性分析...")
            print("\n=== 开始可读性分析 ===")
            base_progress = step_count / total_steps
            readability_results = analyze_pdf_files(["汇总1"], lambda p, f: update_progress(base_progress + p/total_steps, f))
            print("可读性分析完成")
            step_count += 1
            
        # 完成分析
        status_text.text("分析完成！")
        file_info_text.text("所有文件处理完成！")
        progress_bar.progress(100)
        
        # 分析完成后，更新显示维度为当前选择的维度
        st.session_state.displayed_dimensions = analysis_dimensions.copy()
        
        # 丢弃之前的导出数据（恢复为当前分析结果）
        if 'export_data' in st.session_state:
            del st.session_state.export_data
        if 'export_file_name' in st.session_state:
            del st.session_state.export_file_name
            
        # 加载所有分析结果
        results = load_analysis_results()
            
        # 保存到历史记录
        try:
            history_filename = save_analysis_history(results)
            st.success(f"✅ 分析结果已保存到历史记录: {history_filename}")
        except Exception as e:
            st.warning(f"历史记录保存失败: {e}")
            
        # 显示分析结果图表
        with charts_section:
            st.subheader("综合评分分析")
            if 'combined' in results:
                plot_combined_scores(results['combined'])
                plot_radar_chart(results['combined'])
            else:
                st.write("❌ 没有综合评分结果")
                
            if "完整性分析" in analysis_dimensions:
                st.subheader("完整性分析")
                if 'integrity' in results and results['integrity'] is not None and len(results['integrity']) > 0:
                    plot_integrity_results(results['integrity'])
                else:
                    st.write("❌ 没有完整性分析结果")
                
            if "实质性分析" in analysis_dimensions:
                st.subheader("实质性分析")
                if 'substantive' in results and results['substantive'] is not None and len(results['substantive']) > 0:
                    plot_substantive_results(results['substantive'])
                else:
                    st.write("❌ 没有实质性分析结果")
                
            if "可比性分析" in analysis_dimensions:
                st.subheader("可比性分析")
                if 'comparability' in results:
                    plot_comparability_trend(results['comparability'])
                else:
                    st.write("❌ 没有可比性分析结果")
                
            if "可读性分析" in analysis_dimensions:
                st.subheader("可读性分析")
                if 'readability' in results:
                    plot_readability_results(results['readability'])
                else:
                    st.write("❌ 没有可读性分析结果")
                
            if "可靠性分析" in analysis_dimensions:
                st.subheader("可靠性分析")
                if 'reliability' in results:
                    plot_reliability_results(results['reliability'])
                else:
                    st.write("❌ 没有可靠性分析结果")
                
            if "平衡性分析" in analysis_dimensions:
                st.subheader("平衡性分析")
                if 'sentiment' in results and results['sentiment'] is not None and len(results['sentiment']) > 0:
                    plot_sentiment_balance_results(results['sentiment'])
                else:
                    st.write("❌ 没有平衡性分析结果")
            
        # 显示综合评分结果
        with scores_section:
            st.subheader("综合评分结果")
            if 'combined' in results:
                st.dataframe(results['combined'].style.format({"情感评分": "{:.2f}", "完整性评分": "{:.2f}", "实质性评分": "{:.2f}", "综合评分": "{:.2f}", "可比性评分": "{:.2f}", "可读性评分": "{:.2f}", "可靠性评分": "{:.2f}"}))
            else:
                st.write("❌ 没有综合评分结果")
            
            if "完整性分析" in analysis_dimensions:
                st.subheader("完整性分析结果")
                if 'integrity' in results:
                    st.dataframe(results['integrity'])
                else:
                    st.write("❌ 没有完整性分析结果")
            
            if "实质性分析" in analysis_dimensions:
                st.subheader("实质性分析结果")
                if 'substantive' in results:
                    st.dataframe(results['substantive'])
                else:
                    st.write("❌ 没有实质性分析结果")
            
            if "可读性分析" in analysis_dimensions:
                st.subheader("可读性分析结果")
                if 'readability' in results:
                    st.dataframe(results['readability'].style.format({"C": "{:.0f}", "V": "{:.0f}", "T": "{:.0f}", "图片数量": "{:.0f}", "表格数量": "{:.0f}", "R_read": "{:.2f}"}))
                else:
                    st.write("❌ 没有可读性分析结果")
            
            if "可靠性分析" in analysis_dimensions:
                st.subheader("可靠性分析结果")
                if 'reliability' in results:
                    st.dataframe(results['reliability'].style.format({"外部鉴证(E)": "{:.0f}", "利益相关方(S)": "{:.0f}", "真实性承诺(A)": "{:.0f}", "综合可靠性(R)": "{:.2f}"}))  
                else:
                    st.write("❌ 没有可靠性分析结果")
            
            if "可比性分析" in analysis_dimensions:
                st.subheader("可比性分析结果")
                if 'comparability' in results:
                    st.dataframe(results['comparability'].style.format({"相似度": "{:.2f}", "可比性": "{:.2f}"}))
                else:
                    st.write("❌ 没有可比性分析结果")
            
            if "平衡性分析" in analysis_dimensions:
                st.subheader("平衡性结果")
                if 'sentiment' in results and results['sentiment'] is not None and len(results['sentiment']) > 0:
                    st.dataframe(results['sentiment'])
                else:
                    st.write("❌ 没有平衡性分析结果")

# 初始化页面显示（非分析状态且非历史记录查看状态）
if not analyze_button and not load_history_btn:
    # 初始状态：直接显示汇总和汇悻1文件夹的分析结果
    
    # 使用已分析的维度列表（而非当前侧边栏选择）决定显示哪些图表
    displayed_dims = st.session_state.displayed_dimensions
    
    # 加载并显示现有分析结果
    results = load_analysis_results()
    
    with charts_section:
        st.subheader("综合评分分析")
        if 'combined' in results:
            plot_combined_scores(results['combined'])
            plot_radar_chart(results['combined'])
        else:
            st.write("❌ 没有综合评分结果")
        
        st.subheader("完整性分析")
        if "完整性分析" in displayed_dims and 'integrity' in results and results['integrity'] is not None and len(results['integrity']) > 0:
            plot_integrity_results(results['integrity'])
        elif "完整性分析" not in displayed_dims:
            pass
        else:
            st.write("❌ 没有完整性分析结果")

        st.subheader("实质性分析")
        if "实质性分析" in displayed_dims and 'substantive' in results and results['substantive'] is not None and len(results['substantive']) > 0:
            plot_substantive_results(results['substantive'])
        elif "实质性分析" not in displayed_dims:
            pass
        else:
            st.write("❌ 没有实质性分析结果")

        st.subheader("可比性分析")
        if "可比性分析" in displayed_dims and 'comparability' in results:
            plot_comparability_trend(results['comparability'])
        elif "可比性分析" not in displayed_dims:
            pass
        else:
            st.write("❌ 没有可比性分析结果")

        st.subheader("可读性分析")
        if "可读性分析" in displayed_dims and 'readability' in results:
            plot_readability_results(results['readability'])
        elif "可读性分析" not in displayed_dims:
            pass
        else:
            st.write("❌ 没有可读性分析结果")

        st.subheader("可靠性分析")
        if "可靠性分析" in displayed_dims and 'reliability' in results:
            plot_reliability_results(results['reliability'])
        elif "可靠性分析" not in displayed_dims:
            pass
        else:
            st.write("❌ 没有可靠性分析结果")
    
        st.subheader("平衡性分析")
        if "平衡性分析" in displayed_dims and 'sentiment' in results and results['sentiment'] is not None and len(results['sentiment']) > 0:
            plot_sentiment_balance_results(results['sentiment'])
        elif "平衡性分析" not in displayed_dims:
            pass
        else:
            st.write("❌ 没有平衡性分析结果")
    
    with scores_section:
        st.subheader("综合评分结果")
        if 'combined' in results:
            st.dataframe(results['combined'].style.format({"情感评分": "{:.2f}", "完整性评分": "{:.2f}", "实质性评分": "{:.2f}", "综合评分": "{:.2f}", "可比性评分": "{:.2f}", "可读性评分": "{:.2f}", "可靠性评分": "{:.2f}"}))
        else:
            st.write("❌ 没有综合评分结果")

        st.subheader("完整性分析结果")
        if "完整性分析" in displayed_dims and 'integrity' in results:
            st.dataframe(results['integrity'])
        elif "完整性分析" not in displayed_dims:
            pass
        else:
            st.write("❌ 没有完整性分析结果")
            
        st.subheader("实质性分析结果")
        if "实质性分析" in displayed_dims and 'substantive' in results:
            st.dataframe(results['substantive'])
        elif "实质性分析" not in displayed_dims:
            pass
        else:
            st.write("❌ 没有实质性分析结果")
            
        st.subheader("可读性分析结果")
        if "可读性分析" in displayed_dims and 'readability' in results:
            st.dataframe(results['readability'].style.format({"C": "{:.0f}", "V": "{:.0f}", "T": "{:.0f}", "图片数量": "{:.0f}", "表格数量": "{:.0f}", "R_read": "{:.2f}"}))
        elif "可读性分析" not in displayed_dims:
            pass
        else:
            st.write("❌ 没有可读性分析结果")
            
        st.subheader("可靠性分析结果")
        if "可靠性分析" in displayed_dims and 'reliability' in results:
            st.dataframe(results['reliability'].style.format({"外部鉴证(E)": "{:.0f}", "利益相关方(S)": "{:.0f}", "真实性承诺(A)": "{:.0f}", "综合可靠性(R)": "{:.2f}"}))  
        elif "可靠性分析" not in displayed_dims:
            pass
        else:
            st.write("❌ 没有可靠性分析结果")
            
        st.subheader("可比性分析结果")
        if "可比性分析" in displayed_dims and 'comparability' in results:
            st.dataframe(results['comparability'].style.format({"相似度": "{:.2f}", "可比性": "{:.2f}"}))
        elif "可比性分析" not in displayed_dims:
            pass
        else:
            st.write("❌ 没有可比性分析结果")
            
        st.subheader("平衡性结果")
        if "平衡性分析" in displayed_dims and 'sentiment' in results and results['sentiment'] is not None and len(results['sentiment']) > 0:
            st.dataframe(results['sentiment'])
        elif "平衡性分析" not in displayed_dims:
            pass
        else:
            st.write("❌ 没有平衡性分析结果")

# 清理临时文件
import shutil
if analyze_button:
    # 延迟清理，让用户有时间查看结果
    time.sleep(10)
    shutil.rmtree(TEMP_DIR)
