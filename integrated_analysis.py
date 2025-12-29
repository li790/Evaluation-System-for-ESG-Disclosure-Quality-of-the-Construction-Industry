import os
import torch
import re
import csv
import numpy as np
import pandas as pd
import glob
from transformers import BertTokenizer, BertForSequenceClassification, AutoTokenizer, AutoModelForSequenceClassification
from sentence_transformers import SentenceTransformer

class IntegratedAnalysis:
    def __init__(self):
        # 配置参数
        self.model_paths = {
            'sentiment': {
                'base': './models--hfl--chinese-bert-wwm/snapshots/ab0aa81da273504efc8540aa4d0bbaa3016a1bb5',
                'checkpoint': './results1/checkpoint-3186'
            },
            'summary': {
                'base': './models--hfl--chinese-bert-wwm/snapshots/ab0aa81da273504efc8540aa4d0bbaa3016a1bb5',
                'checkpoint': './results/checkpoint-1425'
            },
            'substantive': {
                'base': './models--hfl--chinese-bert-wwm/snapshots/ab0aa81da273504efc8540aa4d0bbaa3016a1bb5',
                'checkpoint': './results2/checkpoint-582'
            }
        }
        
        # 标签映射
        self.summary_label_map = {
            1: '绿色施工与环境管理',
            2: '生态保护与修复',
            3: '资源节约与循环利用',
            4: '可再生能源',
            5: '员工权益',
            6: '安全生产与应急体系',
            7: '工程质量管理',
            8: '供应链管理',
            9: '乡村振兴',
            10: '海外社会责任',
            11: '公司治理结构',
            12: '信息披露与投资者关系',
            13: '风险与合规管理',
            14: '企业战略与可持续发展规划',
            15: '工程技术创新与数字化管理'
        }
        
        self.substantive_label_map = {
            0: '其他/非实质性文本',
            1: '管理体系与培训',
            2: '具体措施与投入',
            3: '绩效与事件',
            4: '绿色技术与材料',
            5: '资源与排放管理',
            6: '目标与认证',
            7: '薪酬与福利',
            8: '培训与晋升',
            9: '员工关怀与满意度',
            10: '社区影响管理',
            11: '公益与本地化',
            12: '乡村振兴参与'
        }
        
        # 实质性维度列表（排除"其他/非实质性文本"）
        self.substantive_dimensions = [label for idx, label in self.substantive_label_map.items() if idx != 0]
        
        # 初始化模型和分词器
        self.models = {}
        self.tokenizers = {}
        
        # ESG相似度模型配置
        self.esg_model_path = './esg_model'
        self.esg_sbert = None
    
    def _safe_move_to_device(self, model, device="cuda"):
        """安全地将模型移动到指定设备，处理meta tensor问题"""
        try:
            # 检查模型是否在meta设备上
            is_meta = any(param.device.type == 'meta' for param in model.parameters())
            
            if is_meta:
                print(f"  检测到meta设备，使用to_empty()方法...")
                # 使用to_empty()先移动到目标设备
                model = model.to_empty(device=device)
                # 然后重新加载权重
                print(f"  模型已移动到 {device}")
            else:
                # 直接移动
                model = model.to(device)
                print(f"  模型已移动到 {device}")
            
            return model
        except Exception as e:
            print(f"  设备移动失败: {e}")
            print(f"  回退到CPU")
            return model.cpu()
    
    def load_sentiment_model(self):
        """加载情感分析模型和分词器"""
        if 'sentiment' not in self.models:
            print("正在加载情感分析模型...")
            self.tokenizers['sentiment'] = BertTokenizer.from_pretrained(self.model_paths['sentiment']['base'])
            # 加载模型
            self.models['sentiment'] = BertForSequenceClassification.from_pretrained(
                self.model_paths['sentiment']['checkpoint']
            )
            
            # 检查GPU可用性并安全移动
            if torch.cuda.is_available():
                self.models['sentiment'] = self._safe_move_to_device(self.models['sentiment'], "cuda")
            print("情感分析模型加载完成！")
    
    def load_summary_model(self):
        """加载完整性分析模型和分词器"""
        if 'summary' not in self.models:
            print("正在加载完整性分析模型...")
            self.tokenizers['summary'] = BertTokenizer.from_pretrained(self.model_paths['summary']['base'])
            # 加载模型
            self.models['summary'] = BertForSequenceClassification.from_pretrained(
                self.model_paths['summary']['checkpoint'],
                num_labels=16  # 15个类别 + 1个其他类别
            )
            self.models['summary'].eval()  # 设置为评估模式
            
            # 检查GPU可用性并安全移动
            if torch.cuda.is_available():
                self.models['summary'] = self._safe_move_to_device(self.models['summary'], "cuda")
            print("完整性分析模型加载完成！")
    
    def load_substantive_model(self):
        """加载实质性分析模型和分词器"""
        if 'substantive' not in self.models:
            print("正在加载实质性分析模型...")
            self.tokenizers['substantive'] = BertTokenizer.from_pretrained(self.model_paths['substantive']['base'])
            # 加载模型
            self.models['substantive'] = BertForSequenceClassification.from_pretrained(
                self.model_paths['substantive']['checkpoint'],
                num_labels=13
            )
            self.models['substantive'].eval()  # 设置为评估模式
            
            # 检查GPU可用性并安全移动
            if torch.cuda.is_available():
                self.models['substantive'] = self._safe_move_to_device(self.models['substantive'], "cuda")
            print("实质性分析模型加载完成！")
    
    def load_esg_model(self):
        """加载ESG相似度模型"""
        if self.esg_sbert is None:
            print("正在加载ESG相似度模型...")
            try:
                # 尝试介指定路径加载模型
                if os.path.exists(self.esg_model_path):
                    print(f"从本地路径加载ESG模型: {self.esg_model_path}")
                    # SentenceTransformer需要特殊处理，先在CPU上加载
                    self.esg_sbert = SentenceTransformer(self.esg_model_path, device='cpu')
                else:
                    # 尝试从 Hugging Face 下载默认模型
                    print("从 Hugging Face 下载ESG默认模型...")
                    self.esg_sbert = SentenceTransformer("shibing624/text2vec-base-chinese", device='cpu')
                    
                # 如果GPU可用，才移动到GPU
                if torch.cuda.is_available():
                    try:
                        print("  尝试将ESG模型移动到GPU...")
                        self.esg_sbert = self.esg_sbert.to('cuda')
                        print("  ESG模型已移动到GPU")
                    except Exception as e:
                        print(f"  GPU移动失败，使用CPU: {e}")
                    
                print("ESG相似度模型加载完成！")
            except Exception as e:
                print(f"ESG模型加载失败: {e}")
                print("请确保模型路径正确或网络连接正常。")
                raise
    
    def load_all_models(self):
        """加载所有模型"""
        self.load_sentiment_model()
        self.load_summary_model()
        self.load_substantive_model()
        self.load_esg_model()
    
    def analyze_sentiment(self, file_path):
        """分析文件的情感"""
        self.load_sentiment_model()
        
        # 读取文件内容
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 将文件内容按行分割
        lines = content.strip().split('\n')
        lines = [line.strip() for line in lines if line.strip()]
        
        if not lines:
            return None
        
        # 统计各类别的数量
        neutral_count = 0
        negative_count = 0
        positive_count = 0
        
        # 分析每一行的情感
        for line in lines:
            if len(line) < 12:  # 跳过太短的行
                continue
            inputs = self.tokenizers['sentiment'](line, return_tensors="pt", truncation=True, padding=True, max_length=512)
            
            # 检查GPU可用性
            if torch.cuda.is_available():
                inputs = {k: v.to("cuda") for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.models['sentiment'](**inputs)
            probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
            predicted_class = torch.argmax(probabilities, dim=-1).item()
            
            if predicted_class == 0:
                neutral_count += 1
            elif predicted_class == 1:
                negative_count += 1
            elif predicted_class == 2:
                positive_count += 1
        
        # 计算各类别的比例
        total = len(lines)
        neutral_ratio = neutral_count / total
        negative_ratio = negative_count / total
        positive_ratio = positive_count / total
        
        # 计算情感评分
        sentiment_score = 1 - abs(positive_ratio - negative_ratio)
        
        # 返回统计结果
        return {
            'file_name': os.path.basename(file_path),
            'total_lines': total,
            'neutral_count': neutral_count,
            'negative_count': negative_count,
            'positive_count': positive_count,
            'neutral_ratio': neutral_ratio,
            'negative_ratio': negative_ratio,
            'positive_ratio': positive_ratio,
            'sentiment_score': sentiment_score
        }
    
    def contains_numeric_info(self, text):
        """检测文本中是否包含数值信息"""
        numeric_pattern = r'\b\d+(\.\d+)?%?\b'
        return bool(re.search(numeric_pattern, text))
    
    def classify_summary(self, file_path, weights=None):
        """分析文档完整性并计算评分"""
        self.load_summary_model()
        
        # 设置默认权重
        if weights is None:
            weights = {label: 1.0 for label in self.summary_label_map.keys()}
        
        # 读取文件内容
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 将文本分成句子或段落进行分析
        sentences = re.split(r'[。！？\n]', content)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        # 初始化评分字典
        scores = {label: 0 for label in self.summary_label_map.keys()}  # 排除其他类（标签0）
        
        # 分析每个句子
        for sentence in sentences:
            if len(sentence) < 12:  # 跳过太短的句子
                continue
            
            # 使用BERT模型进行分类
            inputs = self.tokenizers['summary'](sentence, return_tensors="pt", truncation=True, padding=True, max_length=512)
            
            # 检查GPU可用性
            if torch.cuda.is_available():
                inputs = {k: v.to("cuda") for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.models['summary'](**inputs)
                predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
                predicted_class = torch.argmax(predictions, dim=-1).item()
                confidence = predictions[0][predicted_class].item()
            
            # 只有当置信度超过阈值时才进行分类
            if confidence > 0.5:
                # 跳过其他类别（标签0）
                if predicted_class != 0:
                    # 检查是否包含数值信息，取最高分
                    if self.contains_numeric_info(sentence):
                        # 包含数值信息得2分
                        scores[predicted_class] = max(scores[predicted_class], 2)
                    else:
                        # 不包含数值信息至少得1分
                        scores[predicted_class] = max(scores[predicted_class], 1)
        
        # 计算加权总分
        weighted_sum = sum(scores[label] * weights[label] for label in scores)
        # 计算权重总和
        weight_sum = sum(weights.values())
        # 计算平均分数
        integrity_score = weighted_sum / weight_sum if weight_sum > 0 else 0
        
        return {
            'file_name': os.path.basename(file_path),
            'scores': scores,
            'integrity_score': integrity_score
        }
    
    def embed_long_text(self, text: str, max_seq_len: int = 512, embedding_size=768) -> np.ndarray:
        """
        长文本 → 按句分块 → ESG 领域 SBERT 编码 → 平均池化
        
        参数:
            text: 要编码的长文本
            max_seq_len: 最大序列长度 (默认512，防止显存溢出)
            embedding_size: 嵌入维度
        
        返回:
            文本的平均嵌入向量
        """
        # 移除多余空白字符并确保文本不为空
        text = text.strip().replace('\n', ' ')
        if not text:
            # 返回零向量
            return np.zeros(embedding_size)
        
        # 按句号分割句子
        sentences = text.split('。')
        # 过滤掉空句子
        sentences = [sent.strip() for sent in sentences if sent.strip()]
        
        if not sentences:
            # 返回零向量
            return np.zeros(embedding_size)
        
        # 构建文本块
        chunks, cur, cur_len = [], [], 0
        for sent in sentences:
            # 计算当前句子的token长度
            tok_len = len(self.esg_sbert.tokenize(sent))
            
            # 如果添加当前句子后超过max_seq_len且当前块不为空
            if cur_len + tok_len > max_seq_len and cur:
                chunks.append(''.join(cur))
                cur, cur_len = [], 0
            
            # 添加当前句子到当前块
            cur.append(sent)
            cur_len += tok_len
        
        # 添加最后一个块
        if cur:
            chunks.append(''.join(cur))
        
        # 过滤掉空块
        chunks = [chunk for chunk in chunks if chunk.strip()]
        
        if not chunks:
            # 返回零向量
            return np.zeros(embedding_size)
        
        # 使用ESG模型编码所有块（batch_size=16防止显存溢出）
        embs = self.esg_sbert.encode(chunks, batch_size=16, convert_to_numpy=True)
        
        # 返回平均池化后的向量
        return embs.mean(axis=0)
    
    def cosine_sim(self, a, b):
        """
        计算两个向量的余弦相似度
        
        参数:
            a: 第一个向量
            b: 第二个向量
        
        返回:
            余弦相似度分数
        """
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
    
    def read_file(self, file_path):
        """
        读取文件内容
        
        参数:
            file_path: 文件路径
        
        返回:
            文件内容字符串
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        except UnicodeDecodeError:
            with open(file_path, 'r', encoding='gbk') as f:
                return f.read()
    
    def calculate_yearly_similarities(self, sorted_files):
        """
        计算相邻年份报告的相似度
        
        参数:
            sorted_files: 按年份排序的文件路径列表
        
        返回:
            年份对列表和相似度分数列表
        """
        year_pairs = []
        scores = []
        
        total_comparisons = len(sorted_files) - 1
        print(f"开始计算相邻年份相似度...共需计算 {total_comparisons} 组数据")
        
        for i in range(total_comparisons):
            file1 = sorted_files[i]
            file2 = sorted_files[i + 1]
            
            # 提取年份信息
            year1 = re.search(r"(\d{4})\w*", os.path.basename(file1)).group(1)
            year2 = re.search(r"(\d{4})\w*", os.path.basename(file2)).group(1)
            year_pair = f"{year1}-{year2}"
            
            print(f"\n处理 {year_pair}...")
            
            try:
                # 读取文件内容
                print(f"读取 {year1} 报告...")
                txt1 = self.read_file(file1)
                
                print(f"读取 {year2} 报告...")
                txt2 = self.read_file(file2)
                
                # 生成嵌入
                print(f"编码 {year1} 报告...")
                vec1 = self.embed_long_text(txt1)
                
                print(f"编码 {year2} 报告...")
                vec2 = self.embed_long_text(txt2)
                
                # 计算相似度
                score = self.cosine_sim(vec1, vec2)
                year_pairs.append(year_pair)
                scores.append(score)
                
                print(f"{year_pair} 相似度：{score:.4f}")
                if score > 0.85:
                    print("→ 两篇报告高度重合或主题几乎一致。")
                elif score > 0.65:
                    print("→ 核心议题相近，但存在明显差异。")
                else:
                    print("→ 内容方向差异较大。")
                    
                # 清理内存
                del txt1, txt2, vec1, vec2
                
            except Exception as e:
                print(f"计算{year_pair}相似度失败: {e}")
                continue
        
        return year_pairs, scores
    
    def analyze_substantive(self, file_path, weights=None):
        """分析文件的实质性"""
        self.load_substantive_model()
        
        # 设置默认权重
        if weights is None:
            weights = {dimension: 1.0 for dimension in self.substantive_dimensions}
        
        # 读取文件内容
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 将文本分割成段落或句子进行分析
        paragraphs = [p.strip() for p in content.split('\n') if p.strip()]
        
        # 初始化各维度得分
        dimension_scores = {dimension: 0 for dimension in self.substantive_dimensions}
        dimension_numeric = {dimension: False for dimension in self.substantive_dimensions}
        
        # 分析每个段落
        for para in paragraphs:
            if para and len(para) >= 12:  # 跳过太短的段落
                inputs = self.tokenizers['substantive'](para, return_tensors="pt", truncation=True, padding=True, max_length=512)
                
                # 检查GPU可用性
                if torch.cuda.is_available():
                    inputs = {k: v.to("cuda") for k, v in inputs.items()}
                
                with torch.no_grad():
                    outputs = self.models['substantive'](**inputs)
                    predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
                    predicted_class = torch.argmax(predictions, dim=-1).item()
                    confidence = predictions[0][predicted_class].item()
                
                predicted_label = self.substantive_label_map[predicted_class]
                
                # 只考虑实质性维度
                if predicted_label in self.substantive_dimensions:
                    # 如果有该维度的文本，至少得1分
                    dimension_scores[predicted_label] = max(dimension_scores[predicted_label], 1)
                    
                    # 检查是否包含数值
                    if self.contains_numeric_info(para):
                        dimension_numeric[predicted_label] = True
        
        # 更新得分：如果包含数值，得2分
        for dimension in self.substantive_dimensions:
            if dimension_numeric[dimension]:
                dimension_scores[dimension] = 2
        
        # 计算完整性评分
        total_score = sum(dimension_scores[dim] * weights[dim] for dim in self.substantive_dimensions)
        total_weight = sum(weights.values())
        integrity_score = total_score / total_weight if total_weight > 0 else 0
        
        return {
            'file_name': os.path.basename(file_path),
            'dimension_scores': dimension_scores,
            'integrity_score': integrity_score
        }
    
    def analyze_all_files(self, input_folder, sentiment_weights=None, summary_weights=None, substantive_weights=None, progress_callback=None):
        """分析文件夹中所有文件的三种评价指标"""
        # 创建输出文件夹
        output_folder = "./综合评价结果"
        os.makedirs(output_folder, exist_ok=True)
        
        # 获取所有txt文件
        txt_files = [f for f in os.listdir(input_folder) if f.endswith('.txt')]
        
        if not txt_files:
            print(f"错误：文件夹 {input_folder} 中没有找到.txt文件！")
            return
        
        print(f"正在分析文件夹：{input_folder}")
        print(f"找到 {len(txt_files)} 个.txt文件")
        
        # 按年份排序文件
        def get_year(file_name):
            try:
                year = re.search(r"(\d{4})", file_name)
                return int(year.group(1)) if year else 0
            except:
                return 0
        
        # 构建完整文件路径并排序
        file_paths = [os.path.join(input_folder, f) for f in txt_files]
        sorted_file_paths = sorted(file_paths, key=lambda x: get_year(os.path.basename(x)))
        
        # 提取排序后的文件名
        sorted_file_names = [os.path.basename(f) for f in sorted_file_paths]
        print(f"\n按年份排序后的文件：{sorted_file_names}")
        
        # 计算相邻年份的相似度和可比性
        if len(sorted_file_paths) > 1:
            print("\n=== 开始计算相邻年份报告的可比性 ===")
            year_pairs, similarity_scores = self.calculate_yearly_similarities(sorted_file_paths)
            
            # 计算可比性 (可比性 = 1 - 相似度)
            comparability_scores = [1 - score for score in similarity_scores]
            
            # 保存可比性结果
            comparability_data = {
                '年份对': year_pairs,
                '相似度': similarity_scores,
                '可比性': comparability_scores
            }
            df_comparability = pd.DataFrame(comparability_data)
            comparability_output_path = os.path.join(output_folder, "comparability_results.csv")
            df_comparability.to_csv(comparability_output_path, index=False, encoding='utf-8-sig')
            print(f"\n可比性分析结果已保存到：{comparability_output_path}")
        else:
            print("\n文件数量不足，无法计算可比性")
        
        # 初始化结果列表
        sentiment_results = []
        summary_results = []
        substantive_results = []
        
        # 分析每个文件
        for i, txt_file in enumerate(txt_files, 1):
            file_path = os.path.join(input_folder, txt_file)
            print(f"\n正在分析文件 {i}/{len(txt_files)}：{txt_file}")
            
            # 调用进度回调
            if progress_callback:
                progress = i / len(txt_files)
                progress_callback(progress, txt_file)
            
            try:
                # 情感分析
                sentiment_result = self.analyze_sentiment(file_path)
                if sentiment_result:
                    sentiment_results.append(sentiment_result)
                
                # 完整性分析
                summary_result = self.classify_summary(file_path, summary_weights)
                summary_results.append(summary_result)
                
                # 实质性分析
                substantive_result = self.analyze_substantive(file_path, substantive_weights)
                substantive_results.append(substantive_result)
                
                print(f"文件 {txt_file} 分析完成")
            except Exception as e:
                print(f"分析文件 {txt_file} 时出错：{e}")
                continue
        
        # 保存情感分析结果（使用中文表头，只保留需要的列）
        sentiment_output_path = os.path.join(output_folder, "sentiment_analysis_results.csv")
        df_sentiment = pd.DataFrame(sentiment_results)
        
        # 重命名列为中文并只保留需要的列
        df_sentiment_display = pd.DataFrame({
            '文件名': df_sentiment['file_name'],
            '积极比例': df_sentiment['positive_ratio'],
            '消极比例': df_sentiment['negative_ratio'],
            '中立比例': df_sentiment['neutral_ratio'],
            '平衡性评分': df_sentiment['sentiment_score']
        })
        
        df_sentiment_display.to_csv(sentiment_output_path, index=False, encoding='utf-8-sig')
        print(f"\n情感分析结果已保存到：{sentiment_output_path}")
        
        # 保存完整性分析结果
        summary_output_path = os.path.join(output_folder, "integrity_analysis_results.csv")
        
        # 准备完整性结果的DataFrame
        summary_data = []
        for result in summary_results:
            row = {
                '文件名': result['file_name'],
                '完整性评分': result['integrity_score']
            }
            for label, name in self.summary_label_map.items():
                row[name] = result['scores'][label]
            summary_data.append(row)
        
        df_summary = pd.DataFrame(summary_data)
        df_summary.to_csv(summary_output_path, index=False, encoding='utf-8-sig')
        print(f"完整性分析结果已保存到：{summary_output_path}")
        
        # 保存实质性分析结果
        substantive_output_path = os.path.join(output_folder, "substantive_analysis_results.csv")
        
        # 准备实质性结果的DataFrame
        substantive_data = {
            '文件名': []
        }
        for dimension in self.substantive_dimensions:
            substantive_data[dimension] = []
        substantive_data['实质性评分'] = []
        
        for result in substantive_results:
            substantive_data['文件名'].append(result['file_name'])
            for dimension in self.substantive_dimensions:
                substantive_data[dimension].append(result['dimension_scores'][dimension])
            substantive_data['实质性评分'].append(result['integrity_score'])
        
        df_substantive = pd.DataFrame(substantive_data)
        df_substantive.to_csv(substantive_output_path, index=False, encoding='utf-8-sig')
        print(f"实质性分析结果已保存到：{substantive_output_path}")
        
        # 生成综合评分结果
        combined_results = []
        
        # 创建文件名到可比性的映射（针对每个文件对）
        file_comparability_map = {}
        if len(sorted_file_paths) > 1:
            # 将文件路径与年份对关联
            for i in range(len(year_pairs)):
                # 获取年份对中的后一个年份的文件名
                current_year = year_pairs[i].split('-')[1]
                for file_path in sorted_file_paths:
                    if current_year in os.path.basename(file_path):
                        file_comparability_map[os.path.basename(file_path)] = comparability_scores[i]
                        break
        
        for i in range(len(sentiment_results)):
            file_name = sentiment_results[i]['file_name']
            comparability = file_comparability_map.get(file_name, '')
            
            combined = {
                '文件名': file_name,
                '情感评分': sentiment_results[i]['sentiment_score'],
                '完整性评分': summary_results[i]['integrity_score'],
                '实质性评分': substantive_results[i]['integrity_score'],
                '可比性评分': comparability if comparability != '' else 0,
                '可读性评分': 0,  # 占位符，需要后续实现
                '可靠性评分': 0,  # 占位符，需要后续实现
                '综合评分': (sentiment_results[i]['sentiment_score'] + summary_results[i]['integrity_score'] + substantive_results[i]['integrity_score'] + (comparability if comparability != '' else 0) + 0 + 0) / 6,
            }
            combined_results.append(combined)
        
        combined_output_path = os.path.join(output_folder, "combined_analysis_results.csv")
        df_combined = pd.DataFrame(combined_results)
        df_combined.to_csv(combined_output_path, index=False, encoding='utf-8-sig')
        print(f"综合评分结果已保存到：{combined_output_path}")
        
        print(f"\n所有文件分析完成！共分析了 {len(sentiment_results)} 个文件。")
        print(f"结果保存在：{output_folder}")
    
    def get_user_weights(self, analysis_type):
        """获取用户输入的权重"""
        weights = {}
        
        if analysis_type == 'summary':
            print("\n请为每个完整性分析维度输入权重（默认值为1，直接按Enter保持默认）:")
            for label, name in self.summary_label_map.items():
                while True:
                    weight_input = input(f"{name}: ")
                    if weight_input.strip() == "":
                        # 使用默认值
                        weights[label] = 1.0
                        break
                    try:
                        weight = float(weight_input)
                        weights[label] = weight
                        break
                    except ValueError:
                        print("请输入有效的数字！")
        
        elif analysis_type == 'substantive':
            print("\n请为每个实质性维度输入权重（默认值为1，直接按Enter保持默认）:")
            for dimension in self.substantive_dimensions:
                while True:
                    weight_input = input(f"{dimension}: ")
                    if weight_input.strip() == "":
                        # 使用默认值
                        weights[dimension] = 1.0
                        break
                    try:
                        weight = float(weight_input)
                        weights[dimension] = weight
                        break
                    except ValueError:
                        print("请输入有效的数字！")
        
        elif analysis_type == 'sentiment':
            # 情感分析不需要权重
            weights = None
        
        return weights
    
    def main(self):
        """主函数"""
        print("文件综合评价分析工具")
        print("=" * 50)
        
        # 加载所有模型
        self.load_all_models()
        
        # 获取用户输入的权重
        use_custom_weights = input("\n是否使用自定义权重？(y/n): ").lower()
        
        sentiment_weights = None
        summary_weights = None
        substantive_weights = None
        
        if use_custom_weights == 'y':
            summary_weights = self.get_user_weights('summary')
            substantive_weights = self.get_user_weights('substantive')
        
        # 直接使用"汇总"文件夹作为分析路径
        input_folder = "./汇总"
        
        # 检查路径是否存在
        if not os.path.exists(input_folder):
            print("错误：路径不存在！")
            return
        
        # 分析所有文件
        self.analyze_all_files(input_folder, sentiment_weights, summary_weights, substantive_weights)

if __name__ == "__main__":
    analyzer = IntegratedAnalysis()
    analyzer.main()
