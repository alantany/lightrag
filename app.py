import sys
import os
import io
import PyPDF2
import logging

# 将 LightRAG 目录添加到 Python 路径
current_dir = os.path.dirname(os.path.abspath(__file__))
lightrag_dir = os.path.join(current_dir, 'LightRAG')
sys.path.append(lightrag_dir)

import streamlit as st
from lightrag.lightrag import LightRAG, QueryParam
from lightrag.llm import openai_complete_if_cache, openai_embedding
from lightrag.utils import EmbeddingFunc

@st.cache_resource
def load_lightrag():
    return LightRAG(
        working_dir="./lightrag_data",
        # 移除 log_file 参数
        # 其他参数保持不变
    )

def extract_text_from_pdf(pdf_file):
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text() + "\n"
    return text

def show_process_explanation():
    st.subheader("LightRAG 数据处理过程")
    st.write("""
    1. 文档分块：将上传的 PDF 文档分割成小块文本。
    2. 实体提取：从每个文本块中提取关键实体和关系。
    3. 向量化：将文本块、实体和关系转换为向量表示。
    4. 知识图谱构建：基于提取的实体和关系构建知识图谱。
    5. 存储：将向量和图谱信息存储在相应的数据结构中。
    """)

def get_query_mode_explanation(mode):
    explanations = {
        "naive": """
        **Naive（朴素）模式：**
        - 这是最简单直接的查询方式。
        - **工作原理：** 将用户的查询转换为向量，然后在文本块的向量表示中进行相似度搜索。
        - **优点：** 速度快，适用于简单的信息检索任务。
        - **缺点：** 不利用知识图谱，可能会错过一些上下文信息。
        - **适用场景：** 当需要快速检索或问题比较直接时。
        """,
        "local": """
        **Local（局部）模式：**
        - 这种模式利用知识图谱，但只在查询相关的局部区域进行搜索。
        - **工作原理：** 首先识别查询中的关键实体，然后在知识图谱中查找与这些实体直接相关的节点和边。
        - **优点：** 能够捕捉到与查询直接相关的上下文信息，同时保持较快的搜索速度。
        - **缺点：** 可能会错过一些间接相关但重要的信息。
        - **适用场景：** 当需要考虑一定的上下文，但又不想搜索范围过大时。
        """,
        "global": """
        **Global（全局）模式：**
        - 这种模式在整个知识图谱中进行广泛搜索。
        - **工作原理：** 从查询相关的实体开始，在整个知识图谱中进行广度优先或深度优先搜索。
        - **优点：** 能够发现潜在的、间接相关的信息，适合复杂查询。
        - **缺点：** 搜索速度较慢，可能会引入一些不太相关的信息。
        - **适用场景：** 当需要全面了解某个主题，或处理复杂的、需要多方面信息的查询时。
        """,
        "hybrid": """
        **Hybrid（混合）模式：**
        - 这种模式结合了局部和全局搜索的优点。
        - **工作原理：** 首先进行局部搜索，然后根据需要扩展到全局搜索，可能会使用一些启发式策略来平衡搜索范围和深度。
        - **优点：** 能够在相关性和全面性之间取得平衡，适应性强。
        - **缺点：** 实现复杂，可能需要更多的计算资源。
        - **适用场景：** 当不确定查询的复杂度，或需要在速度和全面性之间取得平衡时。
        """
    }
    return explanations.get(mode, "未知查询模式")

def get_loaded_files(data_dir):
    loaded_files = []
    for filename in os.listdir(data_dir):
        if filename.endswith('.json') or filename.endswith('.graphml'):
            loaded_files.append(filename)
    return loaded_files

def main():
    st.set_page_config(page_title="LightRAG 医疗文档分析", page_icon="🏥", layout="wide")
    
    # 配置日志
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    st.title("🏥 LightRAG 医疗文档分析系统")

    # 初始化LightRAG
    lightrag = load_lightrag()

    # 侧边栏
    st.sidebar.header("关于LightRAG")
    st.sidebar.write("""
    LightRAG是一个轻量级的检索增强生成系统。
    它通过结合检索和生成模型，提供更精确和相关的回答。
    """)

    # 显示数据状态
    data_dir = os.path.join(current_dir, 'lightrag_data')
    if os.path.exists(data_dir) and os.listdir(data_dir):
        st.sidebar.success("数据已加载")
        
        # 显示已加载的文件
        loaded_files = get_loaded_files(data_dir)
        st.sidebar.subheader("已加载的文件:")
        for file in loaded_files:
            st.sidebar.text(file)
    else:
        st.sidebar.warning("尚未加载数据")

    # 文件上传部分
    st.header("上传医疗文档")
    uploaded_file = st.file_uploader("选择一个PDF文件", type="pdf")
    if uploaded_file is not None:
        if st.button("处理文档"):
            with st.spinner('正在处理文档...'):
                pdf_text = extract_text_from_pdf(uploaded_file)
                lightrag.insert(pdf_text)
            st.success("文档处理完成！")
            show_process_explanation()
            st.rerun()  # 使用 st.rerun() 替代 st.experimental_rerun()

    # 查询部分
    st.header("查询")
    query_mode = st.selectbox(
        "选择查询模式",
        ("naive", "local", "global", "hybrid"),
        format_func=lambda x: {"naive": "朴素", "local": "局部", "global": "全局", "hybrid": "混合"}[x]
    )

    # 显示选中的查询模式说明
    st.markdown(get_query_mode_explanation(query_mode))

    query = st.text_input("请输入您的问题:", placeholder="例如：患者的主要症状是什么？")

    if query:  # 如果有输入，直接执行查询
        if not os.listdir(data_dir):
            st.warning("请先上传并处理文档，然后再进行查询。")
        else:
            with st.spinner('正在检索中...'):
                try:
                    result = lightrag.query(query, param=QueryParam(mode=query_mode))
                    if result:
                        st.subheader("检索结果:")
                        st.write(result)
                    else:
                        st.warning("未能获取查询结果，请检查系统日志或重试。")
                except Exception as e:
                    st.error(f"查询过程中发生错误: {str(e)}")
                    st.info("请确保已经正确上传并处理了文档。如果问题持续，请尝试清理数据并重新上传文档。")
                    logging.exception("Query error")

    # 添加清理数据的功能
    if st.sidebar.button("清理所有数据"):
        import shutil
        shutil.rmtree(data_dir, ignore_errors=True)
        os.makedirs(data_dir, exist_ok=True)
        st.sidebar.success("所有数据已清理")
        st.rerun()  # 这里也使用 st.rerun()

if __name__ == "__main__":
    main()