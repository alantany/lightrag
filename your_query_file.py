import logging
import streamlit as st

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def perform_query():
    try:
        # 你的查询代码
        result = some_function_that_might_return_none()
        if result is None:
            logger.error("查询结果为None")
            return "没有找到结果"
        return result[some_index]  # 可能导致'NoneType' object is not subscriptable错误的地方
    except Exception as e:
        logger.exception(f"查询过程中发生错误: {str(e)}")
        return f"查询失败: {str(e)}"

# 在Streamlit应用中使用
query_result = perform_query()
st.write(query_result)
