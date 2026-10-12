import sys
import os
import base64
import uuid

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.runnables import RunnableConfig

from agent.graph import build_agent


# ========== 页面配置 ==========
st.set_page_config(
    page_title="智能私厨 Agent",
    page_icon="🍳",
    layout="centered",
)

st.markdown(
    """
    <style>
        html, body, [data-testid="stAppViewContainer"], section.main {
            overflow-y: auto !important;
            height: auto !important;
            min-height: 100vh !important;
        }
        .block-container {
            padding-bottom: 12rem !important;
        }
        /* 底部输入区固定在底部 */
        div[data-testid="stBottom"] {
            background: white;
            padding-top: 0.5rem;
            border-top: 1px solid #eee;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("🍳 智能私厨 Agent")
st.caption("上传冰箱食材照片，AI 私厨为你量身定制晚餐方案")


# ========== 缓存 Agent ==========
@st.cache_resource
def get_agent():
    return build_agent()


agent = get_agent()


# ========== 会话状态初始化 ==========
if "thread_id" not in st.session_state:
    st.session_state.thread_id = "streamlit_session_1"
if "messages" not in st.session_state:
    st.session_state.messages = []


# ========== 侧边栏：重置对话 ==========
with st.sidebar:
    st.header("⚙️ 控制面板")
    if st.button("🔄 开启新对话"):
        st.session_state.thread_id = f"thread_{uuid.uuid4().hex[:8]}"
        st.session_state.messages = []
        st.success("已开启新对话")
        st.rerun()
    st.caption(f"当前对话 ID：`{st.session_state.thread_id}`")


# ========== 显示历史消息 ==========
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])


# ========== 底部输入区（图片上传 + 输入框） ==========
with st.container(border=False):
    col1, col2 = st.columns([1, 5])

    with col1:
        # 图片上传：点击相机图标上传
        uploaded_file = st.file_uploader(
            "📷",
            type=["jpg", "jpeg", "png", "webp"],
            accept_multiple_files=False,
            label_visibility="collapsed",
            key="image_upload",
        )

    with col2:
        if uploaded_file:
            st.caption(f"📎 已附加：{uploaded_file.name}")
        else:
            st.caption("💡 可直接打字提问，或点击左侧上传食材照片")


# ========== 聊天输入框 ==========
user_input = st.chat_input("输入你的需求，或对菜谱提出修改意见...")

if user_input:
    content_parts = []

    # 如果上传了图片，把图片一并带上
    if uploaded_file:
        img_bytes = uploaded_file.getvalue()
        img_b64 = base64.b64encode(img_bytes).decode("utf-8")
        mime_type = uploaded_file.type or "image/jpeg"
        content_parts.append({
            "type": "image_url",
            "image_url": {"url": f"data:{mime_type};base64,{img_b64}"},
        })

    content_parts.append({"type": "text", "text": user_input})

    user_message = HumanMessage(content=content_parts)

    with st.chat_message("user"):
        if uploaded_file:
            st.caption(f"📎 {uploaded_file.name}")
        st.markdown(user_input)

    # 把用户消息存到 session_state（只存文字，避免图片重复渲染）
    display_content = user_input
    if uploaded_file:
        display_content = f"📎 {uploaded_file.name}\n\n{user_input}"
    st.session_state.messages.append({
        "role": "user",
        "content": display_content,
    })

    config: RunnableConfig = {
        "configurable": {"thread_id": st.session_state.thread_id}
    }

    with st.chat_message("assistant"):
        thinking_placeholder = st.empty()
        thinking_placeholder.info("🤖 私厨正在思考中，请稍候...")

        placeholder = st.empty()
        full_text = ""

        try:
            for chunk, metadata in agent.stream(
                    {"messages": [user_message]},
                    config=config,
                    stream_mode="messages",
            ):
                # 只处理元数据里标记为 "model" 节点的 chunk，排除 middleware 注入的
                node_name = metadata.get("langgraph_node", "") if metadata else ""
                if node_name and node_name != "model":
                    continue

                if (
                        isinstance(chunk, AIMessage)
                        and not getattr(chunk, "tool_calls", None)
                        and isinstance(chunk.content, str)
                        and chunk.content
                ):
                    if not full_text:
                        thinking_placeholder.empty()
                    full_text += chunk.content
                    placeholder.markdown(full_text)

            if not full_text:
                thinking_placeholder.empty()
                placeholder.markdown("⚠️ 模型未返回有效内容，请重试。")
            else:
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": full_text,
                })

        except Exception as e:
            thinking_placeholder.empty()
            st.error(f"出错了：{e}")
            st.stop()