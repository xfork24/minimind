# 文件导读：OpenAI 兼容聊天接口的最小客户端，默认连接本地 11434/v1。
# 这里的 OpenAI 是客户端库，不意味着请求发送到 OpenAI；实际由 base_url 决定。
# 服务须提前启动且识别请求模型名；顶层 while 在导入时也会读取输入/发送请求。
# 无类/函数封装：顺序为连接、构造历史、请求、消费增量、保存回答。
# 流式（Streaming）边生成边传片段，思考与最终文本使用不同字段。
# messages 的 user/assistant 是用户/助手角色；token 为服务端分词单元，max_tokens 限输出 token 数。
# temperature 调采样分布，top-p 限累计概率候选；参数实际生效方式取决于所连接服务。
from openai import OpenAI

# api_key 是示例占位符，实际认证由所连接服务决定；勿将真实密钥提交仓库。
client = OpenAI(
    api_key="sk-123",
    base_url="http://localhost:11434/v1"
)
stream = True
conversation_history_origin = []
conversation_history = conversation_history_origin.copy()
history_messages_num = 0  # 必须设置为偶数（Q+A），为0则不携带历史对话
while True:
    query = input('[Q]: ')
    conversation_history.append({"role": "user", "content": query})
    response = client.chat.completions.create(
        model="minimind-local:latest",
        # history_messages_num=0 时取最后 1 条，即当前问题；非零取 N 条消息而非 N 轮。
        # 完整列表仍增长，这个切片只限制发送上下文，不释放历史内存。
        messages=conversation_history[-(history_messages_num or 1):],
        stream=stream,
        temperature=0.8,
        max_tokens=2048,
        top_p=0.8,
        extra_body={"chat_template_kwargs": {"open_thinking": True}, "reasoning_effort": "medium"} # 思考开关
    )
    # 非流式一次接收 message；流式逐 chunk 读取 delta。两条路径都记录最终 content。
    if not stream:
        if not response.choices or response.choices[0].message is None:
            raise ValueError("LLM returned empty or filtered response")
        assistant_res = response.choices[0].message.content
        print('[A]: ', assistant_res)
    else:
        print('[A]: ', end='', flush=True)
        assistant_res = ''
        # chunk 可能无 choices/delta，先防空；reasoning_content 灰色打印但不加入 assistant_res。
        # 因此回传历史只含答案，不保存思考文本。
        for chunk in response:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta
            if delta is None:
                continue
            r = getattr(delta, 'reasoning_content', None) or ""
            c = delta.content or ""
            if r:
                print(f'\033[90m{r}\033[0m', end="", flush=True)
            if c:
                print(c, end="", flush=True)
            # 增量不是完整回答，顺序拼接 content；None 按空串处理。
            assistant_res += c

    conversation_history.append({"role": "assistant", "content": assistant_res})
    print('\n\n')
