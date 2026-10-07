# 文件导读：dataset 是数据处理包，训练入口通过 dataset.lm_dataset 导入样本适配器。
# 此初始化文件不读取 JSONL（逐行 JSON）数据、不创建分词器，也不启动训练。
# 继续阅读 lm_dataset.py，先理解 token、输入序列与监督标签的对应关系。
