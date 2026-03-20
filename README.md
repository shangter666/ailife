# AiLife - 数字化分身 (Digital Twin) AI Agent

AiLife 是一个基于大语言模型、构建以人为本的“数字化分身”交互管理系统。
本项目从无到有采用了先进的“多重情境记忆架构”与 **LangGraph** 动态工作流编排，使得原本冰冷的聊天机器人在多次交流中不仅能自然对话，还能深入学习、提炼出您独一无二的生活习惯、性格特征和过往的人生轨迹。它是真正**「懂您」**也**「随时随声跟得上您节拍」**的私有数据引擎。

---

## 🌟 核心特性 (Key Features)

### 1. 结构化人格画像 (Structured Memory Profile)
- **原理**：系统在您结束一次聊天回合后，利用独立的反思模块（`reflect_node`），根据历史信息通过高度严格的结构化对象（Pydantic 结合大模型的天然输出机制）更新出属于您的独家 JSON 画像。
- **效果**：大模型每次开口回答前，都会被强制挂入“我是通过这些事实建立起来的分身设定”，保证长线交流中人设不崩。

### 2. 长线情境双轨记忆 (Episodic Vector Memory)
- **原理**：除了高度浓缩的 JSON 以外，系统另接入并自动启用了 **ChromaDB** 向量检索数据库搭配 `sentence-transformers` 轻量级本地开源模型。
- **效果**：哪怕再细碎的曾经的一句“昨天我弄丢了蓝笔”，它也能在您后续聊天里，触发最相关的 3 条上下文记忆（`enrich_context_node`），被顺理成章地捞回来当做回答辅佐参考。摆脱传统大语言模型死板的长文本压缩后丢失细节的通病。

### 3. Agentic 自动路由挂接 (Tool-Use / LangGraph)
- **原理**：系统已非线性写死的一问一答，由于全权使用了时下最先进的 LangGraph 树立循环边界，Agent 本身拥有调用本地 Python 库或现实接口的能力。
- **效果**：自带系统级别的 `tools_condition` 条件。如果它判断您的需求必须借助外部信息，便在开口前自动绕道先触发对应的获取本地时间的函数（甚至未来可拓展联网、视觉等），然后再给您真正想要的结果。

### 4. 彻底的流式体验 (Asynchronous Streaming UI)
- **原理**：后端使用了基于 Python 高性能 Web 框架 **FastAPI** 的 `StreamingResponse`。
- **效果**：网页一开口，就像真的有个有血有肉的人陪着您面对面打字聊天，逐字流式缓冲到前端；且对话落盘等所有底层大活全都不在主线程阻塞，即写即存，飞一般顺滑。

---

## 🛠 技术栈 (Tech Stack)

* **后端主框架**: Python 3, FastAPI, Uvicorn
* **Agent核心链路编排**: LangChain, LangGraph
* **数据与接口规整**: Pydantic, ChatOpenAI 接口兼容器
* **AI记忆引擎组件**: ChromaDB, HuggingFaceEmbeddings (本地无痛生成 Vector Embeddings)
* **纯享版沉浸前端**: 原生 HTML5 / Vanilla JavaScript / CSS (存放在 `/test` 同级目录中)

---

## 🚀 快速启动指南 (Quick Start)

### 第 1 步：配置运行环境
推荐在纯净的 Python 虚拟环境下运行。在项目根目录下激活命令行：
```bash
python3 -m venv venv
source venv/bin/activate
```

### 第 2 步：安装核心依赖
执行如下命令拉取所有技术栈生态（需耐心等待包含了底层如 torch 等庞大计算基座的下载）：
```bash
pip install -r requirements.txt
```

### 第 3 步：录入专属密钥
请找到根目录内的基础配置文件 `config.yaml`，按需替换成您对应的国内外云端 AI 加速平台的 Key：
```yaml
llm_settings:
  api_key: "sk-您的私有密钥"
  base_url: "比如阿里云千问等兼容 OpenAI 格式的调用前缀"
  model_name: "模型名称"
```

### 第 4 步：启动您的分身服务！
确认环境就绪后，以防丢失最新的文件修改，开启热重载运行后端：
```bash
uvicorn main:app --reload
```
看到日志输出 `Uvicorn running on http://0.0.0.0:8000` 后即代表挂载成功。

### 第 5 步：体验前端 UI
无需庞大的前端环境！只需用任意现代浏览器直接把项目中名为 `/test/index.html` 的文件双击打开。
- 在左侧输入或保留您的独立 `User ID` 身份标识符。
- 在页面中下半区域直接开聊体验。
- 点击 **"Load Memory"** 时，除了会在左侧栏展示您的长期结构性格雷达外，系统会自动下发并以时间轴加载出您历史积累沉淀的所有语义气泡碎片。
