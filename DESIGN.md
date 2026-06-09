# Memory Fusion v2 — 设计规范

> 从聊天记录反向工程的完整设计文档。方案C：双管线 + Sidecar。

---

## 设计第1部分：架构总览

### 三层记忆体系（保持和Hermes现有分层兼容）

```
L1: MEMORY.md + FTS5    ← 热缓存+消息历史（不动）
L2: Memory Fusion v2    ← 新系统（替代Honcho+Preventive+Fusion）
L3: Qdrant Cloud        ← 向量存储（已部署）
```

### L2 内部架构

```
┌─────────────────────────────────────────────────────────┐
│                   MemoryCore (门面)                       │
│  remember() / recall() / hybrid_search() / digest()      │
├──────────────┬──────────────┬───────────────────────────┤
│ WritePipeline │ ReadPipeline │     Sidecar Workers        │
│               │              │                            │
│ • SVO提取     │ • 向量检索   │ • digest_worker            │
│ • 批量embedding│ • CrossEncoder│  (自动蒸馏)              │
│ • 语义去重     │ • 4信号排名  │ • conflict_worker          │
│ • 重要度过滤   │ • 辩证合成   │  (冲突检测)               │
│ • Qdrant写入   │ • 访问计数   │ • feedback_worker          │
│               │              │  (反馈追踪)               │
│               │              │ • cleanup_worker           │
│               │              │  (过期清理)               │
│               │              │ • promote_worker           │
│               │              │  (晋升写入)               │
├──────────────┴──────────────┴───────────────────────────┤
│              Storage Adapters                             │
│  QdrantAdapter (向量)  /  SQLiteAdapter (元数据+任务队列) │
├─────────────────────────────────────────────────────────┤
│              _utils (共享工具)                            │
│  retry / embed_text / embed_batch / cosine_similarity    │
│  strip_markdown_json / sanitize_text                     │
└─────────────────────────────────────────────────────────┘
```

### 插件适配层

```
hermes/plugins/memory/fusion/
  plugin.py  — 注册memory toolset，生命周期hooks
  cli.py     — hermes memory recall/remember/status
```

---

## 设计第2部分：需求锁定

- **全替换** Honcho + Preventive + Fusion
- **存储**: Qdrant Cloud (向量) + SQLite (元数据+追踪)
- **Embedding**: Ollama mxbai-embed-large (1024维，零成本)
- **LLM**: 本地提取 + 云端合成
- **部署**: 库 + Hermes 插件适配器
- **功能全覆盖**: 自动蒸馏、去重、多信号排名、辩证合成、冲突解决、反馈闭环、递归追踪

---

## 设计第3部分：架构方案选型

### 方案A：四层管线（垂直分层）

```
对话 → Ingestion(提取+去重) → Storage(Qdrant+SQLite) → Retrieval(重排+合成) → 用户
                                    ↕
                          Intelligence(冲突+反馈+晋升)
```

- 优点：层次清晰，每层职责单一
- 缺点：层间通信多，后台任务和主线程耦合

### 方案B：事件驱动（松耦合）

```
对话结束 → EventBus → [extract_event] → Worker: SVO提取
→ [fact_stored] → Worker: 去重+冲突检测
用户提问 → EventBus → [recall_event] → Worker: 检索+重排+合成
→ [feedback_event] → Worker: 反馈+递归+晋升
```

- 优点：天然后台化，完全解耦，可独立扩展
- 缺点：调试难，事件顺序依赖，过度工程化（个人Agent不需要消息队列）

### 方案C：双管线 + Sidecar（务实融合）⭐ 推荐

```
├── 主线程（用户触发）
│   remember() → SVO提取 → embed → 去重 → 存储
│   recall()  → 向量检索 → 重排 → 合成 → 返回
│
├── 异步事件
│
├── Sidecar（后台守护）
│   digest_worker:   对话结束 → 自动蒸馏 → 入库
│   conflict_worker: 新事实入库 → 冲突检测 → 解决
│   feedback_worker: 用户反馈 → 递归追踪 → 晋升判定
│   cleanup_worker:  定时 → 过期清理 → MEMORY.md写入
│   promote_worker:  高频教训 → 写入L1 MEMORY.md
```

**选C的理由：**
1. Fusion的双管线架构已经过Socrates 6轮审计验证（90分）
2. Hy-Memory的digest本质就是Sidecar模式——对话结束后异步触发
3. Preventive的冲突/反馈/递归本来就是后台任务，不需要实时
4. 比事件驱动简单10倍，比纯管线灵活

---

## 设计第4部分：Sidecar生命周期管理

### 核心原则：不是独立进程

Sidecar是Hermes gateway Python进程内的asyncio后台任务。不启动额外进程，不管理PID，不写systemd service。

### SQLite WAL作为任务队列

所有待处理事件写入SQLite的`pending_tasks`表：

```
remember()完成 → 写入pending_tasks(type=conflict_check, fact_id=xxx)
对话结束     → 写入pending_tasks(type=digest, session_id=xxx)
用户反馈     → 写入pending_tasks(type=feedback, ...)
```

### 后台Worker轮询SQLite

- 不依赖事件总线，不依赖内存队列
- Gateway重启后，Worker自动从SQLite捡起未处理的任务
- 零丢失——SQLite WAL模式本身保证了崩溃安全

### 已验证的模式

Preventive已经验证了这个模式：recurrence_tracker就是SQLite轮询，冲突检测也是。Sidecar本质上是：`asyncio.create_task` + SQLite持久化队列 + 轮询恢复。

---

## 设计第5部分：数据流

### 写入 (remember)

```
用户消息 → SVO提取(本地LLM) → 批量embedding(Ollama) → 语义去重(Qdrant cosine) → 重要度过滤 → Qdrant写入 + SQLite记录
```

### 读取 (recall)

```
查询 → embedding → Qdrant向量检索(3×top_k) → CrossEncoder精排 → 4信号排名(cosine 60% + recency 15% + importance 20% + access 5%) → 辩证合成(云端LLM) → 返回
```

### 后台 (sidecar)

```
对话结束 → SQLite写入pending_task → digest_worker轮询 → 自动蒸馏 → 入库
→ conflict_worker检测 → 解决/报告
→ feedback_worker追踪 → 递归+晋升判定
```

---

## 设计第6部分：LLM分工

### 提取 vs 合成的分离

| 任务 | 模型 | 原因 |
|------|------|------|
| SVO提取 (写入侧) | 本地 Qwen3-30B | 不需要深度推理，延迟不敏感（后台异步） |
| 辩证合成 (读取侧) | 云端 mimo-v2.5-pro | 需要深度推理，用户等待中（延迟敏感） |
| Embedding | 本地 Ollama mxbai-embed-large | 1024维，零成本，~50ms/条 |

### 关键设计决策

提取是**异步后台**的（对话结束后触发），所以本地LLM的5-30秒推理不影响用户体验。合成是**同步等待**的（用户在等回答），所以必须用云端模型保证延迟。

---

## 设计第7部分：从三个系统吸收的能力

### 从 Hy-Memory 吸收

| 能力 | Hy-Memory实现 | Fusion v2实现 |
|------|--------------|---------------|
| SVO提取 | `Extractor` (LLM提取identity+facts+basic_info) | WritePipeline._extract_svo() |
| 自动蒸馏 | `digest()` 异步后台 | digest_worker (Sidecar) |
| Reconcile去重 | `Reconciler` (ADD/SUPERSEDE/UPDATE) | conflict_worker (简化版) |
| 版本化演化链 | `supersedes`/`superseded_by` | SQLite superseded_by字段 |
| 多路检索 | 5种reader + RRF融合 | CrossEncoder + 4信号排名 |

### 从 Preventive Memory 吸收

| 能力 | Preventive实现 | Fusion v2实现 |
|------|---------------|---------------|
| 教训检测 | Rule Engine + DSPy | conflict_worker |
| 复发追踪 | SQLite triggers表 + 时间衰减 | feedback_worker + recurrence_tracker |
| 反馈状态机 | FeedbackCollector (4状态) | feedback_worker |
| 冲突检测 | ConflictDetector (3类型) | conflict_worker |
| Confidence Gate | warn/inject/ignore | 沿用同一门控逻辑 |
| CrossEncoder精排 | bge-reranker-v2-m3 | ReadPipeline (可选增强) |

### 从 Honcho 吸收

| 能力 | Honcho实现 | Fusion v2实现 |
|------|-----------|---------------|
| 辩证推理 | 5级 + 动态调整 + 3-pass | ReadPipeline.synthesize() 5级 |
| Peer Card | honcho_profile | (MEMORY.md L1替代) |
| 语义搜索 | honcho_search | hybrid_search() |

---

## 设计第8部分：配置需求

### 环境变量

| 变量 | 默认值 | 说明 |
|------|--------|------|
| FUSION_QDRANT_URL | localhost:6333 | Qdrant端点 |
| FUSION_QDRANT_API_KEY | (空) | Qdrant Cloud密钥 |
| FUSION_QDRANT_COLLECTION | hermes_fusion | 集合名 |
| FUSION_EMBEDDER_BASE_URL | http://localhost:11434/v1 | Ollama端点 |
| FUSION_EMBEDDER_MODEL | mxbai-embed-large | 嵌入模型 |
| FUSION_LLM_BASE_URL | openrouter | 提取LLM |
| FUSION_LLM_MODEL | hermes-3-405b | 提取模型 |
| FUSION_READER_BASE_URL | localhost:8081 | 合成LLM |
| FUSION_READER_MODEL | local | 合成模型 |
| FUSION_SQLITE_PATH | ~/.hermes/memory.db | SQLite路径 |
| FUSION_DEDUP_THRESHOLD | 0.92 | 去重阈值 |
| FUSION_SCROLL_LIMIT | 500 | Scroll限制 |
| FUSION_SIDECAR_INTERVAL | 30 | Sidecar轮询间隔(秒) |

---

## 设计第9部分：测试需求

### 单元测试
- WritePipeline: SVO提取、embedding、去重、重要度过滤
- ReadPipeline: 向量检索、排名、合成、reasoning_level验证
- MemoryCore: 门面API、资源管理、用户隔离
- Sidecar Workers: 任务队列、轮询、崩溃恢复
- Config: 环境变量、默认值、验证

### 集成测试（需要真实Qdrant）
- 端到端写入+读取
- 多用户隔离
- Sidecar任务执行

### 负面测试
- Qdrant不可用
- LLM超时
- Embedding返回空
- SQLite损坏
- 并发写入

---

## 审计历史

| 轮次 | 分数 | 裁决 | 审计对象 |
|------|------|------|---------|
| Round 1 | 40/100 | FAIL | v1代码（方法论错误） |
| Round 2-4 | 58→98 | PASS | v1代码（P0/P1/P2/P3修复） |
| Round 5 | 78.5/100 | CONDITIONAL | v2代码（发现新P1） |
| Round 6 | 95/100 | PASS | v2代码（全部修复验证） |
| **Round 7** | **待审** | **待定** | **v2设计规范（本文档）** |
