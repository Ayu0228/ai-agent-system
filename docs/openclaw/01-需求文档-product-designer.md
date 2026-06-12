# AI Agent 完整系统搭建 — 需求文档

> **版本:** v1.0  
> **日期:** 2026-06-04  
> **作者:** 产品设计 (product-designer)  
> **状态:** 初稿  
> **审批人:** 阿禹

---

## 目录

1. [项目概述](#1-项目概述)
2. [选型决策矩阵](#2-选型决策矩阵)
3. [自主度定义](#3-自主度定义)
4. [11 个 Agent 角色定义](#4-11-个-agent-角色定义)
5. [成功标准](#5-成功标准)
6. [工具清单](#6-工具清单)
7. [人类监督点](#7-人类监督点)
8. [边界条件](#8-边界条件)
9. [非功能需求](#9-非功能需求)
10. [六大核心组件需求分析](#10-六大核心组件需求分析)
11. [总结与优先级](#11-总结与优先级)

---

## 1. 项目概述

### 1.1 一句话定义

搭建基于 OpenClaw 框架的 11 个 AI Agent 协作系统，通过 6 个核心组件实现 Agent 间的记忆共享、工作流编排、评估反馈、人机协作、安全可观测和经验学习。

**为什么做：** 单个 Agent 能力有限，无法处理跨领域、多步骤的复杂任务。多 Agent 协作系统可以将复杂问题分解为专业子任务，由专门的 Agent 并行或串行处理，提升整体任务完成率和质量。

### 1.2 系统架构总览

```
┌─────────────────────────────────────────────────────────────┐
│                     飞书 (用户交互层)                         │
├─────────────────────────────────────────────────────────────┤
│                    main (陈陈/中枢调度)                       │
│              Orchestrator-Workers 模式                       │
├────────┬────────┬────────┬────────┬────────┬────────────────┤
│research│tech-dev│copywri │data-   │invest- │  其他 5 个     │
│  er    │        │  ter   │analyst │ment    │  专业 Agent    │
│        │        │        │        │analyst │                │
├────────┴────────┴────────┴────────┴────────┴────────────────┤
│                   6 个核心组件                                │
│  共享记忆 │ 工作流编排 │ 评估闭环 │ 审批 │ 安全 │ 经验学习   │
├─────────────────────────────────────────────────────────────┤
│              ChromaDB + SQLite (存储层)                       │
└─────────────────────────────────────────────────────────────┘
```

### 1.3 技术栈

| 层级 | 技术选型 | 说明 |
|------|---------|------|
| 框架 | OpenClaw | Agent 编排和通信框架 |
| 通信 | 飞书频道 + sessions_send | 用户交互 + Agent 间通信 |
| 主力模型 | MIMO v2.5 Pro | 复杂推理、内容生成 |
| 降级模型 | DeepSeek V4 Pro | 简单任务、成本控制 |
| 向量存储 | ChromaDB | 长期记忆、语义检索 |
| 结构化存储 | SQLite | 工作记忆、活动日志、经验库 |
| 运行环境 | M1 Mac | 本地部署 |

---

## 2. 选型决策矩阵

> "For many applications, optimizing single LLM calls with retrieval and in-context examples is usually enough."
> [Source: Anthropic, "Building Effective AI Agents", 2025]

### 2.1 决策框架

对每个组件，按以下顺序判断：

```
单次 LLM 调用 + RAG 够不够？
  → 够 → 用单次调用
  → 不够 → 预定义工作流（Prompt Chaining / Routing）够不够？
    → 够 → 用工作流
    → 不够 → 真的需要 LLM 动态决策？
      → 是 → 用 Agent 模式
      → 否 → 退回工作流
```

### 2.2 各组件选型结果

| 组件 | 选型决策 | 架构模式 | 理由 |
|------|---------|---------|------|
| 共享记忆系统 | **工作流** | Prompt Chaining | 路径固定：写入→向量化→索引→检索→注入。无需 LLM 决策"要不要记忆"，规则引擎即可判断 |
| 工作流编排引擎 | **Agent**（仅 main） | Orchestrator-Workers | main 需要根据任务内容动态决定派给哪个 Agent、以什么顺序——这是 LLM 动态决策 |
| 评估反馈闭环 | **评估器-优化器** | Evaluator-Optimizer | LLM 做质量评分（评估器），结果写入记忆供改进（优化器），但最终优化动作由人工确认 |
| 人机协作审批 | **工作流** | Routing | 根据操作风险等级（L0-L3）路由到不同审批路径，路由规则是确定性的 |
| 安全可观测性 | **工作流** | 纯确定性逻辑 | 日志采集、成本统计、告警触发全部是规则引擎，零 AI 参与 |
| 经验学习闭环 | **评估器-优化器** | Evaluator-Optimizer | LLM 从执行轨迹中提取经验（评估器），人工确认后写入记忆库（优化器） |

### 2.3 不用 Agent 的组件为什么不用

| 组件 | 如果误用 Agent 会怎样 |
|------|---------------------|
| 共享记忆 | 每次记忆操作都走 LLM 决策，延迟从 <500ms 膨胀到 3-5s，成本增加 10 倍，且记忆一致性更难保证 |
| 审批路由 | LLM 判断风险等级可能出错（幻觉），确定性规则引擎 100% 准确且零延迟 |
| 安全可观测 | 让 AI 决定"记不记日志"是安全灾难。日志必须全量采集，规则驱动 |

---

## 3. 自主度定义

> Anthropic 区分 Workflows（预定义代码路径编排 LLM）和 Agents（LLM 自主决定流程控制）。
> [Source: Anthropic, "Building Effective AI Agents", 2025]

### 3.1 三级自主度模型

| 等级 | 名称 | 定义 | 决策主体 |
|------|------|------|---------|
| L1 | Router | 输入分类→固定路由→执行 | 规则引擎 |
| L2 | State Machine | 预定义状态转移图，LLM 执行每步但不改变流程 | 状态机 + LLM |
| L3 | Autonomous Agent | LLM 动态决定下一步做什么、调用什么工具 | LLM |

### 3.2 各 Agent 自主等级分配

| Agent | 角色 | 自主等级 | 说明 |
|-------|------|---------|------|
| **main（陈陈）** | 中枢调度 | **L3 - Autonomous** | 唯一全自主 Agent。接收阿禹指令后动态决定任务拆解、Agent 分配、执行顺序 |
| **researcher** | 调研采集 | L2 - State Machine | 采集→提取→入库 是预定义流水线。LLM 负责理解和提取，但不决定"要不要采集" |
| **tech-dev** | 开发 | L2 - State Machine | 编码任务走固定流程（需求理解→方案设计→编码→测试），高风险操作需审批拦截 |
| **copywriter** | 文案 | L2 - State Machine | 内容生成→审校→发布，流程固定 |
| **data-analyst** | 数据分析 | L2 - State Machine | 数据获取→清洗→分析→可视化，流程固定 |
| **investment-analyst** | 投资分析 | L2 - State Machine | 数据采集→指标计算→报告生成，流程固定 |
| **ops-monitor** | 运维监控 | **L1 - Router** | 监控异常→分类→路由到对应 Agent 处理。不做决策，只做分类转发 |
| **product-designer** | 产品设计 | L2 - State Machine | 需求分析→竞品调研→方案设计→PRD 生成，流程固定 |
| **script-editor** | 脚本编辑 | L2 - State Machine | 脚本编写→审核→修改，流程固定 |
| **visual-designer** | 视觉设计 | L2 - State Machine | 需求理解→素材生成→审核→修改，流程固定 |
| **content-strategist** | 内容策略 | L2 - State Machine | 策略制定→内容规划→效果分析，流程固定 |

### 3.3 自主等级的硬约束

1. **任何 Agent 不能提升自己的自主等级。** 例如 tech-dev 不能从 L2 自己升级到 L3。这个约束通过 AGENTS.md 中的 NEVER 规则强制执行。
2. **main 是唯一 L3 Agent。** 其他 Agent 需要动态决策时，必须向 main 请求路由。
3. **自主等级降级是允许的。** 当 main 不可用时，各 Agent 回退到预定义的默认行为（L2 模式）。

---

## 4. 11 个 Agent 角色定义

### 4.1 角色总览

| Agent | 一句话职责 | 输入来源 | 输出去向 | 核心工具 |
|-------|----------|---------|---------|---------|
| main（陈陈） | 接收阿禹指令，拆解任务，分配给专业 Agent | 阿禹（飞书） | 各专业 Agent | sessions_send, memory_search |
| researcher | 搜索和整理信息 | main 分配 | main / 共享记忆 | web_search, web_fetch, shared_memory_write |
| tech-dev | 编写和调试代码 | main 分配 | main / 文件系统 | read, write, edit, exec |
| copywriter | 撰写和润色文案 | main 分配 | main / 共享记忆 | shared_memory_write |
| data-analyst | 采集和分析数据 | main 分配 | main / 共享记忆 | web_search, exec, shared_memory_write |
| investment-analyst | 投资研究和分析 | main 分配 | main / 共享记忆 | web_search, web_fetch, shared_memory_write |
| ops-monitor | 监控系统状态，分类异常 | 系统事件 | main / 飞书告警 | health_check, alert_send |
| product-designer | 产品设计和 PRD 撰写 | main 分配 | main / 共享记忆 | web_search, shared_memory_write |
| script-editor | 脚本编写和校对 | main 分配 | main / 文件系统 | read, write, edit |
| visual-designer | 视觉方案设计 | main 分配 | main / 共享记忆 | shared_memory_write |
| content-strategist | 内容策略规划 | main 分配 | main / 共享记忆 | web_search, shared_memory_write |

### 4.2 Agent 间通信规则

| 规则 | 说明 |
|------|------|
| 所有任务分配通过 main | 专业 Agent 不能直接给其他专业 Agent 分配任务 |
| Agent 间可通过共享记忆间接协作 | researcher 写入的记忆，tech-dev 可以检索到 |
| 紧急情况可直接通信 | ops-monitor 发现系统异常可直接通知相关 Agent |
| 所有通信有日志记录 | sessions_send 的每条消息都记录到活动日志 |

---

## 5. 成功标准

> "Agents need clear success criteria."
> [Source: Anthropic, "Building Effective AI Agents", 2025]

### 5.1 系统级指标

| 指标 | Phase 1 目标 | Phase 2 目标 | 测量方式 |
|------|-------------|-------------|---------|
| 跨 Agent 任务完成率 | > 75% | > 90% | 端到端任务跟踪 |
| Agent 单次任务 P95 延迟 | < 30s | < 15s | 工具调用计时（不含人工审批等待） |
| 每日总 token 消耗 | < 500K | < 1M | API 用量统计 |
| 单任务成本上限 | $0.50 | $0.30 | 按任务维度 token 归因 |
| 系统可用率 | > 99% | > 99.5% | 健康检查探针 |

### 5.2 各组件成功标准

| 组件 | 指标 | 目标 | 验收标准 |
|------|------|------|---------|
| 共享记忆 | 记忆写入延迟 | P95 < 500ms | 上线后 1 周，11 个 Agent 均能成功写入和检索 |
| 共享记忆 | 检索召回率 | > 80% | 上线后 2 周，人工标注 50 条查询测召回率 |
| 工作流编排 | 执行成功率 | > 90% | 上线后 2 周，daily-collect 和 content-production 流水线稳定运行 |
| 工作流编排 | Agent 路由准确率 | > 85% | 上线后 1 个月，人工标注 30 个任务验证 |
| 评估闭环 | 自动评估与人工一致性 | > 80% | 上线后 2 周，每周抽查 20 条对比 |
| 评估闭环 | 幻觉检测漏报率 | < 10% | Adversarial 测试集验证 |
| 人机审批 | L2/L3 拦截率 | 100% | 上线时即达标，渗透测试确认 |
| 人机审批 | 审批卡片送达延迟 | < 5s | 上线后 1 周 |
| 安全可观测 | Prompt 注入拦截率 | > 95% | 上线时，50 条注入测试集 |
| 安全可观测 | 活动日志完整率 | 100% | 上线后 1 周 |
| 经验学习 | 经验提取准确率 | > 75% | 上线后 2 周，人工审核 30 条 |
| 经验学习 | 注入后任务改进率 | > 10% | 上线后 1 个月，对比注入前后 |

---

## 6. 工具清单

> "It is crucial to design toolsets and their documentation clearly and thoughtfully."
> [Source: Anthropic, "Building Effective AI Agents", 2025]

### 6.1 现有工具（OpenClaw 框架内置）

| 工具 | 调用方 | 输入 Schema | 输出 Schema | 安全等级 |
|------|--------|------------|------------|---------|
| `read` | 全部 Agent | `{path: string, offset?: int, limit?: int}` | `{content: string}` | L0 |
| `write` | 全部 Agent | `{path: string, content: string}` | `{success: bool}` | L1 |
| `edit` | 全部 Agent | `{path: string, edits: [{oldText, newText}]}` | `{success: bool}` | L1 |
| `exec` | 全部 Agent | `{command: string, workdir?: string, timeout?: int}` | `{stdout, stderr, exit_code}` | L2 |
| `web_search` | researcher 等 | `{query: string, count?: int}` | `{results: [{title, url, snippet}]}` | L0 |
| `web_fetch` | researcher 等 | `{url: string, maxChars?: int}` | `{content: string}` | L0 |
| `sessions_send` | 全部 Agent | `{to: string, message: string}` | `{delivered: bool}` | L1 |
| `memory_search` | 全部 Agent | `{query: string, maxResults?: int}` | `{results: [{snippet, score, path}]}` | L0 |
| `memory_get` | 全部 Agent | `{path: string, from?: int, lines?: int}` | `{content: string}` | L0 |

### 6.2 需要新增的工具

#### 共享记忆工具

| 工具 | 调用方 | 输入 Schema | 输出 Schema | 安全等级 |
|------|--------|------------|------------|---------|
| `shared_memory_write` | 全部 Agent | `{content: string, tags: [string], agent_id: string, memory_type: "episodic"\|"semantic"\|"procedural", ttl_days?: int}` | `{id: string, created_at: datetime}` | L1 |
| `shared_memory_search` | 全部 Agent | `{query: string, agent_id?: string, tags?: [string], top_k?: int, min_score?: float}` | `{results: [{id, text, score, source_agent, tags, created_at}]}` | L0 |
| `shared_memory_delete` | 全部 Agent | `{id: string, agent_id: string}` | `{success: bool}` | L2 |
| `shared_memory_update` | 全部 Agent | `{id: string, content?: string, tags?: [string], ttl_days?: int}` | `{success: bool, updated_at: datetime}` | L1 |

#### 经验管理工具

| 工具 | 调用方 | 输入 Schema | 输出 Schema | 安全等级 |
|------|--------|------------|------------|---------|
| `experience_record` | 全部 Agent | `{type: "success"\|"failure"\|"reflection", task: string, agent_id: string, symptom: string, solution: string, tags: [string]}` | `{id: string}` | L1 |
| `experience_retrieve` | 全部 Agent | `{task_type: string, agent_id?: string, top_k?: int}` | `{experiences: [{symptom, solution, date, type, success_rate}]}` | L0 |
| `experience_verify` | main | `{id: string, verified: bool, reason: string}` | `{success: bool}` | L1 |

#### 审批工具

| 工具 | 调用方 | 输入 Schema | 输出 Schema | 安全等级 |
|------|--------|------------|------------|---------|
| `approval_request` | 全部 Agent | `{action: string, risk_level: "L0"\|"L1"\|"L2"\|"L3", summary: string, agent_id: string, context?: string}` | `{approved: bool, reason?: string, approver?: string}` | - |
| `approval_check` | 全部 Agent | `{request_id: string}` | `{status: "pending"\|"approved"\|"rejected"\|"timeout", reason?: string}` | L0 |

#### 评估工具

| 工具 | 调用方 | 输入 Schema | 输出 Schema | 安全等级 |
|------|--------|------------|------------|---------|
| `eval_scorecard` | 全部 Agent | `{task_id: string, agent_id: string, output: string, reference?: string}` | `{scores: {accuracy, completeness, safety, latency}, overall: float, issues: [string]}` | L0 |
| `eval_trajectory` | main | `{task_id: string, steps: [{agent, action, result, duration_ms}]}` | `{trajectory_score: float, step_scores: [{step_id, score, issues}], hidden_failures: [string]}` | L0 |

#### 监控工具

| 工具 | 调用方 | 输入 Schema | 输出 Schema | 安全等级 |
|------|--------|------------|------------|---------|
| `cost_check` | 全部 Agent | `{agent_id: string}` | `{today_tokens: int, today_cost_usd: float, budget_remaining: float}` | L0 |
| `health_check` | ops-monitor | `{component?: string}` | `{status: "healthy"\|"degraded"\|"down", details: dict}` | L0 |
| `alert_send` | ops-monitor | `{level: "info"\|"warn"\|"critical", message: string, channel: string}` | `{sent: bool}` | L1 |

### 6.3 工具设计原则

1. **单一职责**：每个工具只做一件事。`shared_memory_write` 不负责索引优化，那是内部实现。
2. **幂等性**：相同参数的重复调用返回相同结果（写入类工具用 upsert 语义）。
3. **明确错误码**：每个工具定义 3-5 个错误码，Agent 可据此决定重试/降级/报告。
4. **最小上下文传递**：工具调用只传必要参数，不传完整对话历史。
5. **安全等级标注**：L0 可自主执行，L1 事后通知，L2 实时审批，L3 确认码审批。

---

## 7. 人类监督点

> "Deployers should monitor agents to ensure they remain within the bounds of their intended purpose."
> [Source: OpenAI, "Practices for Governing Agentic AI Systems", 2024]

### 7.1 风险分级与审批模式

| 风险等级 | 名称 | 审批模式 | 触发条件 | 典型操作 |
|---------|------|---------|---------|---------|
| **L0** | 只读 | 全自主 | 搜索、读文件、查记忆、健康检查 | `web_search`, `memory_search`, `read`, `health_check` |
| **L1** | 安全写入 | Human-on-the-Loop | 写文件、发消息、写记忆 | `write`, `edit`, `shared_memory_write`, `experience_record` |
| **L2** | 风险操作 | Human-in-the-Loop | 修改配置、调付费 API、发全员消息、删除记忆 | `shared_memory_delete`, `exec`(非危险命令) |
| **L3** | 危险操作 | HITL + 确认码 | `rm -rf`、DROP TABLE、生产环境部署、修改 AGENTS.md | `exec` 中包含危险命令 |

### 7.2 审批交互流程

```
Agent 发起操作
  → 风险等级判定（确定性规则）
    → L0: 直接执行 + 事后日志
    → L1: 执行 + 飞书事后通知
    → L2: 飞书审批卡片 → 阿禹确认 → 执行/拒绝
    → L3: 飞书审批卡片 + 确认码 → 阿禹输入确认码 → 执行/拒绝
```

### 7.3 各 Agent 监督点矩阵

#### main（陈陈）

| 操作 | 监督级别 | 理由 |
|------|---------|------|
| 路由决策 | L0 | 内部推理，无副作用 |
| 任务分配 | L1 | 通知阿禹任务已分配 |
| 修改工作流定义 | L2 | 影响系统行为 |
| 修改 AGENTS.md | L3 | 影响系统安全规则 |

#### researcher（调研）

| 操作 | 监督级别 | 理由 |
|------|---------|------|
| 执行 web_search | L0 | 只读操作 |
| 执行 web_fetch | L0 | 只读操作 |
| 输出调研报告 | L1 | 通知用户报告已生成 |
| 引用外部数据做结论 | L1 | 需标注置信度 |

#### tech-dev（开发）

| 操作 | 监督级别 | 理由 |
|------|---------|------|
| 读取代码文件 | L0 | 只读操作 |
| 修改代码文件 | L2 | 代码变更需要确认 |
| 执行 shell 命令 | L2 | 可能有副作用 |
| 安装依赖包 | L2 | 可能引入安全风险 |
| 部署/发布 | L3 | 生产操作需正式审批 |
| 删除文件 | L3 | 不可逆操作 |

#### copywriter（文案）

| 操作 | 监督级别 | 理由 |
|------|---------|------|
| 生成文案草稿 | L0 | 内部生成，无副作用 |
| 润色和修改 | L0 | 内部生成 |
| 输出最终文案 | L1 | 通知用户 |
| 对外发布内容 | L3 | 涉及对外输出 |

#### data-analyst（数据分析）

| 操作 | 监督级别 | 理由 |
|------|---------|------|
| 读取数据文件 | L0 | 只读操作 |
| 执行数据查询 | L0 | 只读操作 |
| 生成报表 | L1 | 通知用户 |
| 修改数据源配置 | L2 | 影响数据采集 |

#### investment-analyst（投资分析）

| 操作 | 监督级别 | 理由 |
|------|---------|------|
| 搜索市场数据 | L0 | 只读操作 |
| 生成分析报告 | L1 | 通知用户 |
| 引用财务数据做结论 | L1 | 需标注置信度和来源 |

#### ops-monitor（运维监控）

| 操作 | 监督级别 | 理由 |
|------|---------|------|
| 健康检查 | L0 | 只读操作 |
| 发送告警通知 | L1 | 通知用户 |
| 修改监控阈值 | L2 | 影响告警行为 |
| 重启服务 | L3 | 影响系统可用性 |

#### product-designer（产品设计）

| 操作 | 监督级别 | 理由 |
|------|---------|------|
| 竞品调研 | L0 | 只读操作 |
| 生成 PRD | L1 | 通知用户 |
| 修改需求文档 | L1 | 版本化管理 |

#### script-editor（脚本编辑）

| 操作 | 监督级别 | 理由 |
|------|---------|------|
| 读取脚本 | L0 | 只读操作 |
| 编写/修改脚本 | L1 | 通知用户 |
| 执行脚本 | L2 | 可能有副作用 |

#### visual-designer（视觉设计）

| 操作 | 监督级别 | 理由 |
|------|---------|------|
| 生成设计方案 | L0 | 内部生成 |
| 修改设计文件 | L1 | 通知用户 |
| 对外发布设计 | L3 | 涉及对外输出 |

#### content-strategist（内容策略）

| 操作 | 监督级别 | 理由 |
|------|---------|------|
| 搜索参考资料 | L0 | 只读操作 |
| 生成策略文档 | L1 | 通知用户 |
| 修改内容计划 | L1 | 版本化管理 |

### 7.4 特殊监督规则

| 规则 | 说明 |
|------|------|
| **经验验证** | 新经验写入前必须经阿禹审阅确认，防止错误经验污染知识库 |
| **Agent 自主升级** | 永远禁止。任何 Agent 试图修改自主等级 → 直接阻断 + 飞书告警 |
| **批量操作** | 单次操作影响 > 10 个文件/记录 → 自动升级到 L2 |
| **跨 Agent 转发审批** | Agent A 的 L2 操作不能通过 Agent B 绕过审批 |

---

## 8. 边界条件

> "Deploy agentic systems with extensive testing in sandboxed environments, along with the appropriate guardrails."
> [Source: Anthropic, "Building Effective AI Agents", 2025]

### 8.1 Agent 绝对不能做的事（NEVER 清单）

| # | 禁止行为 | 防御机制 |
|---|---------|---------|
| 1 | 修改自己的 AGENTS.md 或系统 Prompt | 文件路径白名单 + 写操作审批 |
| 2 | 提升自己的权限等级或自主等级 | AGENTS.md NEVER 规则 + 运行时检查 |
| 3 | 删除其他 Agent 的记忆或经验数据 | `shared_memory_delete` 的 agent_id 校验 |
| 4 | 调用未在工具清单中列出的外部 API | 工具白名单机制 |
| 5 | 向外部地址发送 API Key、token、用户数据 | 输出内容安全扫描 |
| 6 | 执行 `rm -rf`、`DROP TABLE`、`curl -X DELETE`（未经审批） | exec 命令危险模式检测 + L3 审批 |
| 7 | 伪造成其他 Agent 的身份发送消息 | 消息签名 + agent_id 不可伪造 |
| 8 | 自我复制或创建自身副本 | 系统层面硬约束 |
| 9 | 在输出中泄露 system prompt 内容 | 输出过滤管线 |
| 10 | 绕过审批机制执行高风险操作 | 审批中间件不可跳过 |

### 8.2 输入安全红线

| 攻击类型 | 防御措施 |
|---------|---------|
| 直接 Prompt 注入 | 输入安全扫描：检测注入指令模式（`ignore previous instructions`、角色覆盖等） |
| 间接 Prompt 注入 | RAG 检索结果和工具返回内容标记为"不可信"，处理前安全过滤 |
| 跨 Agent 注入 | Agent 间消息传递添加安全检查，防止 Agent A 通过消息操控 Agent B |
| 编码绕过 | Base64/Unicode 解码后再做安全检查 |
| 多轮渐进越狱 | 跨轮次行为分析：监控多轮对话中的安全边界侵蚀 |

### 8.3 操作安全红线

| 场景 | 约束 |
|------|------|
| 文件系统操作 | 路径白名单，禁止访问 `~/.ssh/`、`~/.openclaw/agents/*/AGENTS.md` 等敏感路径 |
| Shell 命令 | 危险命令模式检测（rm -rf、chmod 777、curl pipe sh 等） |
| 网络请求 | 外部请求域名白名单（Phase 1），出站流量监控 |
| Token 消耗 | 单任务 $0.50 上限，超过自动终止 |
| 执行步骤 | 单任务最大 50 步，超过自动终止并报告 |

### 8.4 降级条件

| 触发条件 | 降级行为 |
|---------|---------|
| 主模型不可用 | 自动切换到备用模型 DeepSeek V4 Pro |
| ChromaDB 不可用 | 降级到 SQLite 关键词检索 |
| 飞书不可用 | 审批请求暂存队列，恢复后批量处理 |
| main（编排器）不可用 | 各 Agent 回退到预定义默认行为 |
| 单日 token 超限 | 非紧急任务暂停，仅处理 L2/L3 审批 |

---

## 9. 非功能需求

> "Latency is the second biggest production obstacle, with 20% of teams citing it."
> [Source: LangChain, "State of Agent Engineering", 2025]

### 9.1 延迟要求

| 场景 | P50 目标 | P95 目标 | P99 目标 |
|------|---------|---------|---------|
| Agent 单次推理（不含工具） | < 3s | < 8s | < 15s |
| 工具调用（本地） | < 200ms | < 500ms | < 1s |
| 工具调用（外部 API） | < 1s | < 3s | < 5s |
| 记忆检索 | < 300ms | < 1s | < 2s |
| 审批卡片送达 | < 2s | < 5s | < 10s |
| 端到端任务（不含人工等待） | < 15s | < 30s | < 60s |

### 9.2 并发与扩展

| 维度 | Phase 1 | Phase 2 |
|------|---------|---------|
| 同时活跃 Agent 数 | 3 | 5 |
| 并发任务数 | 5 | 10 |
| 记忆库条目上限 | 10K | 50K |
| 日志保留周期 | 30 天 | 90 天 |

### 9.3 数据隐私

| 数据类型 | 存储位置 | 传输加密 | 访问控制 |
|---------|---------|---------|---------|
| 记忆数据（ChromaDB + SQLite） | 本地 | 不传输（本地文件） | Agent 级别 ACL |
| 日志数据（SQLite） | 本地 | 不传输 | ops-monitor 只读 |
| Agent 间通信 | 内存 | sessions_send 内部通道 | 发送方/接收方验证 |
| 飞书消息 | 飞书服务器 | HTTPS/TLS | 飞书权限体系 |
| API Key | 环境变量 | 不存储在代码/文档中 | 进程级别隔离 |

**核心原则：记忆数据仅存本地（ChromaDB + SQLite），不上传到第三方。**

### 9.4 合规要求

| 要求 | 实施方式 |
|------|---------|
| API Key 不硬编码 | 环境变量注入，`.env` 文件不进版本控制 |
| 敏感数据不进日志 | 日志脱敏管线：PII 检测 + 自动替换 |
| 操作可追溯 | 100% 活动日志覆盖，保留 30 天 |
| 数据可删除 | 记忆条目支持按 ID 删除 |

### 9.5 可维护性

| 维度 | 要求 |
|------|------|
| 组件代码行数 | < 500 行/组件（Karpathy 原则） |
| 组件间接口 | 通过明确定义的工具 Schema 通信，无隐式依赖 |
| 配置管理 | YAML 配置文件，版本化管理 |
| 回滚能力 | Prompt/工具/模型配置版本化，支持快速回滚 |

### 9.6 可恢复性

| 故障场景 | 恢复策略 |
|---------|----------|
| 单 Agent 崩溃 | 不影响其他 Agent（无单点故障） |
| ChromaDB 不可用 | 降级到 SQLite 关键词检索 |
| 主模型不可用 | 自动降级到备用模型（MIMO v2.5 Pro → DeepSeek V4 Pro） |
| 飞书不可用 | 审批请求暂存队列，恢复后批量处理 |
| main（编排器）不可用 | 各 Agent 回退到预定义默认行为 |

---

## 10. 六大核心组件需求分析

### 10.1 组件一：共享记忆系统

**做的是什么：** 建立跨 Agent 的统一记忆层，让 11 个 Agent 能共享知识、避免重复工作。

**为什么做：** 当前各 Agent 独立运行，researcher 调研的结论 product-designer 不知道，tech-dev 遇到的问题 copywriter 不了解。记忆断裂导致重复劳动和信息不一致。

#### 10.1.1 记忆类型定义

| 记忆类型 | 用途 | 存储介质 | 生命周期 |
|---------|------|---------|----------|
| **短期记忆** | 当前任务的上下文窗口 | 内存（OpenClaw 自动管理） | 会话级别 |
| **工作记忆** | 当前任务的中间状态和决策 | SQLite（结构化） | 任务级别，完成后归档 |
| **长期记忆-情景记忆** | 事件和经历（"发生了什么"） | ChromaDB（向量） | 永久，可设 TTL |
| **长期记忆-语义记忆** | 知识和事实（"是什么"） | ChromaDB（向量） | 永久 |
| **长期记忆-程序记忆** | 技能和经验（"怎么做"） | ChromaDB（向量） | 永久，需验证更新 |

> 记忆分类参考 Lilian Weng 的三类记忆模型：短期记忆（上下文学习）、长期记忆（外部向量库）、工作记忆（任务状态）。
> [Source: Lilian Weng, "LLM Powered Autonomous Agents", 2023]

#### 10.1.2 记忆写入规则

| 触发条件 | 写入类型 | 示例 |
|---------|----------|------|
| 关键决策点 | 语义记忆 | "选择了方案 A 而非方案 B，因为用户场景是..." |
| 用户显式要求记住 | 语义记忆 | "记住这个 API endpoint" |
| 重要事实发现 | 语义记忆 | "竞品 X 的定价是 $99/月" |
| 任务完成 | 情景记忆 | "完成了任务 X，用了方法 Y，耗时 Z" |
| 经验提取 | 程序记忆 | "遇到错误 E 时，解决方案是 S" |

#### 10.1.3 跨 Agent 记忆访问控制

| 记忆类型 | 读权限 | 写权限 |
|---------|--------|--------|
| 通用知识 | 全部 Agent | 全部 Agent |
| Agent 专属知识 | 全部 Agent | 创建者 Agent |
| 用户隐私数据 | 仅 main | 仅 main |
| 经验数据 | 全部 Agent | 需人工确认 |

#### 10.1.4 记忆检索策略

采用多路检索合并（RRF）：

1. **语义检索**：ChromaDB 向量相似度搜索（权重 0.6）
2. **关键词检索**：SQLite FTS 全文搜索（权重 0.3）
3. **时间加权**：优先返回近期记忆（权重 0.1）

合并后 reranking，返回 top_k 结果。

#### 10.1.5 工作量、价值、风险评估

| 维度 | 评估 |
|------|------|
| **工作量** | 中等。ChromaDB + SQLite 已有基础，需新增共享层和访问控制。估计 3-5 天 |
| **用户价值** | 高。解决跨 Agent 信息断裂问题，是其他 5 个组件的基础 |
| **风险** | 中。并发写入一致性、记忆检索精度是主要风险点 |
| **依赖** | 无外部依赖。ChromaDB 和 SQLite 已部署 |

#### 10.1.6 验收标准

- **上线后 1 周**，11 个 Agent 均能成功写入和检索共享记忆
- **上线后 2 周**，记忆检索召回率 > 80%，精确率 > 70%
- **上线后 1 个月**，跨 Agent 任务的信息一致性问题减少 50%

---

### 10.2 组件二：工作流编排引擎

**做的是什么：** 让 main（陈陈）作为编排器，按 YAML 定义的工作流自动分配任务给各 Agent，支持顺序链、并行、条件路由三种模式，实现多 Agent 协作自动化。

**为什么做：** 当前 Agent 间协作靠人工串联，重复性任务（每日数据采集、内容生产流水线）需要自动化。手动串联 11 个 Agent 是不可持续的。

#### 10.2.1 工作流定义格式

采用 YAML 声明式定义，支持变量引用 `$prev.output` 传递上一步输出：

```yaml
workflow:
  name: daily-content-pipeline
  version: "1.0"
  trigger:
    type: schedule
    cron: "0 9 * * *"  # 每天早 9 点
  steps:
    - id: research
      agent: researcher
      type: task
      task:
        prompt: "搜索 AI Agent 领域最新动态"
      timeout: 120
    - id: write_draft
      agent: copywriter
      type: task
      depends_on: [research]
      input:
        research_data: "$research.output"
      task:
        prompt: "基于调研结果撰写内容初稿"
      timeout: 180
    - id: review
      agent: main
      type: task
      depends_on: [write_draft]
      input:
        draft: "$write_draft.output"
      task:
        prompt: "审核内容初稿，判断是否需要修改"
      timeout: 60
```

#### 10.2.2 编排模式

| 模式 | YAML 配置 | 适用场景 | 示例 |
|------|----------|---------|------|
| **顺序链** | `depends_on: [step_a]` | 步骤间有依赖 | 调研→写作→审核 |
| **并行** | `depends_on: []` 或无 depends_on | 独立子任务 | researcher + data-analyst 同时采集 |
| **条件路由** | `condition: $prev.output.status == 'approved'` | 分支判断 | 审核通过→发布，不通过→修改 |
| **循环** | `loop: {max_iterations: 3, until: condition}` | 迭代改进 | 评估不达标→修改→重新评估 |

#### 10.2.3 Agent 路由策略

main（陈陈）作为 orchestrator 的路由决策依据：

| 任务关键词/意图 | 路由目标 | 理由 |
|----------------|---------|------|
| "搜索/查找/调研" | researcher | 信息采集专业 |
| "写代码/开发/修复" | tech-dev | 技术实现专业 |
| "写文章/文案/内容" | copywriter | 内容创作专业 |
| "分析数据/报表" | data-analyst | 数据分析专业 |
| "投资/财务/估值" | investment-analyst | 投资分析专业 |
| "监控/告警/运维" | ops-monitor | 运维监控专业 |
| "设计/PRD/需求" | product-designer | 产品设计专业 |
| "编辑/校对/脚本" | script-editor | 编辑校对专业 |
| "视觉/UI/图片" | visual-designer | 视觉设计专业 |
| "策略/规划/方向" | content-strategist | 策略规划专业 |

路由决策通过 LLM 理解意图后匹配，而非关键词硬匹配，以处理自然语言表述的多样性。

#### 10.2.4 超时和重试策略

| 故障场景 | 策略 | 配置 |
|---------|------|------|
| Agent 响应超时 | 重试 1 次，超时后跳过并标记失败 | timeout: 按步骤配置（60-300s） |
| Agent 返回错误 | 重试 2 次（指数退避：5s/15s/45s） | max_retries: 2 |
| 工具调用失败 | 重试 1 次，失败后降级到备选工具 | 按工具配置 |
| 整个工作流超时 | 终止并通知阿禹 | workflow_timeout: 1800s |
| main 路由失败 | 降级到规则引擎路由 | 降级策略内置 |

#### 10.2.5 工作量、价值、风险评估

| 维度 | 评估 |
|------|------|
| **工作量** | 高。YAML 解析器 + 状态管理 + 超时机制 + 变量传递，估计 5-7 天 |
| **用户价值** | 高。将人工串联 Agent 变为自动化流水线，节省每日 1-2 小时人工操作 |
| **风险** | 中。状态管理复杂度、Agent 间死锁、变量传递错误是主要风险 |
| **依赖** | 依赖共享记忆系统（Agent 间通过记忆传递上下文） |

#### 10.2.6 验收标准

- **上线后 1 周**，daily-collect 和 content-production 两条流水线能自动执行
- **上线后 2 周**，工作流执行成功率 > 90%
- **上线后 1 个月**，至少 3 条工作流流水线稳定运行，人工干预率 < 10%

---

### 10.3 组件三：评估反馈闭环

**做的是什么：** 建立离线 + 在线双轨评估框架，用 LLM-as-Judge 自动评估 Agent 输出质量，生产失败自动回流到测试集形成数据飞轮。

**为什么做：** [Source: LangChain, "State of Agent Engineering", 2025] 指出质量是 32% 团队的首要生产障碍。没有评估闭环，质量改进就是靠感觉。

#### 10.3.1 三维评估指标体系

| 维度 | 评估内容 | 评估方式 | 权重 |
|------|---------|----------|------|
| **Grounding / Context Use** | 输出是否基于检索到的上下文，是否包含幻觉 | LLM-as-Judge + 引用验证 | 40% |
| **User Experience Quality** | 输出是否满足用户需求，格式是否正确 | LLM-as-Judge + 格式校验 | 40% |
| **Security / Safety** | 输出是否包含敏感信息、是否被注入攻击影响 | 规则检测 + LLM-as-Judge | 20% |

> [Source: LangChain, "LLM Evaluation: Trajectories vs Outputs", 2025]

#### 10.3.2 测试数据集分类

| 类别 | 数量目标 | 覆盖内容 |
|------|---------|----------|
| Happy Path | 每 Agent 5 条，共 55 条 | 正常输入→期望正常输出 |
| Edge Cases | 每 Agent 3 条，共 33 条 | 空输入、超长输入、模糊指令、工具失败 |
| Adversarial | 共 20 条 | Prompt 注入、越狱尝试、越权操作 |

#### 10.3.3 轨迹评估

**核心原则：** 评估整个执行路径，不只看最终输出。

[Source: LangChain, "LLM Evaluation: Trajectories vs Outputs", 2025]: "Correct final answers can hide broken reasoning"

| 检查点 | 评估内容 | 失败判定 |
|--------|---------|----------|
| 工具选择 | 是否选择了正确的工具 | 选错工具但最终结果正确 → 标记"隐藏失败" |
| 参数填充 | 工具参数是否正确 | 参数错误 → 标记 |
| 推理链 | 中间推理步骤是否逻辑通顺 | 循环论证/矛盾 → 标记 |
| 错误处理 | 遇到错误时是否正确处理 | 忽略错误继续执行 → 标记 |
| 最终输出 | 结果是否正确 | 输出错误 → 标记 |

#### 10.3.4 LLM-as-Judge 配置

| 配置项 | 设定 | 理由 |
|--------|------|------|
| Judge 模型 | MIMO v2.5 Pro | 需要强推理能力做质量评判 |
| Judge Pass 次数 | 3 次 | 多次取中位数，减少随机偏见 |
| Rubric 格式 | 结构化 JSON 输出 | 便于自动化处理 |
| 校准机制 | 每周 20 条人工标注对比 | 检测 Judge 偏移 |

#### 10.3.5 数据飞轮

```
生产环境
  ├── 任务执行 → 自动评估 → 通过 → 正常结束
  │                         └── 不通过 → 失败案例采集
  │                                         ↓
  │                               人工审查队列（飞书通知）
  │                                         ↓
  │                               标注为：成功/失败/隐藏失败
  │                                         ↓
  │                               回归测试集更新
  │                                         ↓
  └─────────────────────────────── Prompt 改进 → 版本更新 → 回归测试
```

#### 10.3.6 工作量、价值、风险评估

| 维度 | 评估 |
|------|------|
| **工作量** | 中高。评估脚本 + LLM Judge + 飞轮机制，估计 5-7 天 |
| **用户价值** | 高。质量是生产首要障碍，评估闭环是持续改进的基础 |
| **风险** | 中。LLM Judge 偏见（顺序偏见、长度偏见）需要校准机制 |
| **依赖** | 依赖共享记忆系统（存储评估结果） |

#### 10.3.7 验收标准

- **上线后 1 周**，评估框架能对所有 Agent 输出自动评分
- **上线后 2 周**，自动评估与人工评估一致性 > 80%
- **上线后 1 个月**，数据飞轮运转：至少 10 个失败案例回流到测试集并改进 Prompt

---

### 10.4 组件四：人机协作审批

**做的是什么：** 按操作风险等级（L0-L3）定义审批规则，L2/L3 操作通过飞书审批卡片实时通知阿禹，确保高风险操作必须人类确认。

**为什么做：** Agent 自主性越高，风险越大。[Source: OpenAI, "Practices for Governing Agentic AI Systems", 2024] 要求明确 Agent 生命周期中各方的责任。审批是安全底线。

#### 10.4.1 风险分级详细定义

| 等级 | 名称 | 定义 | 审批模式 | 典型操作 |
|------|------|------|---------|----------|
| **L0** | 只读 | 不修改任何状态 | 全自主 | web_search, memory_search, read, health_check |
| **L1** | 安全写入 | 修改可回滚的状态 | 事后通知 | write, edit, shared_memory_write, experience_record |
| **L2** | 风险操作 | 修改重要状态或涉及外部交互 | 实时审批 | exec(非危险命令), shared_memory_delete, 发飞书消息 |
| **L3** | 危险操作 | 不可逆或影响范围大 | 确认码审批 | rm -rf, DROP TABLE, 生产部署, 修改 AGENTS.md |

#### 10.4.2 审批流程

```
Agent 发起操作
  → 风险等级判定（确定性规则，非 LLM）
    → L0: 直接执行 + 写入活动日志
    → L1: 执行 + 飞书事后通知（异步）
    → L2: 暂停 → 飞书审批卡片 → 等待阿禹响应
         → 阿禹批准 → 执行
         → 阿禹拒绝 → 跳过 + 通知 Agent
         → 超时（30min）→ 自动拒绝 + 通知 Agent
    → L3: 暂停 → 飞书审批卡片（含确认码）→ 阿禹输入确认码
         → 码匹配 → 执行
         → 码不匹配 → 拒绝 + 告警
         → 超时（15min）→ 自动拒绝 + 告警
```

#### 10.4.3 飞书审批卡片设计

```json
{
  "msg_type": "interactive",
  "card": {
    "header": {
      "title": { "content": "[L2] Agent 操作审批请求", "tag": "plain_text" },
      "template": "red"
    },
    "elements": [
      {
        "tag": "div",
        "text": {
          "content": "**Agent:** tech-dev\n**操作:** 执行 shell 命令 `pip install requests`\n**风险等级:** L2\n**上下文:** 正在安装 Python 依赖以完成数据分析任务\n**超时:** 30 分钟后自动拒绝",
          "tag": "lark_md"
        }
      },
      {
        "tag": "action",
        "actions": [
          { "tag": "button", "text": { "content": "批准", "tag": "plain_text" }, "type": "primary", "value": { "action": "approve" } },
          { "tag": "button", "text": { "content": "拒绝", "tag": "plain_text" }, "type": "danger", "value": { "action": "reject" } }
        ]
      }
    ]
  }
}
```

#### 10.4.4 审批超时策略

| 场景 | 超时时间 | 超时动作 |
|------|---------|----------|
| L2 操作 | 30 分钟 | 自动拒绝 + 通知 Agent |
| L3 操作 | 15 分钟 | 自动拒绝 + 飞书告警 |
| 批量操作 | 60 分钟 | 自动拒绝 + 人工复核 |
| 阿禹不在线 | - | 审批请求暂存队列，阿禹上线后批量处理 |

#### 10.4.5 工作量、价值、风险评估

| 维度 | 评估 |
|------|------|
| **工作量** | 中等。风险规则 + 飞书卡片 + 超时机制，估计 3-4 天 |
| **用户价值** | 高。安全底线，防止 Agent 失控 |
| **风险** | 低。确定性规则，不依赖 LLM |
| **依赖** | 依赖飞书 Bot API |

#### 10.4.6 验收标准

- **上线时**，L2/L3 操作拦截率 100%（零漏触发）
- **上线后 1 周**，飞书审批卡片送达延迟 < 5s
- **上线后 2 周**，阿禹平均审批响应时间 < 30min
- **上线后 1 个月**，零越权操作（渗透测试确认）

---

### 10.5 组件五：安全可观测性

**做的是什么：** 建立三层防御（输入过滤 + 工具权限 + 审批拦截）+ 全量活动日志 + 成本追踪 + 监控面板，让系统行为完全可见。

**为什么做：** [Source: LangChain, "State of Agent Engineering", 2025] 指出 89% 的生产团队已实现可观测性，这是 table stakes。没有可观测性，调试 Agent 行为就是黑箱。

#### 10.5.1 Prompt 注入防御（三层防线）

| 层级 | 防御内容 | 实现方式 |
|------|---------|----------|
| **第一层：输入过滤** | 检测已知注入模式 | 正则匹配：`ignore previous instructions`、`system prompt`、角色覆盖等 |
| **第二层：工具权限** | 限制工具调用范围 | 工具白名单 + 参数校验 |
| **第三层：输出过滤** | 防止泄露敏感信息 | PII 检测 + system prompt 片段检测 |

#### 10.5.2 活动日志 Schema

```json
{
  "timestamp": "2026-06-04T09:15:30Z",
  "event_type": "tool_call",
  "agent_id": "tech-dev",
  "task_id": "task-abc-123",
  "session_id": "session-xyz-789",
  "tool_name": "exec",
  "tool_input": { "command": "ls -la" },
  "tool_output": { "exit_code": 0, "stdout": "..." },
  "risk_level": "L2",
  "approval_status": "approved",
  "approver": "阿禹",
  "token_usage": { "input": 1200, "output": 350, "total": 1550 },
  "latency_ms": 2300,
  "model": "mimo-v2.5-pro",
  "error": null
}
```

所有日志写入 SQLite `activity_logs` 表，JSONL 格式备份到文件。

#### 10.5.3 成本报表

| 维度 | 报表内容 | 推送频率 |
|------|---------|----------|
| 按 Agent | 每个 Agent 的 token 消耗和成本 | 每日 |
| 按任务 | 每个任务的总 token 消耗 | 每日 |
| 按模型 | MIMO vs DeepSeek 的使用比例 | 每周 |
| 预算告警 | 日消耗超过预设阈值时飞书告警 | 实时 |

#### 10.5.4 监控面板（飞书每日推送）

每日早 9 点推送，内容包含：

| 模块 | 内容 |
|------|------|
| 系统概览 | 活跃 Agent 数、执行任务数、成功率 |
| 成本概览 | 昨日总 token 消耗、总成本、与前日对比 |
| 异常摘要 | 失败任务列表、超时事件、安全告警 |
| 性能摘要 | P95 延迟、工具调用成功率 |
| 经验更新 | 新增经验条目数、待确认经验数 |

#### 10.5.5 全链路 Trace

每个任务分配唯一 `task_id`，贯穿所有 Agent 调用和工具调用。通过 OpenClaw 的 session 机制实现，无需额外 header。

日志查询支持：
- 按 task_id 查完整执行链
- 按 agent_id 查某 Agent 的所有操作
- 按时间范围查系统活动
- 按 risk_level 查高风险操作

#### 10.5.6 工作量、价值、风险评估

| 维度 | 评估 |
|------|------|
| **工作量** | 中等。日志采集 + 成本统计 + 监控推送，估计 4-5 天 |
| **用户价值** | 高。可观测性是调试和安全的基础 |
| **风险** | 低。主要是规则引擎，不依赖 LLM |
| **依赖** | 依赖 SQLite 和飞书 Bot |

#### 10.5.7 验收标准

- **上线时**，Prompt 注入测试集（50 条）拦截率 > 95%
- **上线后 1 周**，活动日志覆盖 100% 的工具调用
- **上线后 2 周**，飞书监控面板每日自动推送
- **上线后 1 个月**，成本追踪准确率 > 99%（对比 API 账单）

---

### 10.6 组件六：经验学习闭环

**做的是什么：** 从任务执行轨迹中自动提取经验（成功模式/失败教训），经人工确认后写入共享记忆，Agent 启动时自动检索相关经验，实现"越用越聪明"。

**为什么做：** 当前系统没有经验积累机制，同样的错误会反复发生。经验学习闭环让系统能从失败中学习，减少重复犯错。

#### 10.6.1 经验条目结构

```json
{
  "id": "exp-20260604-001",
  "type": "failure",
  "agent_id": "tech-dev",
  "task_type": "code_debugging",
  "symptom": "Agent 在调试 Python import 错误时，反复尝试同一解决方案（重装包）3 次未成功",
  "root_cause": "实际问题是虚拟环境未激活，而非包未安装",
  "solution": "遇到 import 错误时，先检查虚拟环境是否激活（which python），再检查包是否安装",
  "tags": ["python", "import", "virtualenv", "debugging"],
  "confidence": 0.85,
  "verified": true,
  "verified_by": "阿禹",
  "verified_at": "2026-06-04T10:30:00Z",
  "created_at": "2026-06-04T09:15:00Z",
  "source_task_id": "task-abc-123",
  "usage_count": 0,
  "success_rate_when_used": null
}
```

#### 10.6.2 经验提取触发条件

| 触发事件 | 提取类型 | 是否需人工确认 |
|---------|----------|---------------|
| 任务失败 | 失败经验提取 | 是（Phase 1 默认阿禹确认） |
| 任务成功但轨迹有隐藏失败 | 反思经验提取 | 是 |
| 同类任务第 3 次成功 | 成功模式提取 | 否（自动入库） |
| 用户显式反馈"这个方法好/不好" | 反馈经验提取 | 否 |

#### 10.6.3 经验检索和注入

Agent 启动时，与记忆一同注入相关经验。只注入已确认（verified=true）的经验，防止未验证的错误经验污染 Agent 行为。

#### 10.6.4 经验衰减策略

| 条件 | 动作 |
|------|------|
| 使用次数 > 10 且成功率 > 80% | 标记为"高价值经验"，永不淘汰 |
| 使用次数 > 5 且成功率 < 30% | 标记为"可能过时"，通知阿禹审查 |
| 创建超过 90 天且使用次数 = 0 | 标记为"待清理"，每月清理 |
| 被新经验覆盖（同症状不同方案） | 旧经验标记为"已被替代" |

#### 10.6.5 工作量、价值、风险评估

| 维度 | 评估 |
|------|------|
| **工作量** | 中等。提取逻辑 + 验证流程 + 衰减机制，估计 3-4 天 |
| **用户价值** | 高。减少重复犯错，系统越用越聪明 |
| **风险** | 中。LLM 提取的经验可能不准确，需要人工确认兜底 |
| **依赖** | 依赖共享记忆系统（存储经验）和评估闭环（触发提取） |

#### 10.6.6 验收标准

- **上线后 1 周**，经验提取流程跑通，成功/失败任务均能触发提取
- **上线后 2 周**，经验库至少 15 条已确认经验
- **上线后 1 个月**，注入经验后同类任务成功率提升 > 10%

---

## 11. 总结与优先级

### 11.1 组件优先级排序

| 优先级 | 组件 | 理由 |
|--------|------|------|
| **P0** | 共享记忆系统 | 其他 5 个组件的基础，没有记忆就无法协作 |
| **P0** | 安全可观测性 | 安全是底线，日志是调试基础 |
| **P1** | 人机协作审批 | 安全防线，与安全可观测性配合 |
| **P1** | 工作流编排引擎 | 自动化核心，但依赖记忆系统 |
| **P2** | 评估反馈闭环 | 质量保障，但可先手动评估 |
| **P2** | 经验学习闭环 | 增值功能，但依赖记忆和评估 |

### 11.2 依赖关系

```
共享记忆系统（P0）──┬──→ 工作流编排引擎（P1）
                     ├──→ 评估反馈闭环（P2）
                     └──→ 经验学习闭环（P2）
                            ↑
安全可观测性（P0）──→ 人机协作审批（P1）
```

### 11.3 里程碑

| 里程碑 | 时间 | 交付物 |
|--------|------|--------|
| M1 | 第 1 周末 | 共享记忆系统 + 安全可观测性上线 |
| M2 | 第 2 周末 | 人机协作审批 + 工作流编排引擎上线 |
| M3 | 第 3 周末 | 评估反馈闭环 + 经验学习闭环上线 |
| M4 | 第 4 周末 | 全系统集成测试 + 阿禹验收 |

---

*文档生成日期：2026-06-04*  
*Author: 产品设计师（product-designer）*