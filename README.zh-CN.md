# bio-ml-preflight

[English](README.md)

用低成本基线和严格切分，判断一份科学数据到底支持什么预测或排序结论、边界在哪里、下一步最值得补什么数据。

它不是 AutoML，不负责发现生物机制，也不会把相关性写成因果关系。

## 能做什么

- 检查缺失值、重复样本、重复实体、批次、重复测量和标签冲突
- 检测随机切分、实体重叠、重复对和可疑 ID 特征造成的泄漏
- 比较随机、分组、冷实体、双冷实体、时间和自定义切分
- 运行 Dummy、线性模型、Extra Trees、梯度提升、近邻等 CPU 基线
- 在完全相同的切分清单和模型族下比较字符哈希与 RDKit Morgan 指纹
- 做独立单位 Bootstrap、分组标签置换和学习曲线
- 检查整体指标稳定但 Top-k 候选不稳定的问题
- 输出分场景能力结论，不生成一个误导性的总分

能力状态包括：`SUPPORTED`、`SUPPORTED_WITH_LIMITS`、`INSUFFICIENT_EVIDENCE`、`CONTRADICTED`、`NOT_ASSESSABLE`。

## 安装

需要 Python 3.11+ 和 [uv](https://docs.astral.sh/uv/)。

```bash
uv sync --extra dev
uv run bio-ml-preflight --help
```

在会隐藏 editable `.pth` 的 macOS 环境中，最省事的入口是：

```bash
make demo
```

## 跑合成数据 Demo

一次运行四种情况：稳定信号、泄漏陷阱、无信号、排序不稳定。

```bash
uv run bio-ml-preflight demo synthetic --budget smoke
```

报告写入 `reports/synthetic/`。

## 跑 Davis 公共数据 Demo

通过 TDC 下载并缓存 Davis 数据，比较随机配对、冷药物和冷靶点切分。

```bash
uv sync --python 3.11 --all-extras
uv run --python 3.11 bio-ml-preflight demo davis --budget smoke
```

完整一些的运行：

```bash
uv run --python 3.11 bio-ml-preflight run examples/davis/case.yaml --budget standard
```

下载的数据在 gitignored 的 `data/cache/`，不会提交到仓库。

## 跑 UCI Parkinsons Telemonitoring 公共数据 Demo

[UCI Parkinsons Telemonitoring](https://archive.ics.uci.edu/dataset/189/parkinson)
包含 42 名参与者的 5,875 次居家语音记录，采用 CC BY 4.0 许可；适配器用校验和固定官方压缩包。case 只用官方声明的 16 个语音测量预测线性插值的 `motor_UPDRS`，年龄、性别、记录时间和 `total_UPDRS` 仅保留为审计上下文，不会悄悄进入模型。

```bash
uv run bio-ml-preflight demo parkinsons --budget smoke
```

smoke 结果清楚显示了独立单位边界。随机记录诊断的 Spearman 中位数为 `0.599`（置换差值 `0.237`、经验 p 值 `0.10`），但 42 名参与者全部同时出现在训练和测试中，因此只能判为 `SUPPORTED_WITH_LIMITS`，不能作为未见参与者或独立记录时点的证据。源数据只有 2,501 个精确的“参与者—时间”代理组，5,875 条记录中有 4,904 条属于重复代理组。两个按参与者整组留出的 smoke 切分均无参与者重叠，Spearman 中位数降至 `0.195`，置换差值 `0.070`、p 值 `0.30`、跨切分标准差 `0.158`；报告会同时记录这三项限制，结论为 `INSUFFICIENT_EVIDENCE`。测量可靠性与批次混杂仍为 `NOT_ASSESSABLE`。这是回顾性结果，不证明未来时间外推、临床效用或因果效应。

## 跑 BBB_Martins 分子二分类 Demo

使用 SMILES 预测二分类 BBB 标签，并比较随机化合物与 scaffold 隔离验证：

```bash
uv run --python 3.11 --all-extras bio-ml-preflight demo bbb --budget smoke
# 等价的一键命令：
make demo-bbb
```

该 case 会在切分前显式排除 12 个存在标签冲突或 SMILES 不一致的化合物 ID（共 24 行），并把策略写入报告。字符哈希与 Morgan 指纹复用同一组四个切分清单和 logistic/Extra Trees 模型。

缓存 smoke 结果中，随机化合物 balanced accuracy 为字符 `0.733`、Morgan `0.808`；scaffold 验证为字符 `0.709`、Morgan `0.749`。两种表示在两个场景中都得到 `SUPPORTED`，化合物跨折数均为 0。它们仍只是回顾性预测证据，不是外部确认或因果结论。

v0.1 不加入 GNN。只有当字符/Morgan 基线在决策相关边界上失败、独立化合物数量足以无泄漏训练图模型，并且存在锁定外部验证时，才值得沿现有特征与评估接口增加学习式图表示。

## 图模型就绪门

BBB case 现在包含严格的二维分子图契约，但不引入 GNN 框架。契约固定规范图身份、节点/边特征、无效结构策略、独立单位、case 专用类别支持下限和 scaffold 验证；审计还会检查每个选定切分中的规范图重叠，而不只检查化合物 ID。

```bash
uv run bio-ml-preflight assess-graph-readiness reports/bbb-martins \
  --external-report reports/petbd-external-confirmation \
  --output reports/bbb-graph-readiness
```

该命令只读取已有 JSON 与 Parquet，不重新拟合基线，也不会再次访问 PETBD holdout。当前产物中，2,006 个开发化合物全部可转换为 1,955 个规范图；两个 scaffold manifest 的规范图重叠均为 0，测试折至少含 104 个阴性化合物；字符/Morgan 基线的能力结论都保持 `SUPPORTED`。与此同时，类别支持充分的 PETBD 确认没有脱离置换分布。因此综合结论为 `NOT_JUSTIFIED_BY_CURRENT_EVIDENCE`：这不是说 GNN 永远无效，而是现有证据没有把固定分子表示识别为主要瓶颈。随机化合物切分分别有 23 和 18 个规范图重叠，不能作为图模型就绪证据。

## 跑锁定的 B3DB 发布后确认

该 Demo 在查看外部模型结果前锁定 BBB_Martins 开发阶段选出的 Morgan 表示与 logistic 模型。适配器固定并校验 B3DB 官方源码，用 RDKit InChIKey 统一身份，并从开发集移除与 175 条发布后记录重合的 10 个化合物，同时保留完整外部集。最终 supplied manifest 含 1,992 条训练记录和 175 条 holdout 记录，化合物重叠为 0；访问账本按数据校验和固定，并为每次访问记录 case 指纹，因此更换输出目录或 case 参数也不能绕过单次访问限制。

```bash
uv run --python 3.11 --all-extras bio-ml-preflight demo bbb-external
# 等价的一键命令：
make demo-bbb-external
```

锁定运行的 balanced accuracy 为 `0.904`，9 次置换中位数 `0.496`（差值 `0.407`，经验 p 值 `0.10`）。但能力结论仍为 `INSUFFICIENT_EVIDENCE`：外部集有 171 个阳性、仅 4 个独立阴性化合物，未达到预先声明的每类至少 20 个。最便宜的下一证据是在相同协议下再收集至少 16 个独立阴性 holdout 化合物。它是公开数据上的发布后伪封存检查，不是真盲或按实验时间前瞻的研究；测量可靠性与批次混杂仍为 `NOT_ASSESSABLE`。运行后的治理审查在不重新读取外部标签的前提下强化了缓存校验、来源记录和访问账本；原始数值产物仍是事实来源。

## 分析自己的表格

支持 CSV、TSV 和 Parquet。

```bash
uv run bio-ml-preflight init-case data.csv --output case.yaml
```

打开 `case.yaml`，确认目标列、预测单位、实体、特征、切分场景和独立采样单位。自动推断的角色默认未确认，工具不会猜测生物学含义。

然后运行：

```bash
uv run bio-ml-preflight validate-case case.yaml
uv run bio-ml-preflight run case.yaml --budget smoke
```

没有明确任务时，先生成可审核的候选任务：

```bash
uv run bio-ml-preflight discover data.csv --output discovery/
```

确认 `case.yaml` 后，可以在不训练模型的情况下检查开发数据质量：

```bash
uv run bio-ml-preflight eda case.yaml --output reports/example-eda
```

EDA 将 `eda.json` 和 `column_profile.parquet` 作为结构化事实来源，同时生成简洁的
Markdown/HTML 报告，以及缺失率、目标分布和实体重复图。它会报告身份或标签冲突、
重复实体、潜在泄漏、测量可靠性边界和缺失的高价值元数据；不会自动删除异常值，也
不会给出一个掩盖问题差异的通用质量分数。启用 holdout 的 case 会在读取数据表前被
拒绝；普通 supplied split 中的非训练行不会进入 EDA。

## 输出

每次运行会生成一个报告目录，主要包括：

- `report.md`、`report.html`、`report.json`
- 数据审计和能力矩阵
- 指标、置换检验和排序稳定性表
- `representation_sensitivity.parquet` 中的逐表示、同模型配对结果
- 固定的切分清单与每次运行的预测结果
- 图表、环境信息、数据指纹和运行配置

先看 `report.md`，需要复查数值时看 `report.json` 和 Parquet 表。

## 开发检查

```bash
uv run ruff check .
uv run ruff format --check .
uv run mypy src
uv run pytest
```
