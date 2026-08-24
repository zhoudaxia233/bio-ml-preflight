# bio-ml-preflight

[English](README.md)

用低成本基线和严格切分，判断一份科学数据到底支持什么预测或排序结论、边界在哪里、下一步最值得补什么数据。

它不是 AutoML，不负责发现生物机制，也不会把相关性写成因果关系。

## 能做什么

- 检查缺失值、重复样本、重复实体、批次、重复测量和标签冲突
- 检测随机切分、实体重叠、重复对和可疑 ID 特征造成的泄漏
- 比较随机、分组、冷实体、双冷实体、时间和自定义切分
- 运行 Dummy、线性模型、Extra Trees、梯度提升、近邻等 CPU 基线
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

## 跑 BBB_Martins 分子二分类 Demo

使用 SMILES 预测二分类 BBB 标签，并比较随机化合物与 scaffold 隔离验证：

```bash
uv run --python 3.11 --all-extras bio-ml-preflight demo bbb --budget smoke
# 等价的一键命令：
make demo-bbb
```

这里使用轻量字符特征，不训练 GNN；目的是先判断数据、切分和泛化边界。

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

## 输出

每次运行会生成一个报告目录，主要包括：

- `report.md`、`report.html`、`report.json`
- 数据审计和能力矩阵
- 指标、置换检验和排序稳定性表
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
