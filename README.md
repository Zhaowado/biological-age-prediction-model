# 体能衰老指数与探索性生物学年龄计算包

这是一个可独立运行的冻结预测包，实现已建立的体能评分体系，并将年龄和性别标准化后的功能偏差换算为探索性 Biological Age（BA）。包内不含原始参与者数据、真实 `oneid`、生日或逐条体测记录。

## 计算链路

```mermaid
flowchart LR
    A[单次体测成绩] --> B[单位与物理边界 QC]
    B --> C[分性别、分年龄参考均值与标准差]
    C --> D[单项功能缺损 z 值]
    D --> E[领域内平均]
    E --> F[FAI: 中心 50, 10 分约为 1 SD]
    F --> G[BA = 年龄锚点 + 0.4360848124 x (FAI - 50)]
```

- FAI 越高表示相对同龄、同性别参考的功能缺损越大。
- 七项共同指标为握力、纵跳、肺活量、台阶指数、坐位体前屈、单脚站立和反应时。
- 男性俯卧撑和女性仰卧起坐作为不同项目保留，不直接合并。
- 七项完整时直接计算 FAI；缺项时只对已冻结且有验证记录的缺失组合使用岭回归桥接，并标为低可信度。
- 参考曲线训练范围为 20–44.999 岁，完整七项的主要参考范围为 20–39 岁。范围外输出会标为低可信度。

公式、领域合成方式和缺项策略见 [docs/方法学说明.md](docs/方法学说明.md)，模型范围见 [docs/MODEL_CARD.md](docs/MODEL_CARD.md)。

## 快速开始

建议 Python 3.10–3.13。

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python calculate_ba.py examples/example_input.csv output/example_output.csv
```

完整验证：

```bash
sh run_example.sh
```

## 单次或单行批量计算

输入 CSV 每行为一次体测：

```bash
python calculate_ba.py input.csv output.csv
```

输入模板为 [examples/input_template.csv](examples/input_template.csv)，去标识真实体测示例为 [examples/example_input.csv](examples/example_input.csv)，预期输出为 [examples/example_output.csv](examples/example_output.csv)。

| 字段 | 单位/取值 | 说明 |
| --- | --- | --- |
| `oneid` | 文本 | 可选，原样带回 |
| `sex` | `F/M`、女/男 | 参考性别 |
| `age` / `exact_age` | 岁 | `exact_age` 优先 |
| `birthday`, `exam_date` | `YYYY-MM-DD` | 两者齐全时优先计算精确年龄 |
| `RA_anchor` / `RA` | 岁 | BA 的年龄锚点，缺失时使用已提供年龄 |
| `grip` | kg | 握力 |
| `jump` | cm | 纵跳 |
| `vital` | mL | 肺活量 |
| `step` | index | 台阶指数 |
| `reach` | cm | 坐位体前屈，可为负 |
| `balance` | s | 单脚站立 |
| `reaction` | s | 反应时 |
| `pushup` | 次 | 男性俯卧撑 |
| `situp` | 次 | 女性仰卧起坐 |

也接受 `Grip-score`、`Vertical-jump-score` 等原始项目名。程序不会自动猜测毫秒/秒、千克/公斤等单位换算。

## 按年计算

输入可以是每行一次访视的宽表，也可以是包含 `exam_item/value` 的长表。同一人同一年内，每个指标取该年最后一次有效值。

```bash
# 每行一次访视
python calculate_annual_ba.py visits.csv annual_BA_wide.csv \
  --output-long annual_BA_detail.csv

# exam_item/value 长表，另提供 oneid/sex/birthday 基本信息
python calculate_annual_ba.py sports_fitness_long.csv annual_BA_wide.csv \
  --basic-info basic_info.csv \
  --output-long annual_BA_detail.csv
```

宽表输出格式为每行一个 `oneid`，每列一个年份。明细表保留计算方法、参考年龄、低可信度和缺失组合标记。

## 关键输出

| 字段 | 含义 |
| --- | --- |
| `FAI` | 年龄/性别标准化功能缺损评分 |
| `BA` | 探索性年龄尺度 |
| `BA_gap` | `BA - BA_age_anchor` |
| `BA_method` | 完整七项或缺项桥接 |
| `BA_low_confidence` | 缺项、人口学缺失或超出主参考范围 |
| `BA_imputation_interval_*` | 缺项桥接验证残差区间，不是个体 BA 的完整置信区间 |
| `status` | `evaluated` 或不支持的缺失组合 |

## 目录

```text
calculate_ba.py             单次批量计算
calculate_annual_ba.py      按人按年聚合计算
src/fitness_ba/             核心计算代码
models/                     冻结参考模型、BA 换算尺度、缺项桥接参数
examples/                   模板、去标识输入和预期输出
tests/                      公式、重现性和年度聚合测试
docs/                       方法学和模型卡
```

## 重要局限

BA 是将 FAI 用训练集年龄标准差转换到“年”单位的约定尺度。换算系数用于设定量纲，不是通过最小化实际年龄预测误差得到的。当前结果未经外部人群、健康结局、重测信度或临床阈值验证，不应用于医疗诊断、个人健康承诺或宣称“逆龄”。

本包为研究合作交付物。未附开源许可证，不自动授予商业使用或再分发权利。
