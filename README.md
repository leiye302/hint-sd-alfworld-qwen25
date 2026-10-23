# HINT-SD on ALFWorld — Qwen2.5-1.5B / 3B, 4×A100 80GB

Independent ALFWorld adaptation of **[HINT-SD: Targeted Hindsight Self-Distillation for Long-Horizon Agents](https://arxiv.org/abs/2605.17873)**. The author's [official implementation](https://github.com/wgcyeo/HINT-SD) is pinned to `bc95b44192b06328a88773dbf3b345fa1c499093`; its loss, selected-turn tensor construction, optimizer/backward, and EMA implementation are **not edited**. This is an environment adaptation, not an author-reported ALFWorld result.

两个尺寸分别从固定版本的官方 **Qwen2.5-1.5B-Instruct / Qwen2.5-3B-Instruct** 开始，无 Route-SFT、RETHINK、静态 skill 或外接 Analyzer/API。不是 GRPO+HINT-SD 混合实验。仓库只需要公开源码、公开基模和官方 ALFWorld 数据；不依赖维护者电脑、私有服务器、私有权重或已采集轨迹。

## 1. 保留的原版方法

Student 在线 rollout → **只回看失败轨迹** → EMA Teacher 找出最多三个与失败有关的步骤并生成局部反馈 → Teacher 在对应原始前缀额外看到反馈 → 仅选中步骤计算 Student-to-Teacher reverse KL。

$$
\mathcal L=\frac{1}{N}\sum_{(i,t)\in\mathcal S}
D_{\rm KL}\left(\pi_\theta(\cdot\mid h_i,a_{i,<t})\;\middle\|\;
\operatorname{sg}\pi_{\bar\theta}(\cdot\mid h_i,f_i,a_{i,<t})\right),
\qquad \bar\theta\leftarrow0.999\bar\theta+0.001\theta.
$$

- 使用官方 `Trainer._prepare_bfcl_multi_step_inputs` 的成功轨迹跳过、位置选择和 mask，再直接使用官方 `training_step` / `compute_loss` / `compute_distillation_loss`。
- 官方默认 Student top-100 token＋一个剩余概率桶；按有效选中 token 归一化。不是在某个采样 token 上套 SDAR sigmoid gate，也不是 CE 拟合 Teacher 新生成的纠正轨迹。
- Teacher 在 no-grad 下评分，参数由 Student 的 LoRA 参数做 EMA；反馈也由该 EMA Teacher 生成。不是固定初始 Teacher，也不调用更大的外部模型。
- 保留 LoRA q/v、rank32、alpha64、dropout0.05、LR5e-6、linear scheduler／5% warmup、每任务4条 rollout、batch1／accumulation2、原生 microbatch backward。
- 失败定位 JSON 无法解析时，该轨迹没有蒸馏目标；不猜一个错误位置，也不偷偷切换成 GRPO。记录反馈原文、解析结果和 `objective/inactive`。
- **“选中 action span”的官方代码口径是该步完整 assistant completion。** ALFWorld 响应中的 `<think>...</think><action>...</action>` 如果位于这个 completion 内，就一起蒸馏；没有额外添加我们 Route-SFT 的 reasoning mask。任务、历史、环境观察和其他未选中 turn 不做此损失。这一口径在测试里核对，不能误称为只训练 `<action>` 标签内部。

## 2. 四卡怎么用

官方 colocated 入口明确限制单进程／单卡，不能直接换成四进程 DDP。本适配保留每种尺寸一个训练进程，只将推理搬到另一个 GPU：

| GPU（默认，可在 `.env` 修改） | 工作 |
| --- | --- |
| 0 | 1.5B Student＋EMA Teacher 的训练/评分 |
| 1 | 1.5B rollout＋hindsight generation，vLLM TP1 |
| 2 | 3B Student＋EMA Teacher 的训练/评分 |
| 3 | 3B rollout＋hindsight generation，vLLM TP1 |

两个尺寸并行，共使用四张已分配的 A100 80GB。**不是四个训练 rank，也没有把全局 batch 乘四。** 通过官方 PEFT 导出当前 Student/Teacher LoRA，并用 vLLM 公共 LoRA API 同步；每个版本使用独立 ID，避免缓存旧权重。仅保留两份临时推理 adapter，不接触训练断点。

训练和推理分别加载同一个未量化 BF16 基模；没有固定 Teacher、量化、异步旧策略 rollout 或跨模型权重混用。推理 kernel/设备布局变化不保证和作者单 H200 逐位相同。若需要单独运行某个尺寸，使用 `scripts/train.sh`，另一对卡可保持空闲。

## 3. 干净 Linux 服务器准备

需要 Linux x86-64、四张自己获配的 A100 80GB、可访问 GitHub/Hugging Face/PyPI，建议32逻辑CPU、128–192GiB RAM、80–120GiB空闲磁盘。LoRA断点远小于全参数 Adam 断点；这只是部署预算，不是实测最低要求。驱动必须兼容官方 lockfile 安装的 Torch/vLLM CUDA 版本，不能仅以“有 A100”代替兼容性检查。

```bash
git clone https://github.com/leiye302/hint-sd-alfworld-qwen25.git
cd hint-sd-alfworld-qwen25

# 需要系统 python3>=3.10、git、uv；没有 uv 时按官方安装说明安装：
# https://docs.astral.sh/uv/getting-started/installation/
# TextWorld 需要的系统构建依赖（无 sudo 时请找管理员或使用合适的容器）：
# sudo apt-get update
# sudo apt-get install -y build-essential libffi-dev python3-dev
bash scripts/setup.sh

cp .env.example .env
# 编辑 HINTSD_GPUS，必须是自己获配的四卡，不能抢占别人的卡。
# 在当前 shell 设置本服务器路径，不需要任何 API key：
export ALFWORLD_DATA="$PWD/data/alfworld"
export HINTSD_ENV_PYTHON="$PWD/.env-alfworld/bin/python"
set -a; source .env; set +a

# 下载官方资源，不是重新采集模型轨迹。
.env-alfworld/bin/alfworld-download
python3 -m hint_alfworld.setup --prepare-data

# 阅读基模许可证后，下载两个固定版本及完整校验收据。
# 1.5B为Apache-2.0；3B为Qwen Research License。
.deps/hint-sd/.venv/bin/python -m hint_alfworld.setup --download-model both
python3 -m hint_alfworld.setup --verify

# CPU检查，和正式GPU训练分开。
uv pip install --python .deps/hint-sd/.venv/bin/python pytest==8.3.5
.deps/hint-sd/.venv/bin/python -m pytest tests -q
bash scripts/run_both.sh --dry-run
```

`setup.sh` 使用作者原始 `uv.lock` 安装训练环境（Python>=3.13），另建 Python3.12 的 ALFWorld CPU 环境，避免 TextWorld 旧依赖改坏原版训练栈。ALFWorld 服务只监听 loopback，并由每个 launcher 启动和关闭，无需手工开 Web 服务或把端口暴露公网。官方资源下载还可能包含非文本环境资源，请预留磁盘；可以使用自己配置的下载镜像，但不能换模型 revision 或游戏文件版本。

本仓库借用固定版本 SDAR 的**公开原生 ALFWorld engine、prompt、动作解析**，不安装或调用 SDAR 的训练算法、skill bank 或损失。训练使用其同一3553个可解 train 游戏，准备时严格校验数量；seen140逐游戏SHA不匹配会拒绝启动。来源见 [THIRD_PARTY.md](THIRD_PARTY.md)。

## 4. 检查后开跑

首先在 tmux、screen 或调度器持久会话里完成一个更新的真实 GPU smoke：

```bash
# 两种尺寸并行，各做1个更新，评估只用固定seen140的前2题。
# 只写 runs/hint_sd_1.5b_smoke 和 runs/hint_sd_3b_smoke，不冒充正式成绩。
bash scripts/run_both.sh --smoke
```

检查两条日志里的模型加载、非空 rollout、反馈命中/未命中、有限 loss/grad、Student adapter 更新、EMA、完整 checkpoint 和 GPU 退出。基模未能找出错误时可能出现零目标批，原版会 no-op；不能把 no-op 日志当作已学到纠错。确认 smoke 能够实际更新再正式运行：

```bash
bash scripts/run_both.sh

# 只跑一种尺寸，明确一张训练卡＋一张推理卡：
# bash scripts/train.sh --size 1.5b --train-gpu 0 --infer-gpu 1
# bash scripts/train.sh --size 3b --train-gpu 2 --infer-gpu 3
```

默认每个尺寸150 optimizer updates、初始及每10步真实 seen140、每20步及最终150保存，保留最近两个完整训练断点。最终 Student LoRA 在 `final_adapter/`；同目录的模型获取清单能确定其基模来源。记录在：

```text
runs/hint_sd_1.5b/                  # 另一路为hint_sd_3b
  binding.json                     # 原版源码、适配代码、模型版本、数据/配方SHA
  training.log / environment.log
  metrics.jsonl                    # loss、grad norm、LR、rollout、选中步骤与inactive等
  rollouts/train.jsonl             # 每步真实输入token IDs、响应token IDs、执行动作及反馈
  feedback.jsonl                   # EMA生成的原始反馈和解析结果
  evaluation/update_000/summary.json
  evaluation/update_150/summary.json
  checkpoint-140/                  # Student＋EMA、optimizer/scheduler/RNG/Trainer状态
  checkpoint-150/complete.json     # 完成标志及逐文件SHA；部分写入断点不能续训
  final_adapter/                   # 推理时仅加载Student；不需要Teacher或hint
  completed.json
```

不要整机 `pkill`、`ray stop` 或重置显卡。每条运行只管理自己的训练、推理和CPU环境进程。程序不会清除其他目录，也不会自动覆盖同名运行。中断后先确认旧进程已退出，再从已校验的完整断点恢复：

```bash
bash scripts/run_both.sh --resume
# 或单条 scripts/train.sh ... --resume
```

保存了 Student/EMA/optimizer/scheduler/RNG 和 Trainer 数据进度；模型/数据/源码/配方不同会拒绝恢复。vLLM采样内部状态和已执行 TextWorld 实例不在原版断点中，因此不承诺恢复后和不中断训练逐位相同；正在执行但未完成的 update 不能算已保存。无完整断点时不会盲目从头覆盖。

## 5. ALFWorld 评测协议

与 [SDAR](https://github.com/leiye302/sdar-alfworld-qwen25)、[RLSD](https://github.com/leiye302/rlsd-alfworld-qwen25) 对比仓库使用字节相同的 [seen140_tasks.json](configs/seen140_tasks.json)：

- `valid_seen`固定140个不同游戏，固定顺序和游戏SHA；环境 seed=`2026+task_index`。
- 一任务一条 rollout、最多50动作；原生 SDAR action parser，不加强也不放宽执行规则。
- history2，只保留最近 observation/执行action，不把旧reasoning或未来反馈放入Student输入。
- prompt2048左截断、response512；评测temperature0.4、top_p1、top_k=-1；不加格式重试、marker或hint。
- 报告成功数/140、reasoning、strict-protocol（仅遥测）、原生action-valid、当前可执行和长度截断。协议率不能冒充环境成功率。
- 最终用150步，不用验证集挑出的最优模型替代最终模型；两个尺寸使用同一套任务，绝不各自随机取140题。

手工复测基模或某个已保存的Student（两卡需空闲，输出目录必须是新的）：

```bash
bash scripts/eval.sh --size 1.5b --eval-base --train-gpu 0 --infer-gpu 1 \
  --output "$PWD/runs/base_15b_seen140"
bash scripts/eval.sh --size 3b \
  --eval-checkpoint "$PWD/runs/hint_sd_3b/checkpoint-150" --train-gpu 2 --infer-gpu 3 \
  --output "$PWD/runs/hint_sd_3b_repeat_eval"
python3 -m hint_alfworld.report
```

评测直接生成Student轨迹，没有hindsight服务调用、Teacher反馈或API。CPU环境服务仍需运行；launcher自动管理。训练期的特权反馈不会进入评测输入。

## 6. 相对原版的改动及验证边界

| 项目 | 保留 / 必要适配 |
| --- | --- |
| 方法 | 原版HINT-SD-Multi，失败定位＋最多3步，选中completion reverse KL、EMA、LoRA和backward不变 |
| 环境 | BFCL/AppWorld换为原生ALFWorld文本游戏；错误举例改成实体、状态、前提与计划；JSON/schema/parser仍为原版 |
| 前缀 | 保存vLLM实际输入/输出token IDs；Teacher在同一实际Student前缀额外插入局部反馈，不重建含更多历史的Student前缀 |
| 设备 | 官方单卡共置换成每尺寸1张训练＋1张同步推理卡；通过公共LoRA接口替代上游私有vLLM weight-loader访问 |
| 实验预算 | 两尺寸各150更新，初始/每10步seen140，每20步/最终保存；不是原论文BFCL/AppWorld的15 epochs |
| 每步数据量 | 保留原版batch1、accumulation2、rollout4，即每个optimizer update通常8条新episode；**不声称和SDAR/RLSD的128条/update等数据量** |
| 解码/评测 | ALFWorld共同的history、长度和seen140协议；训练温度保留HINT-SD的0.6，不伪称完全相同采样配方 |

本仓库不修改原版loss，也不为了对齐其他算法悄悄加入GRPO、reference KL、Route-SFT或静态skill。若比较“同样150步”，请同时报告轨迹/token/GPU时间预算，不能把不同方法的update数自动当作相同训练量。

CPU检查覆盖源码锁、原版KL数值与梯度、目标mask、EMA更新、真实前缀保留、数据隔离、GPU分配及断点完整性；GitHub Actions使用干净Linux检出执行。**尚未在四张A100上完成真实CUDA/NCCL/vLLM训练，也未附带任何训练成绩。** CPU检查不是模型能学会ALFWorld或设备运行通过的证明。正式使用前必须跑上面的GPU smoke；遇到版本/资源错误保留日志，不得换loss、换Teacher或跳过校验掩盖问题。

## 7. 许可与引用

包装适配代码Apache-2.0；HINT-SD与SDAR源码遵循其各自Apache-2.0，ALFWorld/模型/资源另遵循来源许可。基模和游戏不随本仓库重新分发。引用原HINT-SD论文，并明确说明这是ALFWorld与Qwen2.5-1.5B/3B的环境适配，而非作者公布的ALFWorld复现分数。
