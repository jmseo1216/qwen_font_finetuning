# Qwen Font Outline → Skeleton Fine-tuning

Qwen decoder-only causal LM과 QLoRA를 사용해 font outline SVG/path를 skeleton SVG path로 변환하는 실험 파이프라인입니다. 기존 `jmseo1216/font_llm_finetuning`의 동일 파일명 pair, path 중심 전처리, JSONL 기반 학습, raster/Chamfer 평가 흐름을 유지하면서 모델 계층을 T5 계열 encoder-decoder에서 Qwen causal LM으로 교체했습니다.

> **주의:** Qwen 역시 geometry를 직접 이해하는 전용 모델이 아닙니다. 높은 train/random-split 점수만으로 일반화를 판단하지 말고 반드시 unseen font, unseen glyph, font+glyph holdout 결과를 확인하세요.

## 프로젝트 구조

```text
preprocess/                 paired SVG → JSONL 및 holdout split
dataset/                    ChatML causal LM dataset, assistant-only masking
training/                   Transformers + PEFT + bitsandbytes QLoRA
evaluation/                 valid/command/raster/Chamfer/bbox metric
inference/                  outline SVG → skeleton SVG/path 생성
configs/                    1.5B PoC 및 7B 실험 YAML
scripts/                    preprocess/train/infer/eval 실행 wrapper
src/qwen_font_pipeline/     prompt 및 SVG/path 공용 유틸리티
tests/                      dependency-light unit tests
```

## 설치

Python 3.10+와 CUDA 환경을 권장합니다.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 데이터 배치와 JSONL

기존 파이프라인처럼 outline과 skeleton SVG의 파일명을 동일하게 맞춥니다. 권장 파일명은 `<font>_<glyph-code>.svg`입니다. 예를 들면 `Roboto_65.svg`입니다.

```text
data/raw/outline/Roboto_65.svg
data/raw/skeleton/Roboto_65.svg
```

생성되는 각 JSONL row에는 `id`, `font`, `code`, `input_svg`, `target_svg`, `input_path`, `target_path`, canvas 정보와 split metadata가 들어갑니다.

```json
{"id":"Roboto_65","font":"Roboto","code":65,"input_svg":"...","target_svg":"...","input_path":"M ...","target_path":"M ..."}
```

## 1. 전처리 및 일반화 split

```bash
bash scripts/run_preprocess.sh
```

기본값은 `font_glyph`입니다. 필요하면 다음 모드를 선택합니다.

```bash
SPLIT_MODE=random bash scripts/run_preprocess.sh
SPLIT_MODE=font bash scripts/run_preprocess.sh
SPLIT_MODE=glyph bash scripts/run_preprocess.sh
SPLIT_MODE=font_glyph bash scripts/run_preprocess.sh
```

- `random`: sample 기준 무작위 split.
- `font`: test font를 train에서 제외.
- `glyph`: test glyph code를 train에서 제외.
- `font_glyph`: 선택된 test font 또는 glyph code가 train에 들어가지 않도록 제외.

## 2. Qwen QLoRA 학습

```bash
bash scripts/run_train_qwen_lora.sh
```

기본 `configs/qwen2.5-coder-1.5b-qlora.yaml`은 RTX 4090 24GB PoC용입니다. `Qwen/Qwen2.5-Coder-1.5B-Instruct`, 4-bit NF4 double quantization, Qwen projection modules LoRA, gradient checkpointing, bf16, batch size 1, gradient accumulation 16, sequence length 2048을 사용합니다. YAML의 `model_name_or_path`만 바꾸면 `Qwen/Qwen2.5-1.5B-Instruct` 등 다른 Qwen 모델도 실험할 수 있습니다. 7B 실험 시작점은 `configs/qwen2.5-coder-7b-qlora.yaml`입니다.

학습 입력은 Qwen ChatML instruction이며 loss는 assistant skeleton answer 토큰에만 적용됩니다.

```text
<|im_start|>system
You convert font outline SVG path into skeleton SVG path. Output only valid SVG path commands.
<|im_end|>
<|im_start|>user
Outline path:
...
<|im_end|>
<|im_start|>assistant
...
<|im_end|>
```

### RTX 4080 SUPER 16GB 권장값

1. 먼저 1.5B config로 시작합니다.
2. `max_seq_length: 1024` 또는 `1536`, `batch_size: 1`, `grad_accumulation_steps: 16`, `lora_rank: 8` 또는 `16`을 사용합니다.
3. OOM이면 sequence length, LoRA rank 순서로 낮춥니다.
4. 7B는 16GB에서 데이터 길이에 따라 빠듯하므로 1.5B baseline 이후에만 시도합니다.

## 3. 추론

```bash
INPUT_SVG=data/raw/outline/Roboto_65.svg \
ADAPTER_DIR=checkpoints/qwen2.5-coder-1.5b-qlora \
OUTPUT=predicted_skeleton.svg \
bash scripts/run_infer.sh --repair
```

추론기는 SVG에서 outline path를 추출하고, 생성된 assistant 부분만 디코딩하며, path syntax를 검증합니다. `--repair`는 Markdown wrapper 제거와 불완전한 마지막 command 절단만 수행하는 보수적 옵션입니다. `OUTPUT=predicted_skeleton.txt`로 path만 저장할 수도 있습니다.

## 4. 평가

prediction JSONL은 다음처럼 준비합니다.

```json
{"id":"Roboto_65","predicted_path":"M ..."}
```

```bash
PREDICTIONS=predictions/test_predictions.jsonl bash scripts/run_eval.sh
```

평가는 전체 결과와 JSONL의 `generalization_split`별 결과를 저장합니다.

- SVG/path valid rate
- SVG command sequence accuracy
- raster pixel L1 및 IoU
- raster foreground Chamfer distance
- bbox center 및 scale error

## T5 파이프라인과 차이

| 항목 | 기존 T5/ByT5/Flan-T5 | Qwen 파이프라인 |
| --- | --- | --- |
| 모델 | encoder-decoder | decoder-only causal LM |
| 입력/출력 | source와 target 별도 tokenization | system/user/assistant ChatML sequence |
| loss | decoder target | assistant answer token만, prompt token은 `-100` masking |
| 경량 학습 | LoRA 중심 | bitsandbytes 4-bit NF4 QLoRA |
| 기본 모델 | T5 small 계열 | Qwen2.5-Coder-1.5B-Instruct |

## 다음 실험 권장 순서

1. 1.5B PoC로 end-to-end 동작과 invalid rate를 확인합니다.
2. 동일 config로 `random`, `font`, `glyph`, `font_glyph` split을 비교하여 암기와 일반화 차이를 측정합니다.
3. 1.5B Coder와 일반 Qwen2.5 Instruct를 비교합니다.
4. sequence length와 LoRA rank를 조정합니다.
5. 24GB 이상 환경에서 7B config를 실험합니다.
6. 이후 path 정규화, 좌표 양자화, topology constraint decoding을 별도 ablation으로 추가합니다.
