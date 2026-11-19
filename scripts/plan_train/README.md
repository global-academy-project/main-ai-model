# 계획 단계 LoRA 어댑터 학습 (B, 2026-11-17, v2 = T30 규칙 수정 반영)

시나리오 A 계획 단계(`plan_brief`)의 3키 추출 `{summary, equipment, materials}` 을 Qwen3-0.6B·1.7B 위에 LoRA 어댑터로 학습한다.
어댑터만 저장하고(기본 모델은 합치지 않는다, C 의 `extract` 와 같은 방식) 서버가 `LLM_ADAPTERS` 로 붙인다.
계획: `계획단계_어댑터_학습_계획_v0.md`. 서버쪽 연결: T26 (`plan_adapter.py`, `plan_brief` 가 어댑터를 쓸 수 있으면 짧은 프롬프트).

## 파일

| 파일 | 하는 일 |
| --- | --- |
| `scripts/make_plan_train.py` | 학습·검증 데이터 생성 (`data/plan-train/train.jsonl`, `valid.jsonl`). 서버 검증 함수로 모든 줄을 다시 확인 |
| `scripts/plan_train/train_lora.py` | LoRA 학습 (transformers + peft 만). 학습 뒤 서버와 같은 방식으로 valid 를 생성해 간이 확인 |
| `scripts/plan_train/plan_train_colab.ipynb` | Colab 에서 위 둘 + 측정을 차례로 돌리는 노트북 |
| `scripts/plan_eval/eval_set_multiline.jsonl` | 여러 줄(발화를 줄바꿈으로 이은) 평가 문장 12개. 기본 40문장과 따로 잰다 |
| `scripts/plan_eval/eval_set_user_hard.jsonl` | 현장 문장 5 + 어려운 문장 22 (11/17). 이것도 따로 잰다. 학습 데이터가 이 문장과 겹치지 않게 막는다 |
| `apply_plan_train_eval.py` | 기존 평가 도구 4개(`harness`, `model_runner`, `run_quality`, `compare`)를 고치는 정확 일치 적용기 |

## 서버와 같게 맞춘 것

| 항목 | 학습 | 서버 |
| --- | --- | --- |
| 시스템 프롬프트 | `ADAPTER_SYSTEM_PROMPT` (한 줄) | 같음 (`plan_adapter.py`) |
| 사용자 메시지 | `build_adapter_input(WorkContext(...))` (작업 내용 / 주변 물질 / 투입 장비) | 같음 |
| 주변 물질·투입 장비 줄 | 발화마다 `merge_slots` 로 이어서 만든 슬롯 | `work_register` 가 슬롯을 같은 방식으로 만든다 |
| 채팅 템플릿 | `apply_chat_template(add_generation_prompt=True, enable_thinking=False)` | `llm_transformers.generate_chat` (엔진 설정 `enable_thinking=False`) |
| 정답 | JSON + `<|im_end|>`(eos). **손실은 정답+eos 에만** | 그리디, eos 에서 멈춤, `BRIEF_MAX_TOKENS=160` |
| 검증 | 데이터 줄마다 `clean_equipment`·`clean_materials`·`clean_summary` 를 거쳐 하나도 안 버려지는 줄만 | 같은 함수 |
| 붙이는 방식 | `get_peft_model` 로 학습, `save_pretrained` | `PeftModel.from_pretrained(adapter_name="work_register")` |

## 학습 데이터 (합성)

- 8가지 유형(평문·부정·조건부·사전 밖 이름·여러 개·없음·헷갈리는 말·**말 많은 문장(verbose)**) 비율은 `KIND_WEIGHTS`, 약 4분의 1(2,400건 중 593건)은 발화 2~3개로 나눠 줄바꿈으로 잇는다 (서버에서 되묻기 답이 이어 붙는 모양).
- **verbose (11/17, T30)**: 과거형 서술에 이유·잡담이 섞인 문장("…교체했음. 볼트가 안 풀려서 임팩트렌치 가져와서 풀었고 … ㅎㅎ"). 요약은 '장소 작업'만(끝맺음·동사를 끌고 오지 않음, 가끔 ", N명" 포함). 학습 요약이 모두 '장소+부품+작업명' 한 틀이라 틀 밖 문장에서 요약이 잘리거나 어색해지던 것(T28 9번)을 줄이려는 것이다. 2,400건 중 297건.
- 학습용 사전 밖 이름에서 질산·초산·아세트산을 뺐다(사전에 들어옴). 검증용 이름에서 윈치·헬륨을 뺐다(어려운 평가 문장 정답). 대신 사전 안 이름으로 사다리차·황산·염산·질산·초산이 학습에 나온다(사다리차 109곳·황산 59·염산 72·질산 61·초산 55).
- **뒤 발화가 앞의 말을 거두는 줄**이 2,400건 중 118건(4.9%) 들어 있다 ("그라인더로 하려 했는데 / 그라인더 대신 드릴로"). 서버 검증(`mentioned`)은 이 거둠을 못 걸러내므로(T28 2번) 모델이 스스로 빼야 하는 모양이다. 장비는 슬롯 줄(`merge_slots`)에서도 빠지고, **주변 물질 슬롯은 뒤 발화에서 거둬도 빠지지 않아서**(`merge_slots` 는 장비만 뺀다) 물질은 슬롯에 없는 사전 밖 이름만 이 모양으로 넣었다.
- 사전 밖 물질·장비는 학습용/검증용 이름을 따로 두고 서로 겹치지 않는다 (검증이 "처음 보는 이름"을 본다).
- **평가 문장과 겹치면 버린다**: 40문장 + 여러 줄 12문장 + 현장·어려운 27문장(총 79) 모두, 정답 이름·문장 단위·3글자 묶음 유사도(0.6)로 검사. 사전 밖 어휘가 평가 문장 본문에 나와도 막는다 (정답에서 거둔 말 포함).
- "쓰지 않는다" 류는 서버의 부정 정규식(`text_rules.NEGATION_AFTER`)이 잡는 표현으로만 정답에서 뺀다. 정규식은 두 번 넓어졌다: 11/17 정중형·완료형(`안 씁니다`·`쓰지 않음` …, `POLITE_NEGATION`, T28)과 11/17 2차 `사용 금지`·`쓸 일 없`·`쓸 계획 없`·`쓸 필요가 없`(`PROHIBIT_NEGATION`, T30). 이 표현들은 이제 일반 부정이다. 아직 못 잡는 말(`빼고 진행함`·`가져오지 않음`·`안 가져옴`·`반입하지 않음`·`반입 안 함`)만 **슬롯 줄에는 남고 정답에서는 빠지는** 별도 줄(soft)로 넣는다. **이 규칙들이 없는 app 으로는 생성기가 처음에 멈춘다** (`require_polite_negation`, `require_rules_v2`).
- 같은 seed 면 바이트까지 같다. 기본 `--train 2400 --valid 240 --seed 2026`.

## 실행

```powershell
# 데이터 (PC 에서도 된다; 평가 문장과 겹치는지 자동 검사)
python -m scripts.make_plan_train --out data/plan-train
python -m scripts.make_plan_train --sample 6          # 파일을 쓰지 않고 예시만

# 학습은 Colab 권장 (plan_train_colab.ipynb). PC 에서 하려면 GPU 가 필요하다
python -m scripts.plan_train.train_lora --model Qwen/Qwen3-0.6B --data data/plan-train --out adapters/plan_brief_0.6b
python -m scripts.plan_train.train_lora --model Qwen/Qwen3-1.7B --data data/plan-train --out adapters/plan_brief_1.7b

# 측정 (어댑터가 있으면 짧은 프롬프트가 기본, 서버와 같은 방식으로 붙인다)
python -m scripts.plan_eval.run_quality --model Qwen/Qwen3-0.6B --adapter adapters/plan_brief_0.6b --out out/plan_eval
python -m scripts.plan_eval.run_quality --model Qwen/Qwen3-0.6B --adapter adapters/plan_brief_0.6b --out out/plan_eval_multi --eval-set scripts/plan_eval/eval_set_multiline.jsonl
python -m scripts.plan_eval.compare "out/plan_eval/*.summary.json" --out out/plan_eval/compare.md   # 따옴표로 감싸도 된다 (compare 가 직접 푼다)
```

- 어댑터 측정의 비교 기준은 세 가지다: 긴 프롬프트·어댑터 없음(지금 서버), **짧은 프롬프트·어댑터 없음**(프롬프트만 바꾼 효과), 짧은 프롬프트·어댑터. 마지막 둘을 가르지 않으면 어댑터 덕인지 프롬프트 덕인지 알 수 없다.
- 기본 40문장과 여러 줄 12문장은 `--out` 폴더를 따로 쓴다 (파일 이름에 평가 세트가 들어 있지 않아 같은 폴더면 덮어쓴다).

## 저장되는 것 (`--out` 폴더)

`adapter_config.json`, `adapter_model.safetensors`, `train_meta.json`. `train_meta.json` 에는 기본 모델 이름·허브 커밋 해시, 학습 데이터 SHA256, 시스템 프롬프트 SHA256, 하이퍼파라미터, 에폭별 손실, 생성 확인 결과, torch·transformers·peft 판이 있다.
어댑터는 학습한 기본 모델·판에만 붙는다. 서버(`_attach_adapter`)는 기본 모델 일치를 **검사하지 않으므로**, `run_quality --adapter` 가 대신 모델 이름이 다르면 멈추고 커밋 해시가 다르면 경고한다 (`summary.json` 의 `adapter_info.revision_match`).

## 서버에 쓰려면 (④, 결과를 본 뒤)

`.env`/compose 에 `LLM_ADAPTERS={"work_register": "<어댑터 폴더>"}` 한 줄 (없으면 기본 위치 `adapters/work_register`). 그 폴더에 어댑터 파일(설정 + 가중치)이 있을 때만 `WORK_REGISTER` 엔진이 어댑터를 붙이고 짧은 프롬프트를 쓴다.
A 엔진 모델(`model_id`)이 학습한 기본 모델과 같아야 한다. 1.7B 로 정하면 A 가 D 와 같은 모델을 쓰게 되어 메모리·생성 잠금이 달라진다 (T22). 이 결정은 측정 결과를 본 뒤에 한다.

## 검증 범위

- **샌드박스(검증함)**: 생성기 88건·규칙 v2 54건·어려운 평가 세트 5건·학습기 19건(아주 작은 랜덤 Qwen3 로 CPU 학습·저장·다시 읽기·서버 방식 붙이기·마스킹 손실 값 대조)·평가 확장 31건. 학습 입력이 서버 호출과 같은 토큰인지 직접 비교.
- **미확인**: 진짜 Qwen3 모델·토크나이저로 한 번도 돌리지 않았다 (샌드박스에 허브 접근이 없다). GPU 경로(float16 autocast + 손실 스케일러, bfloat16)는 CPU 만 돌렸다. 그래서 노트북에 스모크 단계를 넣었다.
- **재학습 전**: 이 판(v2)으로 학습한 어댑터의 정확도는 아직 모른다. 앞 판(v1) 어댑터는 40문장 99%·여러 줄 12문장 98%·어려운 27문장 엄격 90% 였고, 사전·부정 규칙이 바뀌어 **v1 숫자와 직접 비교할 수 없다** (같은 규칙으로 기준선부터 다시 잰다).
