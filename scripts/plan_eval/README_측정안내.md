# 계획 단계 모델 비교 측정 (베이스 0.6B vs 1.7B)

2026-11-16 · B · 시나리오 A 계획 단계(`plan_brief`)에 쓸 모델을 학습 전에 비교하기 위한 측정 묶음이다.
이 묶음은 **새 파일만** 더한다. 기존 파일은 고치지 않는다. 지우면 원래대로다.

## 무엇을 재나

서버가 계획 LLM에 보내는 것과 **같은 프롬프트**(`build_brief_prompt()` + `build_planner_input()`)를 평가 문장 40개로 두 모델에 보낸다.
출력은 서버와 **같은 검증 함수**(`clean_equipment`·`clean_materials`·`clean_summary`)로 걸러 채점한다 (베끼지 않고 불러 쓴다).

| 측정 | 내용 |
| --- | --- |
| 형식 | 첫 시도 JSON 파싱 성공률, 키·자료형, 생성 한도에서 잘린 비율, 토큰 수 |
| 공구·물질 | 모델이 낸 후보 중 검증 통과분의 정밀도·재현율·F1. 최종 값(슬롯 사전 + 모델)도 같이. 엄격(이름 일치)·느슨(포함) 둘 다 |
| 기준선 | 모델 없이 슬롯 사전(`find_terms`)만 쓴 값. 모델이 이보다 나아야 쓸 이유가 있다 |
| 지어내기 | 뽑을 것이 없는 문장(`none`)에서 후보를 냈는지 (검증 전·후) |
| 요약 | 그대로 쓸 수 있는 비율, 60자 초과 비율, 원문에 근거가 없는 단어가 든 비율(거친 지표) |
| 속도 | `run_speed.py`가 PC에서 따로 잰다 (프롬프트 읽는 시간, 생성 초당 토큰, dtype별) |

**요약이 사실과 맞는지는 코드가 못 정한다.** `*.review.csv`에 사람이 O/X를 적는다.
검증은 "원문에 없는 숫자"와 길이만 본다 (`plan_brief.clean_summary`). 이것이 요약이 가장 약한 고리인 이유다.

## 평가 문장 40개 (`eval_set.jsonl`)

| 유형 | 수 | 보는 것 |
| --- | --- | --- |
| plain | 8 | 평범한 문장 |
| negation | 5 | "크레인은 안 씀", "그라인더 말고", "토치 없이", "용접 대신" 같은 부정·대체는 뺀다 |
| conditional | 5 | "고착되면 토치로", "필요하면 절단기" 같은 조건부는 넣는다 |
| out_of_dict | 10 | 슬롯 사전에 없는 물질·장비 (염산, 황산, 아르곤, 핸드브레이커 …). 모델이 사전보다 더해야 할 몫 |
| multi | 4 | 여러 개 |
| none | 4 | 뽑을 것이 없다 (지어내는지) |
| distractor | 4 | T-105·플랜지·드럼처럼 물질로 오인하기 쉬운 설비·부품·장소 |

- 프롬프트 예시 문장과 계획서 9장 데모 문장은 쓰지 않았다.
- **학습 데이터를 만들 때 이 40문장은 빼야 한다** (평가 오염 방지). 문장·정답 라벨은 내가 쓴 **임시값**이며 현장 문장이 아니다. 실무자 문장으로 바꾸면 더 믿을 수 있다.
- `load_cases()`가 불러올 때 정답이 원문에 없거나 부정돼 있는지 서버 규칙(`mentioned`)으로 검사한다.
- 입력 슬롯(주변 물질·투입 장비)은 `slots.py`와 같은 `find_terms` 호출로 채워서 실제 프롬프트와 같게 만든다.

## 실행

### Colab (품질) — `plan_eval_colab.ipynb`
1. `app.zip`(팀 `app/` 폴더)과 이 묶음(`plan_eval.zip`)을 올린다. GPU 런타임(T4면 충분).
2. 위에서부터 차례로 실행. 0.6B → 1.7B → 비교표 → 결과 zip 내려받기.
3. `out/plan_eval/*.review.csv`를 열어 `요약_사실일치`에 O/X를 적는다.

dtype: 서버는 `torch_dtype="auto"`(체크포인트 dtype)이다. T4는 bfloat16이 안 돼서 노트북이 float16으로 대신한다.

### PC (속도, CPU)
```powershell
python -m scripts.plan_eval.run_speed --model Qwen/Qwen3-0.6B --dtypes auto,float32,bfloat16 --n 5
python -m scripts.plan_eval.run_speed --model Qwen/Qwen3-1.7B --dtypes auto,float32 --n 5
```
- `auto`가 서버와 같은 설정이다. float32와 비교해 CPU에서 bfloat16이 느린지 본다 (**추정**일 뿐 확인된 것이 아니다).
- 첫 문장은 워밍업으로 버린다. 다른 프로그램을 끄고 돌린다.
- 출력은 프롬프트 토큰 수·읽는 시간, 생성 토큰 수·시간·초당 토큰, 합계.

### PC에서 품질을 직접 돌릴 때
```powershell
python -m scripts.plan_eval.run_quality --model Qwen/Qwen3-0.6B --dtype auto --device cpu --out out/plan_eval
python -m scripts.plan_eval.compare out/plan_eval/*.summary.json --out out/plan_eval/compare.md
```
CPU에서는 40문장에 오래 걸린다. `--limit 10`으로 줄일 수 있다.

## 시험

```powershell
python -B -m pytest -p no:cacheprovider tests\test_plan_eval_harness.py
```
34건. torch·transformers 없이 돈다 (pydantic·pydantic-settings·pytest만). 모델 호출부는 **가짜 torch·transformers**로 시험했다.

## 알려둔 한계

- **실제 모델로는 아직 한 번도 돌려 보지 않았다.** `model_runner.py`는 서버의 `generate_chat`과 같은 호출(템플릿 인자·그리디·디코딩·끊김 판정)로 쓰였지만, 실제 transformers에서의 동작은 첫 Colab 실행으로 확인된다.
- `transformers` 버전에 따라 `dtype=`과 `torch_dtype=` 중 하나만 받는다. 둘 다 시도한다.
- 요약 근거 지표(`원문 근거 비율`)는 임시 휴리스틱이다 (두 글자 이상 단어의 끝 한 글자를 뗀 앞부분이 원문에 있는지). 의역이 많으면 낮게 나온다. 사람이 보는 표의 보조 자료일 뿐이다.
- 재시도는 흉내 내지 않는다. 서버는 JSON 파싱에 실패하면 1회 재시도하므로, 파싱 성공률이 낮을수록 계획 시간이 최대 약 2배가 된다.
- 어댑터를 붙인 모델 평가(`--adapter`)는 아직 없다. 학습 노트북을 만들 때 같이 넣는다.

## 알게 된 것: 슬롯 사전(`text_rules.find_terms`)의 부분 문자열 일치

측정을 만들다가 발견했다. `find_terms`는 한국어 단어 경계를 보지 않아서 짧은 이름이 긴 이름 안에서도 잡힌다.

| 입력 | 잡히는 것 |
| --- | --- |
| `메탄올 드럼` | **메탄**, 메탄올 |
| `과산화수소 통` | **수소** |
| `황화수소 확인` | 황화수소, **수소** |

평가 40문장 중 4건(p05·n01·o03·m02)에서 재현됐다. 이 값이 `slots.nearby_materials`가 되어 계획 프롬프트의 "주변 물질" 줄, MSDS 조회(`메탄`은 CAS 조회 대상), 허가서 주변 위험물질 표, TBM 응급조치 줄에 쓰일 수 있다 (코드 읽기와 재현은 했지만 **화면에서 확인한 것은 아니다**). 이 묶음은 `text_rules.py`를 고치지 않았고, `compare.py`가 "슬롯 사전만으로 정답에 없는 이름이 잡힌 문장"을 따로 센다. 고칠지는 별도 결정이다.
