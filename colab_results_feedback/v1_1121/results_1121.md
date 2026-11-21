$ /content/venv/bin/python experiments/p0/p0.py env golden regress --tiny --root /content/work/p0-tiny
== env ==
{
  "python": "3.11.18",
  "torch": "2.7.1+cu128",
  "cuda_runtime": "12.8",
  "cuda_available": true,
  "gpu_name": null,
  "driver_version": "580.82.07",
  "bf16_matmul_finite": null,
  "mismatched": {
    "torch": {
      "installed": "2.7.1+cu128",
      "pinned": "2.6.0"
    }
  },
  "git": {
    "head": "cdea00b095d26a14a71a2100d3f4ef358ec2dd62",
    "branch": "HEAD"
  }
}
WARNING: installed package versions differ from pyproject.toml (see 'mismatched')

== golden ==
  [instrumented-hofl-a] starting: ho_fl.train --model opt --dataset sst2 --method hofl --beta 0.75 --rounds 3 --evaluate-every 3 --seed 42 --save-state-rounds 3 --model-path /content/work/p0-tiny/assets/tiny-opt --token-cache /content/work/p0-tiny/assets/tiny-sst2-cache.json --cache-dir /content/work/p0-tiny/assets/hf-cache-instrumented --device cpu --output /content/work/p0-tiny/golden/cdcaf6de099a-cpu-torch2.7.1/instrumented-hofl-a
    [instrumented-hofl-a] eval @ round 0: accuracy=0.4817 task_loss=0.6989 t=3s
    [instrumented-hofl-a] round 1 train_loss=0.6951 t=3s
    [instrumented-hofl-a] eval @ round 3: accuracy=0.4817 task_loss=0.6990 t=4s
  [instrumented-hofl-a] completed in 5s
  [instrumented-hofl-b] starting: ho_fl.train --model opt --dataset sst2 --method hofl --beta 0.75 --rounds 3 --evaluate-every 3 --seed 42 --save-state-rounds 3 --model-path /content/work/p0-tiny/assets/tiny-opt --token-cache /content/work/p0-tiny/assets/tiny-sst2-cache.json --cache-dir /content/work/p0-tiny/assets/hf-cache-instrumented --device cpu --output /content/work/p0-tiny/golden/cdcaf6de099a-cpu-torch2.7.1/instrumented-hofl-b
    [instrumented-hofl-b] eval @ round 0: accuracy=0.4817 task_loss=0.6989 t=3s
    [instrumented-hofl-b] round 1 train_loss=0.6951 t=3s
    [instrumented-hofl-b] eval @ round 3: accuracy=0.4817 task_loss=0.6990 t=4s
  [instrumented-hofl-b] completed in 5s
  [baseline-hofl] starting: ho_fl.train --model opt --dataset sst2 --method hofl --beta 0.75 --rounds 3 --evaluate-every 3 --seed 42 --model-path /content/work/p0-tiny/assets/tiny-opt --token-cache /content/work/p0-tiny/assets/tiny-sst2-cache.json --cache-dir /content/work/p0-tiny/assets/hf-cache-baseline --device cpu --output /content/work/p0-tiny/golden/cdcaf6de099a-cpu-torch2.7.1/baseline-hofl
    [baseline-hofl] eval @ round 0: accuracy=0.4817 task_loss=0.6989 t=3s
    [baseline-hofl] round 1 train_loss=0.6951 t=3s
    [baseline-hofl] eval @ round 3: accuracy=0.4817 task_loss=0.6990 t=4s
  [baseline-hofl] completed in 5s
  [instrumented-fedavg-a] starting: ho_fl.train --model opt --dataset sst2 --method fedavg --rounds 3 --evaluate-every 3 --seed 42 --save-state-rounds 3 --model-path /content/work/p0-tiny/assets/tiny-opt --token-cache /content/work/p0-tiny/assets/tiny-sst2-cache.json --cache-dir /content/work/p0-tiny/assets/hf-cache-instrumented --device cpu --output /content/work/p0-tiny/golden/cdcaf6de099a-cpu-torch2.7.1/instrumented-fedavg-a
    [instrumented-fedavg-a] eval @ round 0: accuracy=0.4817 task_loss=0.6989 t=3s
    [instrumented-fedavg-a] round 1 train_loss=0.6998 t=3s
    [instrumented-fedavg-a] eval @ round 3: accuracy=0.4771 task_loss=0.6989 t=4s
  [instrumented-fedavg-a] completed in 5s
  [instrumented-fedavg-b] starting: ho_fl.train --model opt --dataset sst2 --method fedavg --rounds 3 --evaluate-every 3 --seed 42 --save-state-rounds 3 --model-path /content/work/p0-tiny/assets/tiny-opt --token-cache /content/work/p0-tiny/assets/tiny-sst2-cache.json --cache-dir /content/work/p0-tiny/assets/hf-cache-instrumented --device cpu --output /content/work/p0-tiny/golden/cdcaf6de099a-cpu-torch2.7.1/instrumented-fedavg-b
    [instrumented-fedavg-b] eval @ round 0: accuracy=0.4817 task_loss=0.6989 t=3s
    [instrumented-fedavg-b] round 1 train_loss=0.6998 t=3s
    [instrumented-fedavg-b] eval @ round 3: accuracy=0.4771 task_loss=0.6989 t=4s
  [instrumented-fedavg-b] completed in 5s
  [baseline-fedavg] starting: ho_fl.train --model opt --dataset sst2 --method fedavg --rounds 3 --evaluate-every 3 --seed 42 --model-path /content/work/p0-tiny/assets/tiny-opt --token-cache /content/work/p0-tiny/assets/tiny-sst2-cache.json --cache-dir /content/work/p0-tiny/assets/hf-cache-baseline --device cpu --output /content/work/p0-tiny/golden/cdcaf6de099a-cpu-torch2.7.1/baseline-fedavg
    [baseline-fedavg] eval @ round 0: accuracy=0.4817 task_loss=0.6989 t=3s
    [baseline-fedavg] round 1 train_loss=0.6998 t=3s
    [baseline-fedavg] eval @ round 3: accuracy=0.4771 task_loss=0.6989 t=4s
  [baseline-fedavg] completed in 5s
  [instrumented-decomfl-a] starting: ho_fl.train --model opt --dataset sst2 --method decomfl --rounds 3 --evaluate-every 3 --seed 42 --save-state-rounds 3 --model-path /content/work/p0-tiny/assets/tiny-opt --token-cache /content/work/p0-tiny/assets/tiny-sst2-cache.json --cache-dir /content/work/p0-tiny/assets/hf-cache-instrumented --device cpu --output /content/work/p0-tiny/golden/cdcaf6de099a-cpu-torch2.7.1/instrumented-decomfl-a
    [instrumented-decomfl-a] eval @ round 0: accuracy=0.4817 task_loss=0.6989 t=3s
    [instrumented-decomfl-a] round 1 train_loss=0.6998 t=4s
    [instrumented-decomfl-a] eval @ round 3: accuracy=0.4794 task_loss=0.6989 t=5s
  [instrumented-decomfl-a] completed in 6s
  [instrumented-decomfl-b] starting: ho_fl.train --model opt --dataset sst2 --method decomfl --rounds 3 --evaluate-every 3 --seed 42 --save-state-rounds 3 --model-path /content/work/p0-tiny/assets/tiny-opt --token-cache /content/work/p0-tiny/assets/tiny-sst2-cache.json --cache-dir /content/work/p0-tiny/assets/hf-cache-instrumented --device cpu --output /content/work/p0-tiny/golden/cdcaf6de099a-cpu-torch2.7.1/instrumented-decomfl-b
    [instrumented-decomfl-b] eval @ round 0: accuracy=0.4817 task_loss=0.6989 t=3s
    [instrumented-decomfl-b] round 1 train_loss=0.6998 t=4s
    [instrumented-decomfl-b] eval @ round 3: accuracy=0.4794 task_loss=0.6989 t=5s
  [instrumented-decomfl-b] completed in 6s
  [baseline-decomfl] starting: ho_fl.train --model opt --dataset sst2 --method decomfl --rounds 3 --evaluate-every 3 --seed 42 --model-path /content/work/p0-tiny/assets/tiny-opt --token-cache /content/work/p0-tiny/assets/tiny-sst2-cache.json --cache-dir /content/work/p0-tiny/assets/hf-cache-baseline --device cpu --output /content/work/p0-tiny/golden/cdcaf6de099a-cpu-torch2.7.1/baseline-decomfl
    [baseline-decomfl] eval @ round 0: accuracy=0.4817 task_loss=0.6989 t=3s
    [baseline-decomfl] round 1 train_loss=0.6998 t=4s
    [baseline-decomfl] eval @ round 3: accuracy=0.4794 task_loss=0.6989 t=5s
  [baseline-decomfl] completed in 6s
  [instrumented-replay] starting: ho_fl.replay --model-path /content/work/p0-tiny/assets/tiny-opt --token-cache /content/work/p0-tiny/assets/tiny-sst2-cache.json --cache-dir /content/work/p0-tiny/assets/hf-cache-instrumented --device cpu --output /content/work/p0-tiny/golden/cdcaf6de099a-cpu-torch2.7.1/instrumented-replay
  [instrumented-replay] completed in 5s

| config | repeat runs bitwise | max |state diff| between repeats | logs identical to upstream | final hash |
| --- | --- | --- | --- | --- |
| hofl | yes | 0.0000 | yes | b6f96f07a6fbf977 |
| fedavg | yes | 0.0000 | yes | d561590fc9f968ab |
| decomfl | yes | 0.0000 | yes | b812ad0e1a6243c5 |
replay on the instrumented revision: {'status': 'PASSED', 'all_bitwise_equal': False, 'comparisons': 672, 'max_abs_error': 3.637978807091713e-12, 'max_relative_l2_error': 6.056144850633646e-12, 'device_backend': 'cpu', 'torch_version': '2.7.1+cu128'}
golden mode: bitwise; written to /content/work/p0-tiny/golden/cdcaf6de099a-cpu-torch2.7.1/golden.json

== regress ==
  [head-hofl] starting: ho_fl.train --model opt --dataset sst2 --method hofl --beta 0.75 --rounds 3 --evaluate-every 3 --seed 42 --save-state-rounds 3 --model-path /content/work/p0-tiny/assets/tiny-opt --token-cache /content/work/p0-tiny/assets/tiny-sst2-cache.json --cache-dir /content/work/p0-tiny/assets/hf-cache-head --device cpu --output /content/work/p0-tiny/regress/cdea00b095d2-cpu-torch2.7.1/head-hofl
    [head-hofl] eval @ round 0: accuracy=0.4817 task_loss=0.6989 t=3s
    [head-hofl] round 1 train_loss=0.6951 t=3s
    [head-hofl] eval @ round 3: accuracy=0.4817 task_loss=0.6990 t=4s
  [head-hofl] completed in 5s
  [head-fedavg] starting: ho_fl.train --model opt --dataset sst2 --method fedavg --rounds 3 --evaluate-every 3 --seed 42 --save-state-rounds 3 --model-path /content/work/p0-tiny/assets/tiny-opt --token-cache /content/work/p0-tiny/assets/tiny-sst2-cache.json --cache-dir /content/work/p0-tiny/assets/hf-cache-head --device cpu --output /content/work/p0-tiny/regress/cdea00b095d2-cpu-torch2.7.1/head-fedavg
    [head-fedavg] eval @ round 0: accuracy=0.4817 task_loss=0.6989 t=3s
    [head-fedavg] round 1 train_loss=0.6998 t=3s
    [head-fedavg] eval @ round 3: accuracy=0.4771 task_loss=0.6989 t=4s
  [head-fedavg] completed in 5s
  [head-decomfl] starting: ho_fl.train --model opt --dataset sst2 --method decomfl --rounds 3 --evaluate-every 3 --seed 42 --save-state-rounds 3 --model-path /content/work/p0-tiny/assets/tiny-opt --token-cache /content/work/p0-tiny/assets/tiny-sst2-cache.json --cache-dir /content/work/p0-tiny/assets/hf-cache-head --device cpu --output /content/work/p0-tiny/regress/cdea00b095d2-cpu-torch2.7.1/head-decomfl
    [head-decomfl] eval @ round 0: accuracy=0.4817 task_loss=0.6989 t=3s
    [head-decomfl] round 1 train_loss=0.6998 t=4s
    [head-decomfl] eval @ round 3: accuracy=0.4794 task_loss=0.6989 t=5s
  [head-decomfl] completed in 6s
  [head-replay] starting: ho_fl.replay --model-path /content/work/p0-tiny/assets/tiny-opt --token-cache /content/work/p0-tiny/assets/tiny-sst2-cache.json --cache-dir /content/work/p0-tiny/assets/hf-cache-head --device cpu --output /content/work/p0-tiny/regress/cdea00b095d2-cpu-torch2.7.1/head-replay
  [head-replay] completed in 5s

| config | I1 verdict | losses equal | evaluations equal | final hash equal | max |state diff| | first difference |
| --- | --- | --- | --- | --- | --- | --- |
| hofl | PASS_BITWISE | yes | yes | yes | 0.0000 |  |
| fedavg | PASS_BITWISE | yes | yes | yes | 0.0000 |  |
| decomfl | PASS_BITWISE | yes | yes | yes | 0.0000 |  |
I2 (ho-fl-replay on the current tree): PASS {'status': 'PASSED', 'all_bitwise_equal': False, 'comparisons': 672, 'max_abs_error': 3.637978807091713e-12, 'max_relative_l2_error': 6.056144850633646e-12, 'device_backend': 'cpu', 'torch_version': '2.7.1+cu128'}

stage results: env=ok, golden=ok, regress=ok
$ /content/venv/bin/python experiments/p1/p1.py supply queue feedback --tiny --quiet --jobs 2 --root /content/work/p1-tiny
== supply (E1a, opt) ==
Supply map for opt: three tiers of 10 clients, cuts [3, 6, 9], depth 12.
S1, S2, S3 are the segments that start at the three cuts; width = first-order reports per round.

dep_round_v1 beta=0
  expected width S1 / S2 / S3: 2.00 (brief 2.00) / 4.00 (brief 4.00) / 6.00 (brief 6.00)
  effective sources (N_eff) S1 / S2 / S3: 10.00 (brief 10.00) / 20.00 (brief 20.00) / 30.00 (brief 30.00)
  S1 largest single share: 10.0% (brief 10.0%); S1 share of the 6 largest sources: 60.0% (brief 60.0%); largest presence of a client: 20.0% (brief 20.0%)
  simulated rounds with S1 width 0 / 1 / <=2 / <3: 5.44% (brief 6.40%) / 25.56% (brief 26.50%) / 68.19% (brief 69.20%) / 68.19%
  participation bias term N*sum((p_i-1/N)^2): 0.000 (brief 0.000); sampling-weighted d_ZO/q: 73728 (brief 73728)

dep_round_v1 beta=0.75
  expected width S1 / S2 / S3: 5.00 (brief 5.00) / 5.50 (brief 5.50) / 6.00 (brief 6.00)
  effective sources (N_eff) S1 / S2 / S3: 6.49 (brief 6.49) / 7.81 (brief 7.81) / 9.23 (brief 9.23)
  S1 largest single share: 16.0% (brief 16.0%); S1 share of the 6 largest sources: 96.0% (brief 96.0%); largest presence of a client: 80.0% (brief 80.0%)
  simulated rounds with S1 width 0 / 1 / <=2 / <3: 0.00% (brief 0.00%) / 0.00% (brief 0.01%) / 0.12% (brief 0.40%) / 0.12%
  participation bias term N*sum((p_i-1/N)^2): 2.250 (brief 2.250); sampling-weighted d_ZO/q: 46080 (brief 46080)

tier_equalised beta=0.75
  expected width S1 / S2 / S3: 5.00 (brief 5.00) / 5.50 (brief 5.50) / 6.00 (brief 6.00)
  effective sources (N_eff) S1 / S2 / S3: 10.00 (brief 10.00) / 11.98 (brief 11.98) / 14.12 (brief 14.12)
  S1 largest single share: 10.0% (brief 10.0%); S1 share of the 6 largest sources: 60.0% (brief 60.0%); largest presence of a client: 50.0% (brief 50.0%)
  simulated rounds with S1 width 0 / 1 / <=2 / <3: 0.00% (brief 0.00%) / 0.00% (brief 0.02%) / 0.50% (brief 0.40%) / 0.50%
  participation bias term N*sum((p_i-1/N)^2): 1.125 (brief 1.125); sampling-weighted d_ZO/q: 46080 (brief 46080)

tier_quota 3,1,2
  expected width S1 / S2 / S3: 3.00 (brief 3.00) / 4.00 (brief 4.00) / 6.00 (brief 6.00)
  effective sources (N_eff) S1 / S2 / S3: 10.00 (brief 10.00) / 16.00 (brief 16.00) / 25.71 (brief 25.71)
  S1 largest single share: 10.0% (brief 10.0%); S1 share of the 6 largest sources: 60.0% (brief 60.0%); largest presence of a client: 30.0% (brief 30.0%)
  simulated rounds with S1 width 0 / 1 / <=2 / <3: 0.00% (brief 0.00%) / 0.00% (brief 0.00%) / 0.00% (brief 0.00%) / 0.00%
  participation bias term N*sum((p_i-1/N)^2): 0.167 (brief 0.167); sampling-weighted d_ZO/q: 67584 (brief 67584)

tier_quota 4,1,1
  expected width S1 / S2 / S3: 4.00 (brief 4.00) / 5.00 (brief 5.00) / 6.00 (brief 6.00)
  effective sources (N_eff) S1 / S2 / S3: 10.00 (brief 10.00) / 14.71 (brief 14.71) / 20.00 (brief 20.00)
  S1 largest single share: 10.0% (brief 10.0%); S1 share of the 6 largest sources: 60.0% (brief 60.0%); largest presence of a client: 40.0% (brief 40.0%)
  simulated rounds with S1 width 0 / 1 / <=2 / <3: 0.00% (brief 0.00%) / 0.00% (brief 0.00%) / 0.00% (brief 0.00%) / 0.00%
  participation bias term N*sum((p_i-1/N)^2): 0.500 (brief 0.500); sampling-weighted d_ZO/q: 55296 (brief 55296)

tier_quota 2,2,2
  expected width S1 / S2 / S3: 2.00 / 4.00 / 6.00
  effective sources (N_eff) S1 / S2 / S3: 10.00 / 20.00 / 30.00
  S1 largest single share: 10.0%; S1 share of the 6 largest sources: 60.0%; largest presence of a client: 20.0%
  simulated rounds with S1 width 0 / 1 / <=2 / <3: 0.00% / 0.00% / 100.00% / 100.00%
  participation bias term N*sum((p_i-1/N)^2): 0.000; sampling-weighted d_ZO/q: 73728

== queue (opt, seeds [42], blocks [1, 2, 3], 12 runs of 3 rounds, up to 2 at a time, cpu-torch2.7.1) ==
  [preflight opt/hofl] starting: ho_fl.train --model opt --dataset ag_news --partition dirichlet_05 --rounds 3 --method hofl --beta 0.0 --layer-stats --seed 42 --probe-rounds 1 --evaluate-every 3 --model-path /content/work/p1-tiny/assets/tiny-opt --token-cache /content/work/p1-tiny/assets/tiny-ag_news-cache.json --cache-dir /content/work/p1-tiny/assets/hf-cache --device cpu --output /content/work/p1-tiny/preflight/cpu-torch2.7.1/opt/hofl
    [preflight opt/hofl] eval @ round 0: accuracy=0.2526 task_loss=1.3901 t=4s
    [preflight opt/hofl] round 1 train_loss=1.3917 t=5s
    [preflight opt/hofl] eval @ round 3: accuracy=0.2529 task_loss=1.3901 t=7s
  [preflight opt/hofl] completed in 7s
  opt/hofl: 0.6s per round, 0.8s per evaluation, 3s start-up, peak GPU memory 0.0 GiB
  [preflight opt/hofl-zo-off] starting: ho_fl.train --model opt --dataset ag_news --partition dirichlet_05 --rounds 3 --method hofl --beta 0.0 --zo-mode off --agg-electorate fo --agg-weight cohort --layer-stats --seed 42 --evaluate-every 3 --model-path /content/work/p1-tiny/assets/tiny-opt --token-cache /content/work/p1-tiny/assets/tiny-ag_news-cache.json --cache-dir /content/work/p1-tiny/assets/hf-cache --device cpu --output /content/work/p1-tiny/preflight/cpu-torch2.7.1/opt/hofl-zo-off
    [preflight opt/hofl-zo-off] eval @ round 0: accuracy=0.2526 task_loss=1.3901 t=4s
    [preflight opt/hofl-zo-off] round 1 train_loss=1.3917 t=4s
    [preflight opt/hofl-zo-off] eval @ round 3: accuracy=0.2528 task_loss=1.3901 t=6s
  [preflight opt/hofl-zo-off] completed in 6s
  opt/hofl-zo-off: 0.4s per round, 1.0s per evaluation, 3s start-up, peak GPU memory 0.0 GiB
  [preflight opt/fedavg] starting: ho_fl.train --model opt --dataset ag_news --partition dirichlet_05 --rounds 3 --method fedavg --seed 42 --evaluate-every 3 --model-path /content/work/p1-tiny/assets/tiny-opt --token-cache /content/work/p1-tiny/assets/tiny-ag_news-cache.json --cache-dir /content/work/p1-tiny/assets/hf-cache --device cpu --output /content/work/p1-tiny/preflight/cpu-torch2.7.1/opt/fedavg
    [preflight opt/fedavg] eval @ round 0: accuracy=0.2526 task_loss=1.3901 t=4s
    [preflight opt/fedavg] round 1 train_loss=1.3916 t=4s
    [preflight opt/fedavg] eval @ round 3: accuracy=0.2524 task_loss=1.3901 t=6s
  [preflight opt/fedavg] completed in 7s
  opt/fedavg: 0.4s per round, 1.0s per evaluation, 3s start-up, peak GPU memory 0.0 GiB
  [preflight opt/decomfl] starting: ho_fl.train --model opt --dataset ag_news --partition dirichlet_05 --rounds 3 --method decomfl --seed 42 --evaluate-every 3 --model-path /content/work/p1-tiny/assets/tiny-opt --token-cache /content/work/p1-tiny/assets/tiny-ag_news-cache.json --cache-dir /content/work/p1-tiny/assets/hf-cache --device cpu --output /content/work/p1-tiny/preflight/cpu-torch2.7.1/opt/decomfl
    [preflight opt/decomfl] eval @ round 0: accuracy=0.2526 task_loss=1.3901 t=4s
    [preflight opt/decomfl] round 1 train_loss=1.3918 t=5s
    [preflight opt/decomfl] eval @ round 3: accuracy=0.2530 task_loss=1.3901 t=7s
  [preflight opt/decomfl] completed in 8s
  opt/decomfl: 0.8s per round, 1.0s per evaluation, 3s start-up, peak GPU memory 0.0 GiB
  [opt-s42-A-b0-r3] starting: ho_fl.train --model opt --dataset ag_news --partition dirichlet_05 --rounds 3 --method hofl --beta 0.0 --layer-stats --seed 42 --probe-rounds 1,3 --model-path /content/work/p1-tiny/assets/tiny-opt --token-cache /content/work/p1-tiny/assets/tiny-ag_news-cache.json --cache-dir /content/work/p1-tiny/assets/hf-cache --device cpu --output /content/work/p1-tiny/runs/opt-s42-A-b0-r3
  [opt-s42-C-b0-r3] starting: ho_fl.train --model opt --dataset ag_news --partition dirichlet_05 --rounds 3 --method hofl --beta 0.0 --agg-electorate fo --agg-weight count --layer-stats --seed 42 --model-path /content/work/p1-tiny/assets/tiny-opt --token-cache /content/work/p1-tiny/assets/tiny-ag_news-cache.json --cache-dir /content/work/p1-tiny/assets/hf-cache --device cpu --output /content/work/p1-tiny/runs/opt-s42-C-b0-r3
    [opt-s42-A-b0-r3] eval @ round 0: accuracy=0.2526 task_loss=1.3901 t=4s
    [opt-s42-A-b0-r3] round 1 train_loss=1.3917 t=5s
    [opt-s42-C-b0-r3] eval @ round 0: accuracy=0.2526 task_loss=1.3901 t=4s
    [opt-s42-A-b0-r3] eval @ round 3: accuracy=0.2529 task_loss=1.3901 t=7s
    [opt-s42-C-b0-r3] round 1 train_loss=1.3917 t=4s
  [opt-s42-A-b0-r3] completed in 8s
  [opt-s42-A-b0.75-r3] starting: ho_fl.train --model opt --dataset ag_news --partition dirichlet_05 --rounds 3 --method hofl --beta 0.75 --layer-stats --seed 42 --probe-rounds 1,3 --model-path /content/work/p1-tiny/assets/tiny-opt --token-cache /content/work/p1-tiny/assets/tiny-ag_news-cache.json --cache-dir /content/work/p1-tiny/assets/hf-cache --device cpu --output /content/work/p1-tiny/runs/opt-s42-A-b0.75-r3
    [opt-s42-C-b0-r3] eval @ round 3: accuracy=0.2532 task_loss=1.3901 t=6s
  [opt-s42-C-b0-r3] completed in 7s
  [opt-s42-Ax2-b0-r3] starting: ho_fl.train --model opt --dataset ag_news --partition dirichlet_05 --rounds 3 --method hofl --beta 0.0 --server-lr 2.0 --layer-stats --seed 42 --model-path /content/work/p1-tiny/assets/tiny-opt --token-cache /content/work/p1-tiny/assets/tiny-ag_news-cache.json --cache-dir /content/work/p1-tiny/assets/hf-cache --device cpu --output /content/work/p1-tiny/runs/opt-s42-Ax2-b0-r3
    [opt-s42-A-b0.75-r3] eval @ round 0: accuracy=0.2526 task_loss=1.3901 t=4s
    [opt-s42-A-b0.75-r3] round 1 train_loss=1.3981 t=5s
    [opt-s42-Ax2-b0-r3] eval @ round 0: accuracy=0.2526 task_loss=1.3901 t=4s
    [opt-s42-A-b0.75-r3] eval @ round 3: accuracy=0.2525 task_loss=1.3901 t=7s
    [opt-s42-Ax2-b0-r3] round 1 train_loss=1.3917 t=4s
  [opt-s42-A-b0.75-r3] completed in 8s
  [opt-s42-Ax3-b0-r3] starting: ho_fl.train --model opt --dataset ag_news --partition dirichlet_05 --rounds 3 --method hofl --beta 0.0 --server-lr 3.0 --layer-stats --seed 42 --model-path /content/work/p1-tiny/assets/tiny-opt --token-cache /content/work/p1-tiny/assets/tiny-ag_news-cache.json --cache-dir /content/work/p1-tiny/assets/hf-cache --device cpu --output /content/work/p1-tiny/runs/opt-s42-Ax3-b0-r3
    [opt-s42-Ax2-b0-r3] eval @ round 3: accuracy=0.2528 task_loss=1.3901 t=6s
  [opt-s42-Ax2-b0-r3] completed in 7s
  [opt-s42-B-b0-r3] starting: ho_fl.train --model opt --dataset ag_news --partition dirichlet_05 --rounds 3 --method hofl --beta 0.0 --zo-mode off --agg-electorate fo --agg-weight cohort --layer-stats --seed 42 --model-path /content/work/p1-tiny/assets/tiny-opt --token-cache /content/work/p1-tiny/assets/tiny-ag_news-cache.json --cache-dir /content/work/p1-tiny/assets/hf-cache --device cpu --output /content/work/p1-tiny/runs/opt-s42-B-b0-r3
    [opt-s42-Ax3-b0-r3] eval @ round 0: accuracy=0.2526 task_loss=1.3901 t=4s
    [opt-s42-Ax3-b0-r3] round 1 train_loss=1.3917 t=4s
    [opt-s42-Ax3-b0-r3] eval @ round 3: accuracy=0.2530 task_loss=1.3901 t=6s
    [opt-s42-B-b0-r3] eval @ round 0: accuracy=0.2526 task_loss=1.3901 t=4s
  [opt-s42-Ax3-b0-r3] completed in 7s
  [opt-s42-B-b0.75-r3] starting: ho_fl.train --model opt --dataset ag_news --partition dirichlet_05 --rounds 3 --method hofl --beta 0.75 --zo-mode off --agg-electorate fo --agg-weight cohort --layer-stats --seed 42 --model-path /content/work/p1-tiny/assets/tiny-opt --token-cache /content/work/p1-tiny/assets/tiny-ag_news-cache.json --cache-dir /content/work/p1-tiny/assets/hf-cache --device cpu --output /content/work/p1-tiny/runs/opt-s42-B-b0.75-r3
    [opt-s42-B-b0-r3] round 1 train_loss=1.3917 t=4s
    [opt-s42-B-b0-r3] eval @ round 3: accuracy=0.2528 task_loss=1.3901 t=6s
  [opt-s42-B-b0-r3] completed in 6s
  [opt-s42-fedavg-r3] starting: ho_fl.train --model opt --dataset ag_news --partition dirichlet_05 --rounds 3 --method fedavg --seed 42 --model-path /content/work/p1-tiny/assets/tiny-opt --token-cache /content/work/p1-tiny/assets/tiny-ag_news-cache.json --cache-dir /content/work/p1-tiny/assets/hf-cache --device cpu --output /content/work/p1-tiny/runs/opt-s42-fedavg-r3
    [opt-s42-B-b0.75-r3] eval @ round 0: accuracy=0.2526 task_loss=1.3901 t=4s
    [opt-s42-B-b0.75-r3] round 1 train_loss=1.3981 t=4s
    [opt-s42-B-b0.75-r3] eval @ round 3: accuracy=0.2520 task_loss=1.3901 t=6s
  [opt-s42-B-b0.75-r3] completed in 7s
  [opt-s42-decomfl-r3] starting: ho_fl.train --model opt --dataset ag_news --partition dirichlet_05 --rounds 3 --method decomfl --seed 42 --model-path /content/work/p1-tiny/assets/tiny-opt --token-cache /content/work/p1-tiny/assets/tiny-ag_news-cache.json --cache-dir /content/work/p1-tiny/assets/hf-cache --device cpu --output /content/work/p1-tiny/runs/opt-s42-decomfl-r3
    [opt-s42-fedavg-r3] eval @ round 0: accuracy=0.2526 task_loss=1.3901 t=4s
    [opt-s42-fedavg-r3] round 1 train_loss=1.3916 t=4s
    [opt-s42-fedavg-r3] eval @ round 3: accuracy=0.2524 task_loss=1.3901 t=6s
  [opt-s42-fedavg-r3] completed in 7s
  [opt-s42-C-b0.75-r3] starting: ho_fl.train --model opt --dataset ag_news --partition dirichlet_05 --rounds 3 --method hofl --beta 0.75 --agg-electorate fo --agg-weight count --layer-stats --seed 42 --model-path /content/work/p1-tiny/assets/tiny-opt --token-cache /content/work/p1-tiny/assets/tiny-ag_news-cache.json --cache-dir /content/work/p1-tiny/assets/hf-cache --device cpu --output /content/work/p1-tiny/runs/opt-s42-C-b0.75-r3
    [opt-s42-decomfl-r3] eval @ round 0: accuracy=0.2526 task_loss=1.3901 t=4s
    [opt-s42-decomfl-r3] round 1 train_loss=1.3918 t=5s
    [opt-s42-C-b0.75-r3] eval @ round 0: accuracy=0.2526 task_loss=1.3901 t=4s
    [opt-s42-decomfl-r3] eval @ round 3: accuracy=0.2530 task_loss=1.3901 t=7s
    [opt-s42-C-b0.75-r3] round 1 train_loss=1.3981 t=4s
  [opt-s42-decomfl-r3] completed in 8s
  [opt-s42-D-b0-r3] starting: ho_fl.train --model opt --dataset ag_news --partition dirichlet_05 --rounds 3 --method hofl --beta 0.0 --zo-mode off --agg-electorate fo --agg-weight count --layer-stats --seed 42 --model-path /content/work/p1-tiny/assets/tiny-opt --token-cache /content/work/p1-tiny/assets/tiny-ag_news-cache.json --cache-dir /content/work/p1-tiny/assets/hf-cache --device cpu --output /content/work/p1-tiny/runs/opt-s42-D-b0-r3
    [opt-s42-C-b0.75-r3] eval @ round 3: accuracy=0.2528 task_loss=1.3901 t=6s
  [opt-s42-C-b0.75-r3] completed in 7s
  [opt-s42-D-b0.75-r3] starting: ho_fl.train --model opt --dataset ag_news --partition dirichlet_05 --rounds 3 --method hofl --beta 0.75 --zo-mode off --agg-electorate fo --agg-weight count --layer-stats --seed 42 --model-path /content/work/p1-tiny/assets/tiny-opt --token-cache /content/work/p1-tiny/assets/tiny-ag_news-cache.json --cache-dir /content/work/p1-tiny/assets/hf-cache --device cpu --output /content/work/p1-tiny/runs/opt-s42-D-b0.75-r3
    [opt-s42-D-b0-r3] eval @ round 0: accuracy=0.2526 task_loss=1.3901 t=4s
    [opt-s42-D-b0-r3] round 1 train_loss=1.3917 t=4s
    [opt-s42-D-b0-r3] eval @ round 3: accuracy=0.2521 task_loss=1.3901 t=6s
  [opt-s42-D-b0-r3] completed in 6s
    [opt-s42-D-b0.75-r3] eval @ round 0: accuracy=0.2526 task_loss=1.3901 t=4s
    [opt-s42-D-b0.75-r3] round 1 train_loss=1.3981 t=4s
    [opt-s42-D-b0.75-r3] eval @ round 3: accuracy=0.2522 task_loss=1.3901 t=6s
  [opt-s42-D-b0.75-r3] completed in 6s

| run | status | wall min | estimate min | note |
| --- | --- | --- | --- | --- |
| opt-s42-A-b0-r3 | COMPLETED | 0.1 | 0.3 |  |
| opt-s42-C-b0-r3 | COMPLETED | 0.1 | 0.3 |  |
| opt-s42-A-b0.75-r3 | COMPLETED | 0.1 | 0.2 |  |
| opt-s42-Ax2-b0-r3 | COMPLETED | 0.1 | 0.1 |  |
| opt-s42-Ax3-b0-r3 | COMPLETED | 0.1 | 0.1 |  |
| opt-s42-B-b0-r3 | COMPLETED | 0.1 | 0.1 |  |
| opt-s42-B-b0.75-r3 | COMPLETED | 0.1 | 0.1 |  |
| opt-s42-fedavg-r3 | COMPLETED | 0.1 | 0.1 |  |
| opt-s42-decomfl-r3 | COMPLETED | 0.1 | 0.2 |  |
| opt-s42-C-b0.75-r3 | COMPLETED | 0.1 | 0.1 |  |
| opt-s42-D-b0-r3 | COMPLETED | 0.1 | 0.1 |  |
| opt-s42-D-b0.75-r3 | COMPLETED | 0.1 | 0.1 |  |

== feedback ==
H1a: CANNOT_JUDGE; H1: b0 CANNOT_JUDGE, b0.75 CANNOT_JUDGE; 12 of 12 runs completed
written: /content/work/p1-tiny/feedback/feedback_P1.md, CSV files and /content/work/p1-tiny/feedback/feedback_P1_bundle.zip

stage results: supply=ok, queue=ok, feedback=ok

dry run ok, 5.44 h of budget left





$ /content/venv/bin/python experiments/p0/p0.py env --root /content/work/p0 --cache-dir /content/cache --device cuda
== env ==
{
  "python": "3.11.18",
  "torch": "2.7.1+cu128",
  "cuda_runtime": "12.8",
  "cuda_available": true,
  "gpu_name": "NVIDIA RTX PRO 6000 Blackwell Server Edition",
  "driver_version": "580.82.07",
  "bf16_matmul_finite": true,
  "mismatched": {
    "torch": {
      "installed": "2.7.1+cu128",
      "pinned": "2.6.0"
    }
  },
  "git": {
    "head": "cdea00b095d26a14a71a2100d3f4ef358ec2dd62",
    "branch": "HEAD"
  }
}
WARNING: installed package versions differ from pyproject.toml (see 'mismatched')

stage results: env=ok
$ /content/venv/bin/python experiments/p0/p0.py tests --root /content/work/p0 --cache-dir /content/cache --device cuda
== tests ==
..................................................................FF.... [ 34%]
........................................................................ [ 69%]
................................................................         [100%]
=================================== FAILURES ===================================
____ test_queue_runs_several_jobs_at_a_time_and_budgets_for_the_worst_case _____

tmp_path = PosixPath('/tmp/pytest-of-root/pytest-0/test_queue_runs_several_jobs_a0')

    def test_queue_runs_several_jobs_at_a_time_and_budgets_for_the_worst_case(tmp_path):
        runner = FakeRunner(duration=1.5)  # longer than the half-second poll that hands the start lock on
        record = queue(tmp_path, runner, make_specs(2) + [dict(make_specs(4)[3])], jobs=2)
>       assert runner.max_active == 2 and record["jobs"] == 2 and [entry["status"] for entry in record["queue"]] == ["COMPLETED"] * 3
E       assert (1 == 2)
E        +  where 1 = <test_grid_and_analysis.FakeRunner object at 0x7b6d68f0acd0>.max_active

tests/test_grid_and_analysis.py:86: AssertionError
____________________ test_memory_limits_the_number_of_jobs _____________________

    def test_memory_limits_the_number_of_jobs():
        timings = {"a": {"peak_mb": 30000.0}, "b": {"peak_mb": 10000.0}}
        assert grid.allowed_jobs(4, timings, {"a", "b"}, memory=(0.0, 96000.0)) == 2
        assert grid.allowed_jobs(4, timings, {"b"}, memory=(0.0, 96000.0)) == 4
        assert grid.allowed_jobs(4, timings, {"a"}, memory=(0.0, 24000.0)) == 1
        assert grid.allowed_jobs(3, {"a": {"peak_mb": None}}, {"a"}, memory=(0.0, 96000.0)) == 1  # a GPU without a measurement
>       assert grid.allowed_jobs(3, timings, {"a"}, memory=None) == 3 and grid.allowed_jobs(1, timings, {"a"}, memory=(0.0, 1.0)) == 1
E       AssertionError: assert (2 == 3)
E        +  where 2 = <function allowed_jobs at 0x7b6d76fa3380>(3, {'a': {'peak_mb': 30000.0}, 'b': {'peak_mb': 10000.0}}, {'a'}, memory=None)
E        +    where <function allowed_jobs at 0x7b6d76fa3380> = grid.allowed_jobs

tests/test_grid_and_analysis.py:99: AssertionError
=========================== short test summary info ============================
FAILED tests/test_grid_and_analysis.py::test_queue_runs_several_jobs_at_a_time_and_budgets_for_the_worst_case
FAILED tests/test_grid_and_analysis.py::test_memory_limits_the_number_of_jobs
2 failed, 206 passed, 3 deselected in 71.56s (0:01:11)
pytest exit code 1 in 73s

stage results: tests=ATTENTION
unit tests: FAILURES (listed above)



$ /content/venv/bin/python experiments/p0/p0.py golden --root /content/work/p0 --cache-dir /content/cache --device cuda
== golden ==
  [instrumented-hofl-a] starting: ho_fl.train --model opt --dataset sst2 --method hofl --beta 0.75 --rounds 3 --evaluate-every 3 --seed 42 --save-state-rounds 3 --device cuda --cache-dir /content/cache-instrumented --output /content/work/p0/golden/cdcaf6de099a-nvidia-rtx-pro-6000-blackwell-server-edition-torch2.7.1/instrumented-hofl-a
    [instrumented-hofl-a] eval @ round 0: accuracy=0.5172 task_loss=0.9131 t=12s
    [instrumented-hofl-a] round 1 train_loss=0.7539 t=12s
    [instrumented-hofl-a] eval @ round 3: accuracy=0.5252 task_loss=0.8255 t=13s
  [instrumented-hofl-a] completed in 14s
  [instrumented-hofl-b] starting: ho_fl.train --model opt --dataset sst2 --method hofl --beta 0.75 --rounds 3 --evaluate-every 3 --seed 42 --save-state-rounds 3 --device cuda --cache-dir /content/cache-instrumented --output /content/work/p0/golden/cdcaf6de099a-nvidia-rtx-pro-6000-blackwell-server-edition-torch2.7.1/instrumented-hofl-b
    [instrumented-hofl-b] eval @ round 0: accuracy=0.5172 task_loss=0.9131 t=4s
    [instrumented-hofl-b] round 1 train_loss=0.7539 t=4s
    [instrumented-hofl-b] eval @ round 3: accuracy=0.5252 task_loss=0.8255 t=5s
  [instrumented-hofl-b] completed in 6s
  [baseline-hofl] starting: ho_fl.train --model opt --dataset sst2 --method hofl --beta 0.75 --rounds 3 --evaluate-every 3 --seed 42 --device cuda --cache-dir /content/cache-baseline --output /content/work/p0/golden/cdcaf6de099a-nvidia-rtx-pro-6000-blackwell-server-edition-torch2.7.1/baseline-hofl
    [baseline-hofl] eval @ round 0: accuracy=0.5172 task_loss=0.9131 t=10s
    [baseline-hofl] round 1 train_loss=0.7539 t=11s
    [baseline-hofl] eval @ round 3: accuracy=0.5252 task_loss=0.8255 t=11s
  [baseline-hofl] completed in 12s
  [instrumented-fedavg-a] starting: ho_fl.train --model opt --dataset sst2 --method fedavg --rounds 3 --evaluate-every 3 --seed 42 --save-state-rounds 3 --device cuda --cache-dir /content/cache-instrumented --output /content/work/p0/golden/cdcaf6de099a-nvidia-rtx-pro-6000-blackwell-server-edition-torch2.7.1/instrumented-fedavg-a
    [instrumented-fedavg-a] eval @ round 0: accuracy=0.5172 task_loss=0.9131 t=4s
    [instrumented-fedavg-a] round 1 train_loss=0.7305 t=4s
    [instrumented-fedavg-a] eval @ round 3: accuracy=0.5287 task_loss=0.8168 t=5s
  [instrumented-fedavg-a] completed in 5s
  [instrumented-fedavg-b] starting: ho_fl.train --model opt --dataset sst2 --method fedavg --rounds 3 --evaluate-every 3 --seed 42 --save-state-rounds 3 --device cuda --cache-dir /content/cache-instrumented --output /content/work/p0/golden/cdcaf6de099a-nvidia-rtx-pro-6000-blackwell-server-edition-torch2.7.1/instrumented-fedavg-b
    [instrumented-fedavg-b] eval @ round 0: accuracy=0.5172 task_loss=0.9131 t=4s
    [instrumented-fedavg-b] round 1 train_loss=0.7305 t=4s
    [instrumented-fedavg-b] eval @ round 3: accuracy=0.5287 task_loss=0.8168 t=5s
  [instrumented-fedavg-b] completed in 5s
  [baseline-fedavg] starting: ho_fl.train --model opt --dataset sst2 --method fedavg --rounds 3 --evaluate-every 3 --seed 42 --device cuda --cache-dir /content/cache-baseline --output /content/work/p0/golden/cdcaf6de099a-nvidia-rtx-pro-6000-blackwell-server-edition-torch2.7.1/baseline-fedavg
    [baseline-fedavg] eval @ round 0: accuracy=0.5172 task_loss=0.9131 t=4s
    [baseline-fedavg] round 1 train_loss=0.7305 t=4s
    [baseline-fedavg] eval @ round 3: accuracy=0.5287 task_loss=0.8168 t=5s
  [baseline-fedavg] completed in 5s
  [instrumented-decomfl-a] starting: ho_fl.train --model opt --dataset sst2 --method decomfl --rounds 3 --evaluate-every 3 --seed 42 --save-state-rounds 3 --device cuda --cache-dir /content/cache-instrumented --output /content/work/p0/golden/cdcaf6de099a-nvidia-rtx-pro-6000-blackwell-server-edition-torch2.7.1/instrumented-decomfl-a
    [instrumented-decomfl-a] eval @ round 0: accuracy=0.5172 task_loss=0.9131 t=4s
    [instrumented-decomfl-a] round 1 train_loss=0.7396 t=4s
    [instrumented-decomfl-a] eval @ round 3: accuracy=0.5172 task_loss=0.9122 t=6s
  [instrumented-decomfl-a] completed in 6s
  [instrumented-decomfl-b] starting: ho_fl.train --model opt --dataset sst2 --method decomfl --rounds 3 --evaluate-every 3 --seed 42 --save-state-rounds 3 --device cuda --cache-dir /content/cache-instrumented --output /content/work/p0/golden/cdcaf6de099a-nvidia-rtx-pro-6000-blackwell-server-edition-torch2.7.1/instrumented-decomfl-b
    [instrumented-decomfl-b] eval @ round 0: accuracy=0.5172 task_loss=0.9131 t=4s
    [instrumented-decomfl-b] round 1 train_loss=0.7396 t=4s
    [instrumented-decomfl-b] eval @ round 3: accuracy=0.5172 task_loss=0.9122 t=6s
  [instrumented-decomfl-b] completed in 6s
  [baseline-decomfl] starting: ho_fl.train --model opt --dataset sst2 --method decomfl --rounds 3 --evaluate-every 3 --seed 42 --device cuda --cache-dir /content/cache-baseline --output /content/work/p0/golden/cdcaf6de099a-nvidia-rtx-pro-6000-blackwell-server-edition-torch2.7.1/baseline-decomfl
    [baseline-decomfl] eval @ round 0: accuracy=0.5172 task_loss=0.9131 t=4s
    [baseline-decomfl] round 1 train_loss=0.7396 t=4s
    [baseline-decomfl] eval @ round 3: accuracy=0.5172 task_loss=0.9122 t=6s
  [baseline-decomfl] completed in 6s
  [instrumented-replay] starting: ho_fl.replay --device cuda --cache-dir /content/cache-instrumented --output /content/work/p0/golden/cdcaf6de099a-nvidia-rtx-pro-6000-blackwell-server-edition-torch2.7.1/instrumented-replay
  [instrumented-replay] completed in 7s

| config | repeat runs bitwise | max |state diff| between repeats | logs identical to upstream | final hash |
| --- | --- | --- | --- | --- |
| hofl | yes | 0.0000 | yes | 1afdad12c2b24867 |
| fedavg | yes | 0.0000 | yes | c6bd7da1c424c6b9 |
| decomfl | yes | 0.0000 | yes | 61c5c2d874375111 |
replay on the instrumented revision: {'status': 'PASSED', 'all_bitwise_equal': False, 'comparisons': 672, 'max_abs_error': 3.637978807091713e-12, 'max_relative_l2_error': 2.8487253938003907e-11, 'device_backend': 'cuda', 'torch_version': '2.7.1+cu128'}
golden mode: bitwise; written to /content/work/p0/golden/cdcaf6de099a-nvidia-rtx-pro-6000-blackwell-server-edition-torch2.7.1/golden.json

stage results: golden=ok
$ /content/venv/bin/python experiments/p0/p0.py regress --root /content/work/p0 --cache-dir /content/cache --device cuda
== regress ==
  [head-hofl] starting: ho_fl.train --model opt --dataset sst2 --method hofl --beta 0.75 --rounds 3 --evaluate-every 3 --seed 42 --save-state-rounds 3 --device cuda --cache-dir /content/cache-head --output /content/work/p0/regress/cdea00b095d2-nvidia-rtx-pro-6000-blackwell-server-edition-torch2.7.1/head-hofl
    [head-hofl] eval @ round 0: accuracy=0.5172 task_loss=0.9131 t=10s
    [head-hofl] round 1 train_loss=0.7539 t=10s
    [head-hofl] eval @ round 3: accuracy=0.5252 task_loss=0.8255 t=11s
  [head-hofl] completed in 12s
  [head-fedavg] starting: ho_fl.train --model opt --dataset sst2 --method fedavg --rounds 3 --evaluate-every 3 --seed 42 --save-state-rounds 3 --device cuda --cache-dir /content/cache-head --output /content/work/p0/regress/cdea00b095d2-nvidia-rtx-pro-6000-blackwell-server-edition-torch2.7.1/head-fedavg
    [head-fedavg] eval @ round 0: accuracy=0.5172 task_loss=0.9131 t=4s
    [head-fedavg] round 1 train_loss=0.7305 t=4s
    [head-fedavg] eval @ round 3: accuracy=0.5287 task_loss=0.8168 t=5s
  [head-fedavg] completed in 5s
  [head-decomfl] starting: ho_fl.train --model opt --dataset sst2 --method decomfl --rounds 3 --evaluate-every 3 --seed 42 --save-state-rounds 3 --device cuda --cache-dir /content/cache-head --output /content/work/p0/regress/cdea00b095d2-nvidia-rtx-pro-6000-blackwell-server-edition-torch2.7.1/head-decomfl
    [head-decomfl] eval @ round 0: accuracy=0.5172 task_loss=0.9131 t=4s
    [head-decomfl] round 1 train_loss=0.7396 t=4s
    [head-decomfl] eval @ round 3: accuracy=0.5172 task_loss=0.9122 t=6s
  [head-decomfl] completed in 6s
  [head-replay] starting: ho_fl.replay --device cuda --cache-dir /content/cache-head --output /content/work/p0/regress/cdea00b095d2-nvidia-rtx-pro-6000-blackwell-server-edition-torch2.7.1/head-replay
  [head-replay] completed in 7s

| config | I1 verdict | losses equal | evaluations equal | final hash equal | max |state diff| | first difference |
| --- | --- | --- | --- | --- | --- | --- |
| hofl | PASS_BITWISE | yes | yes | yes | 0.0000 |  |
| fedavg | PASS_BITWISE | yes | yes | yes | 0.0000 |  |
| decomfl | PASS_BITWISE | yes | yes | yes | 0.0000 |  |
I2 (ho-fl-replay on the current tree): PASS {'status': 'PASSED', 'all_bitwise_equal': False, 'comparisons': 672, 'max_abs_error': 3.637978807091713e-12, 'max_relative_l2_error': 2.8487253938003907e-11, 'device_backend': 'cuda', 'torch_version': '2.7.1+cu128'}

stage results: regress=ok
$ /content/venv/bin/python experiments/p0/p0.py feedback --root /content/work/p0 --cache-dir /content/cache --device cuda --quiet
== feedback ==
checklist: tests=FAIL, I1=PASS, I2=PASS, E0=NOT_RUN, G0=BLOCKED
written: /content/work/p0/feedback/feedback_P0.md, /content/work/p0/feedback/feedback_P0.json, /content/work/p0/feedback/feedback_P0_bundle.zip

stage results: feedback=ATTENTION

golden: ok | regression (I1, I2): ok
5.39 h of budget left



$ /content/venv/bin/python experiments/p1/p1.py supply --root /content/work/p1 --cache-dir /content/cache --device cuda --model qwen --seeds 42
== supply (E1a, qwen) ==
Supply map for qwen: three tiers of 10 clients, cuts [8, 16, 24], depth 28.
S1, S2, S3 are the segments that start at the three cuts; width = first-order reports per round.

dep_round_v1 beta=0
  expected width S1 / S2 / S3: 2.00 (brief 2.00) / 4.00 (brief 4.00) / 6.00 (brief 6.00)
  effective sources (N_eff) S1 / S2 / S3: 10.00 (brief 10.00) / 20.00 (brief 20.00) / 30.00 (brief 30.00)
  S1 largest single share: 10.0% (brief 10.0%); S1 share of the 6 largest sources: 60.0% (brief 60.0%); largest presence of a client: 20.0% (brief 20.0%)
  simulated rounds with S1 width 0 / 1 / <=2 / <3: 6.51% (brief 6.40%) / 25.90% (brief 26.50%) / 69.14% (brief 69.20%) / 69.14%
  participation bias term N*sum((p_i-1/N)^2): 0.000 (brief 0.000); sampling-weighted d_ZO/q: 311296 (brief 311296)

dep_round_v1 beta=0.75
  expected width S1 / S2 / S3: 5.00 (brief 5.00) / 5.50 (brief 5.50) / 6.00 (brief 6.00)
  effective sources (N_eff) S1 / S2 / S3: 6.49 (brief 6.49) / 7.81 (brief 7.81) / 9.23 (brief 9.23)
  S1 largest single share: 16.0% (brief 16.0%); S1 share of the 6 largest sources: 96.0% (brief 96.0%); largest presence of a client: 80.0% (brief 80.0%)
  simulated rounds with S1 width 0 / 1 / <=2 / <3: 0.00% (brief 0.00%) / 0.04% (brief 0.01%) / 0.41% (brief 0.40%) / 0.41%
  participation bias term N*sum((p_i-1/N)^2): 2.250 (brief 2.250); sampling-weighted d_ZO/q: 194560 (brief 194560)

tier_equalised beta=0.75
  expected width S1 / S2 / S3: 5.00 (brief 5.00) / 5.50 (brief 5.50) / 6.00 (brief 6.00)
  effective sources (N_eff) S1 / S2 / S3: 10.00 (brief 10.00) / 11.98 (brief 11.98) / 14.12 (brief 14.12)
  S1 largest single share: 10.0% (brief 10.0%); S1 share of the 6 largest sources: 60.0% (brief 60.0%); largest presence of a client: 50.0% (brief 50.0%)
  simulated rounds with S1 width 0 / 1 / <=2 / <3: 0.00% (brief 0.00%) / 0.02% (brief 0.02%) / 0.41% (brief 0.40%) / 0.41%
  participation bias term N*sum((p_i-1/N)^2): 1.125 (brief 1.125); sampling-weighted d_ZO/q: 194560 (brief 194560)

tier_quota 3,1,2
  expected width S1 / S2 / S3: 3.00 (brief 3.00) / 4.00 (brief 4.00) / 6.00 (brief 6.00)
  effective sources (N_eff) S1 / S2 / S3: 10.00 (brief 10.00) / 16.00 (brief 16.00) / 25.71 (brief 25.71)
  S1 largest single share: 10.0% (brief 10.0%); S1 share of the 6 largest sources: 60.0% (brief 60.0%); largest presence of a client: 30.0% (brief 30.0%)
  simulated rounds with S1 width 0 / 1 / <=2 / <3: 0.00% (brief 0.00%) / 0.00% (brief 0.00%) / 0.00% (brief 0.00%) / 0.00%
  participation bias term N*sum((p_i-1/N)^2): 0.167 (brief 0.167); sampling-weighted d_ZO/q: 285355 (brief 285355)

tier_quota 4,1,1
  expected width S1 / S2 / S3: 4.00 (brief 4.00) / 5.00 (brief 5.00) / 6.00 (brief 6.00)
  effective sources (N_eff) S1 / S2 / S3: 10.00 (brief 10.00) / 14.71 (brief 14.71) / 20.00 (brief 20.00)
  S1 largest single share: 10.0% (brief 10.0%); S1 share of the 6 largest sources: 60.0% (brief 60.0%); largest presence of a client: 40.0% (brief 40.0%)
  simulated rounds with S1 width 0 / 1 / <=2 / <3: 0.00% (brief 0.00%) / 0.00% (brief 0.00%) / 0.00% (brief 0.00%) / 0.00%
  participation bias term N*sum((p_i-1/N)^2): 0.500 (brief 0.500); sampling-weighted d_ZO/q: 233472 (brief 233472)

tier_quota 2,2,2
  expected width S1 / S2 / S3: 2.00 / 4.00 / 6.00
  effective sources (N_eff) S1 / S2 / S3: 10.00 / 20.00 / 30.00
  S1 largest single share: 10.0%; S1 share of the 6 largest sources: 60.0%; largest presence of a client: 20.0%
  simulated rounds with S1 width 0 / 1 / <=2 / <3: 0.00% / 0.00% / 100.00% / 100.00%
  participation bias term N*sum((p_i-1/N)^2): 0.000; sampling-weighted d_ZO/q: 311296

stage results: supply=ok


$ /content/venv/bin/python experiments/p1/p1.py queue --root /content/work/p1 --cache-dir /content/cache --device cuda --model qwen --seeds 42 --blocks 1 --jobs 2 --deadline 1795233849.7437687
== queue (qwen, seeds [42], blocks [1], 5 runs of 160 rounds, up to 2 at a time, nvidia-rtx-pro-6000-blackwell-server-edition-torch2.7.1) ==
  [preflight qwen/hofl] starting: ho_fl.train --model qwen --dataset ag_news --partition dirichlet_05 --rounds 3 --method hofl --beta 0.0 --layer-stats --seed 42 --probe-rounds 1 --evaluate-every 3 --device cuda --cache-dir /content/cache --output /content/work/p1/preflight/nvidia-rtx-pro-6000-blackwell-server-edition-torch2.7.1/qwen/hofl
    [preflight qwen/hofl] eval @ round 0: accuracy=0.5579 task_loss=1.1715 t=43s
    [preflight qwen/hofl] round 1 train_loss=1.1488 t=48s
    [preflight qwen/hofl] eval @ round 3: accuracy=0.5601 task_loss=1.1705 t=65s
  [preflight qwen/hofl] completed in 66s
  qwen/hofl: 3.0s per round, 11.8s per evaluation, 31s start-up, peak GPU memory 9.8 GiB
  [qwen-s42-A-b0] starting: ho_fl.train --model qwen --dataset ag_news --partition dirichlet_05 --rounds 160 --method hofl --beta 0.0 --layer-stats --seed 42 --probe-rounds 1,20,40,80,160 --device cuda --cache-dir /content/cache --output /content/work/p1/runs/qwen-s42-A-b0
  [qwen-s42-C-b0] starting: ho_fl.train --model qwen --dataset ag_news --partition dirichlet_05 --rounds 160 --method hofl --beta 0.0 --agg-electorate fo --agg-weight count --layer-stats --seed 42 --device cuda --cache-dir /content/cache --output /content/work/p1/runs/qwen-s42-C-b0
    [qwen-s42-A-b0] eval @ round 0: accuracy=0.5579 task_loss=1.1715 t=25s
    [qwen-s42-C-b0] eval @ round 0: accuracy=0.5579 task_loss=1.1715 t=29s
    [qwen-s42-C-b0] round 1 train_loss=1.1488 t=33s
    [qwen-s42-A-b0] round 1 train_loss=1.1488 t=39s
    [qwen-s42-C-b0] round 10 train_loss=1.1666 t=80s
    [qwen-s42-A-b0] round 10 train_loss=1.2138 t=86s
    [qwen-s42-C-b0] eval @ round 10: accuracy=0.6245 task_loss=1.1127 t=105s
    [qwen-s42-A-b0] eval @ round 10: accuracy=0.5812 task_loss=1.1508 t=111s
    [qwen-s42-C-b0] round 20 train_loss=0.9985 t=158s
    [qwen-s42-A-b0] round 20 train_loss=1.1043 t=175s
    [qwen-s42-C-b0] eval @ round 20: accuracy=0.7124 task_loss=0.9974 t=177s
    [qwen-s42-A-b0] eval @ round 20: accuracy=0.6304 task_loss=1.1064 t=199s
    [qwen-s42-C-b0] round 30 train_loss=0.8551 t=232s
    [qwen-s42-A-b0] round 30 train_loss=0.9687 t=253s
    [qwen-s42-C-b0] eval @ round 30: accuracy=0.7651 task_loss=0.8520 t=256s
    [qwen-s42-A-b0] eval @ round 30: accuracy=0.6742 task_loss=1.0466 t=277s
    [qwen-s42-C-b0] round 40 train_loss=0.6468 t=310s
    [qwen-s42-C-b0] eval @ round 40: accuracy=0.8076 task_loss=0.7083 t=332s
    [qwen-s42-A-b0] round 40 train_loss=0.9049 t=340s
    [qwen-s42-A-b0] eval @ round 40: accuracy=0.7089 task_loss=0.9773 t=364s
    [qwen-s42-C-b0] round 50 train_loss=0.6014 t=385s
    [qwen-s42-C-b0] eval @ round 50: accuracy=0.8341 task_loss=0.5899 t=408s
    [qwen-s42-A-b0] round 50 train_loss=0.9098 t=418s
    [qwen-s42-A-b0] eval @ round 50: accuracy=0.7470 task_loss=0.8887 t=442s
    [qwen-s42-C-b0] round 60 train_loss=0.5736 t=463s
    [qwen-s42-C-b0] eval @ round 60: accuracy=0.8509 task_loss=0.5321 t=486s
    [qwen-s42-A-b0] round 60 train_loss=0.7941 t=496s
    [qwen-s42-A-b0] eval @ round 60: accuracy=0.7757 task_loss=0.8075 t=520s
    [qwen-s42-C-b0] round 70 train_loss=0.5166 t=541s
    [qwen-s42-C-b0] eval @ round 70: accuracy=0.8592 task_loss=0.5158 t=565s
    [qwen-s42-A-b0] round 70 train_loss=0.7289 t=575s
    [qwen-s42-A-b0] eval @ round 70: accuracy=0.8012 task_loss=0.7305 t=599s
    [qwen-s42-C-b0] round 80 train_loss=0.3887 t=619s
    [qwen-s42-C-b0] eval @ round 80: accuracy=0.8545 task_loss=0.5044 t=643s
    [qwen-s42-A-b0] round 80 train_loss=0.5925 t=661s
    [qwen-s42-A-b0] eval @ round 80: accuracy=0.8179 task_loss=0.6537 t=685s
    [qwen-s42-C-b0] round 90 train_loss=0.4004 t=694s
    [qwen-s42-C-b0] eval @ round 90: accuracy=0.8639 task_loss=0.4775 t=717s
    [qwen-s42-A-b0] round 90 train_loss=0.5058 t=740s
    [qwen-s42-A-b0] eval @ round 90: accuracy=0.8321 task_loss=0.6074 t=764s
    [qwen-s42-C-b0] round 100 train_loss=0.3645 t=773s
    [qwen-s42-C-b0] eval @ round 100: accuracy=0.8658 task_loss=0.4579 t=797s
    [qwen-s42-A-b0] round 100 train_loss=0.4892 t=819s
    [qwen-s42-A-b0] eval @ round 100: accuracy=0.8433 task_loss=0.5690 t=842s
    [qwen-s42-C-b0] round 110 train_loss=0.5340 t=851s
    [qwen-s42-C-b0] eval @ round 110: accuracy=0.8666 task_loss=0.4603 t=874s
    [qwen-s42-A-b0] round 110 train_loss=0.6024 t=897s
    [qwen-s42-A-b0] eval @ round 110: accuracy=0.8446 task_loss=0.5560 t=920s
    [qwen-s42-C-b0] round 120 train_loss=0.4668 t=929s
    [qwen-s42-C-b0] eval @ round 120: accuracy=0.8611 task_loss=0.4820 t=953s
    [qwen-s42-A-b0] round 120 train_loss=0.5373 t=975s
    [qwen-s42-A-b0] eval @ round 120: accuracy=0.8455 task_loss=0.5673 t=999s
    [qwen-s42-C-b0] round 130 train_loss=0.2894 t=1008s
    [qwen-s42-C-b0] eval @ round 130: accuracy=0.8613 task_loss=0.4705 t=1031s
    [qwen-s42-A-b0] round 130 train_loss=0.3669 t=1054s
    [qwen-s42-A-b0] eval @ round 130: accuracy=0.8491 task_loss=0.5566 t=1077s
    [qwen-s42-C-b0] round 140 train_loss=0.3864 t=1086s
    [qwen-s42-C-b0] eval @ round 140: accuracy=0.8676 task_loss=0.4388 t=1110s
    [qwen-s42-A-b0] round 140 train_loss=0.4818 t=1132s
    [qwen-s42-A-b0] eval @ round 140: accuracy=0.8517 task_loss=0.5394 t=1155s
    [qwen-s42-C-b0] round 150 train_loss=0.3472 t=1165s
    [qwen-s42-C-b0] eval @ round 150: accuracy=0.8662 task_loss=0.4281 t=1188s
    [qwen-s42-A-b0] round 150 train_loss=0.4409 t=1211s
    [qwen-s42-A-b0] eval @ round 150: accuracy=0.8543 task_loss=0.5265 t=1234s
    [qwen-s42-C-b0] round 160 train_loss=0.3450 t=1244s
    [qwen-s42-C-b0] eval @ round 160: accuracy=0.8778 task_loss=0.3884 t=1267s
  [qwen-s42-C-b0] completed in 1268s
  [qwen-s42-A-b0.75] starting: ho_fl.train --model qwen --dataset ag_news --partition dirichlet_05 --rounds 160 --method hofl --beta 0.75 --layer-stats --seed 42 --probe-rounds 1,20,40,80,160 --device cuda --cache-dir /content/cache --output /content/work/p1/runs/qwen-s42-A-b0.75
    [qwen-s42-A-b0] round 160 train_loss=0.4666 t=1295s
    [qwen-s42-A-b0.75] eval @ round 0: accuracy=0.5579 task_loss=1.1715 t=24s
    [qwen-s42-A-b0.75] round 1 train_loss=1.3112 t=38s
    [qwen-s42-A-b0] eval @ round 160: accuracy=0.8600 task_loss=0.4968 t=1314s
  [qwen-s42-A-b0] completed in 1315s
  [qwen-s42-Ax2-b0] starting: ho_fl.train --model qwen --dataset ag_news --partition dirichlet_05 --rounds 160 --method hofl --beta 0.0 --server-lr 2.0 --layer-stats --seed 42 --device cuda --cache-dir /content/cache --output /content/work/p1/runs/qwen-s42-Ax2-b0
    [qwen-s42-Ax2-b0] eval @ round 0: accuracy=0.5579 task_loss=1.1715 t=28s
    [qwen-s42-Ax2-b0] round 1 train_loss=1.1488 t=34s
    [qwen-s42-A-b0.75] round 10 train_loss=1.2187 t=80s
    [qwen-s42-A-b0.75] eval @ round 10: accuracy=0.6092 task_loss=1.1271 t=104s
    [qwen-s42-Ax2-b0] round 10 train_loss=1.1556 t=83s
    [qwen-s42-Ax2-b0] eval @ round 10: accuracy=0.6434 task_loss=1.0996 t=106s
    [qwen-s42-A-b0.75] round 20 train_loss=1.0341 t=164s
    [qwen-s42-A-b0.75] eval @ round 20: accuracy=0.6991 task_loss=1.0192 t=188s
    [qwen-s42-Ax2-b0] round 20 train_loss=0.9883 t=157s
    [qwen-s42-Ax2-b0] eval @ round 20: accuracy=0.7033 task_loss=0.9764 t=181s
    [qwen-s42-A-b0.75] round 30 train_loss=0.8473 t=240s
    [qwen-s42-A-b0.75] eval @ round 30: accuracy=0.7283 task_loss=0.8923 t=264s
    [qwen-s42-Ax2-b0] round 30 train_loss=0.8332 t=235s
    [qwen-s42-Ax2-b0] eval @ round 30: accuracy=0.7625 task_loss=0.8208 t=259s
    [qwen-s42-A-b0.75] round 40 train_loss=0.7561 t=325s
    [qwen-s42-A-b0.75] eval @ round 40: accuracy=0.7616 task_loss=0.7503 t=348s
    [qwen-s42-Ax2-b0] round 40 train_loss=0.6111 t=310s
    [qwen-s42-Ax2-b0] eval @ round 40: accuracy=0.8091 task_loss=0.6697 t=334s
    [qwen-s42-A-b0.75] round 50 train_loss=0.5846 t=400s
    [qwen-s42-A-b0.75] eval @ round 50: accuracy=0.7971 task_loss=0.6369 t=424s
    [qwen-s42-Ax2-b0] round 50 train_loss=0.5892 t=388s
    [qwen-s42-Ax2-b0] eval @ round 50: accuracy=0.8387 task_loss=0.5785 t=411s
    [qwen-s42-A-b0.75] round 60 train_loss=0.4890 t=476s
    [qwen-s42-A-b0.75] eval @ round 60: accuracy=0.8222 task_loss=0.5845 t=499s
    [qwen-s42-Ax2-b0] round 60 train_loss=0.5929 t=465s
    [qwen-s42-Ax2-b0] eval @ round 60: accuracy=0.8509 task_loss=0.5399 t=488s
    [qwen-s42-A-b0.75] round 70 train_loss=0.4336 t=552s
    [qwen-s42-A-b0.75] eval @ round 70: accuracy=0.8347 task_loss=0.5677 t=575s
    [qwen-s42-Ax2-b0] round 70 train_loss=0.5346 t=544s
    [qwen-s42-Ax2-b0] eval @ round 70: accuracy=0.8545 task_loss=0.5354 t=567s
    [qwen-s42-A-b0.75] round 80 train_loss=0.4548 t=634s
    [qwen-s42-A-b0.75] eval @ round 80: accuracy=0.8418 task_loss=0.5505 t=658s
    [qwen-s42-Ax2-b0] round 80 train_loss=0.3862 t=618s
    [qwen-s42-Ax2-b0] eval @ round 80: accuracy=0.8559 task_loss=0.5050 t=642s
    [qwen-s42-A-b0.75] round 90 train_loss=0.4353 t=711s
    [qwen-s42-A-b0.75] eval @ round 90: accuracy=0.8421 task_loss=0.5524 t=734s
    [qwen-s42-Ax2-b0] round 90 train_loss=0.4005 t=696s
    [qwen-s42-Ax2-b0] eval @ round 90: accuracy=0.8638 task_loss=0.4736 t=720s
    [qwen-s42-A-b0.75] round 100 train_loss=0.5367 t=788s
    [qwen-s42-A-b0.75] eval @ round 100: accuracy=0.8484 task_loss=0.5323 t=811s
    [qwen-s42-Ax2-b0] round 100 train_loss=0.3693 t=775s
    [qwen-s42-Ax2-b0] eval @ round 100: accuracy=0.8680 task_loss=0.4514 t=798s
    [qwen-s42-A-b0.75] round 110 train_loss=0.4995 t=863s
    [qwen-s42-A-b0.75] eval @ round 110: accuracy=0.8476 task_loss=0.5071 t=887s
    [qwen-s42-Ax2-b0] round 110 train_loss=0.5067 t=853s
    [qwen-s42-Ax2-b0] eval @ round 110: accuracy=0.8709 task_loss=0.4452 t=876s
    [qwen-s42-A-b0.75] round 120 train_loss=0.3576 t=940s
    [qwen-s42-A-b0.75] eval @ round 120: accuracy=0.8564 task_loss=0.4853 t=963s
    [qwen-s42-Ax2-b0] round 120 train_loss=0.4504 t=930s
    [qwen-s42-Ax2-b0] eval @ round 120: accuracy=0.8641 task_loss=0.4610 t=954s
    [qwen-s42-A-b0.75] round 130 train_loss=0.3782 t=1015s
    [qwen-s42-A-b0.75] eval @ round 130: accuracy=0.8572 task_loss=0.4854 t=1039s
    [qwen-s42-Ax2-b0] round 130 train_loss=0.2757 t=1009s
    [qwen-s42-Ax2-b0] eval @ round 130: accuracy=0.8668 task_loss=0.4453 t=1032s
    [qwen-s42-A-b0.75] round 140 train_loss=0.4169 t=1091s
    [qwen-s42-A-b0.75] eval @ round 140: accuracy=0.8530 task_loss=0.4925 t=1115s
    [qwen-s42-Ax2-b0] round 140 train_loss=0.3798 t=1086s
    [qwen-s42-Ax2-b0] eval @ round 140: accuracy=0.8687 task_loss=0.4249 t=1109s
    [qwen-s42-A-b0.75] round 150 train_loss=0.3742 t=1167s
    [qwen-s42-A-b0.75] eval @ round 150: accuracy=0.8533 task_loss=0.4928 t=1191s
    [qwen-s42-Ax2-b0] round 150 train_loss=0.3413 t=1165s
    [qwen-s42-Ax2-b0] eval @ round 150: accuracy=0.8700 task_loss=0.4085 t=1188s
    [qwen-s42-A-b0.75] round 160 train_loss=0.4324 t=1251s
    [qwen-s42-A-b0.75] eval @ round 160: accuracy=0.8599 task_loss=0.4509 t=1274s
  [qwen-s42-A-b0.75] completed in 1275s
  [qwen-s42-Ax3-b0] starting: ho_fl.train --model qwen --dataset ag_news --partition dirichlet_05 --rounds 160 --method hofl --beta 0.0 --server-lr 3.0 --layer-stats --seed 42 --device cuda --cache-dir /content/cache --output /content/work/p1/runs/qwen-s42-Ax3-b0
    [qwen-s42-Ax2-b0] round 160 train_loss=0.3389 t=1236s
    [qwen-s42-Ax2-b0] eval @ round 160: accuracy=0.8772 task_loss=0.3798 t=1260s
  [qwen-s42-Ax2-b0] completed in 1261s
    [qwen-s42-Ax3-b0] eval @ round 0: accuracy=0.5579 task_loss=1.1715 t=29s
    [qwen-s42-Ax3-b0] round 1 train_loss=1.1488 t=32s
    [qwen-s42-Ax3-b0] round 10 train_loss=1.0815 t=55s
    [qwen-s42-Ax3-b0] eval @ round 10: accuracy=0.6812 task_loss=1.0310 t=67s
    [qwen-s42-Ax3-b0] round 20 train_loss=0.8396 t=93s
    [qwen-s42-Ax3-b0] eval @ round 20: accuracy=0.7459 task_loss=0.8309 t=105s
    [qwen-s42-Ax3-b0] round 30 train_loss=0.7106 t=131s
    [qwen-s42-Ax3-b0] eval @ round 30: accuracy=0.8243 task_loss=0.6168 t=142s
    [qwen-s42-Ax3-b0] round 40 train_loss=0.5013 t=169s
    [qwen-s42-Ax3-b0] eval @ round 40: accuracy=0.8447 task_loss=0.5470 t=180s
    [qwen-s42-Ax3-b0] round 50 train_loss=0.5144 t=206s
    [qwen-s42-Ax3-b0] eval @ round 50: accuracy=0.8591 task_loss=0.5153 t=218s
    [qwen-s42-Ax3-b0] round 60 train_loss=0.5343 t=244s
    [qwen-s42-Ax3-b0] eval @ round 60: accuracy=0.8650 task_loss=0.4759 t=255s
    [qwen-s42-Ax3-b0] round 70 train_loss=0.4587 t=282s
    [qwen-s42-Ax3-b0] eval @ round 70: accuracy=0.8663 task_loss=0.4733 t=293s
    [qwen-s42-Ax3-b0] round 80 train_loss=0.3327 t=319s
    [qwen-s42-Ax3-b0] eval @ round 80: accuracy=0.8661 task_loss=0.4415 t=331s
    [qwen-s42-Ax3-b0] round 90 train_loss=0.3435 t=357s
    [qwen-s42-Ax3-b0] eval @ round 90: accuracy=0.8757 task_loss=0.4123 t=369s
    [qwen-s42-Ax3-b0] round 100 train_loss=0.3220 t=395s
    [qwen-s42-Ax3-b0] eval @ round 100: accuracy=0.8767 task_loss=0.3921 t=407s
    [qwen-s42-Ax3-b0] round 110 train_loss=0.4473 t=433s
    [qwen-s42-Ax3-b0] eval @ round 110: accuracy=0.8761 task_loss=0.3947 t=445s
    [qwen-s42-Ax3-b0] round 120 train_loss=0.3907 t=471s
    [qwen-s42-Ax3-b0] eval @ round 120: accuracy=0.8700 task_loss=0.4049 t=483s
    [qwen-s42-Ax3-b0] round 130 train_loss=0.2521 t=509s
    [qwen-s42-Ax3-b0] eval @ round 130: accuracy=0.8717 task_loss=0.4003 t=521s
    [qwen-s42-Ax3-b0] round 140 train_loss=0.3504 t=546s
    [qwen-s42-Ax3-b0] eval @ round 140: accuracy=0.8755 task_loss=0.3814 t=558s
    [qwen-s42-Ax3-b0] round 150 train_loss=0.3085 t=584s
    [qwen-s42-Ax3-b0] eval @ round 150: accuracy=0.8772 task_loss=0.3706 t=596s
    [qwen-s42-Ax3-b0] round 160 train_loss=0.3048 t=622s
    [qwen-s42-Ax3-b0] eval @ round 160: accuracy=0.8859 task_loss=0.3504 t=634s
  [qwen-s42-Ax3-b0] completed in 635s

| run | status | wall min | estimate min | note |
| --- | --- | --- | --- | --- |
| qwen-s42-A-b0 | COMPLETED | 21.9 | 29.7 |  |
| qwen-s42-C-b0 | COMPLETED | 21.1 | 29.7 |  |
| qwen-s42-A-b0.75 | COMPLETED | 21.3 | 26.4 |  |
| qwen-s42-Ax2-b0 | COMPLETED | 21.0 | 26.9 |  |
| qwen-s42-Ax3-b0 | COMPLETED | 10.6 | 26.8 |  |

stage results: queue=ok
4.48 h of budget left
$ /content/venv/bin/python experiments/p1/p1.py feedback --root /content/work/p1 --cache-dir /content/cache --device cuda --model qwen --seeds 42 --quiet
== feedback ==
H1a: CANNOT_JUDGE; H1: b0 NOT_RUN, b0.75 NOT_RUN; 5 of 12 runs completed
written: /content/work/p1/feedback/feedback_P1.md, CSV files and /content/work/p1/feedback/feedback_P1_bundle.zip

stage results: feedback=ok
P1 feedback
Generated 2026-11-20T23:45:09+00:00 for revision cdea00b095d2 on nvidia-rtx-pro-6000-blackwell-server-edition-torch2.7.1. Model qwen, AG News Dirichlet 0.5, seeds [42], 160 rounds per run; 5 of 12 planned runs completed.

1. Baseline regression in this session
I1: {'decomfl': 'PASS_BITWISE', 'fedavg': 'PASS_BITWISE', 'hofl': 'PASS_BITWISE'} (golden mode: bitwise); I2 replay: PASS.

2. E1c: ablation ladder
arm	what it is	seeds	final accuracy % per seed	mean	eval loss	forward-eq. per round	uplink MB per round	paper (sd)
A-b0	HO-FL baseline, uniform sampling (equal-weight mean over all reports)	42	86.00	86.00	0.4968	90.0	11.34	85.51 (0.43)
C-b0	fo_mean, uniform sampling: each layer averages its first-order reports only	42	87.78	87.78	0.3884	90.0	11.34	
A-b0.75	HO-FL baseline, biased sampling (the HO-FL default)	42	85.99	85.99	0.4509	90.0	16.85	86.69 (0.56)
Ax2-b0	step-size control: baseline with the server step doubled	42	87.72	87.72	0.3798	90.0	11.34	
Ax3-b0	step-size control: baseline with the server step tripled	42	88.59	88.59	0.3504	90.0	11.34	
B-b0	fo_diluted without zeroth-order updates, uniform sampling: A minus B is the net contribution of the zeroth-order updates	not run						
B-b0.75	fo_diluted without zeroth-order updates, biased sampling	not run						
fedavg	FedAvg: every client trains every layer first-order (upper bound)	not run						
decomfl	DeComFL: every client trains every layer zeroth-order (lower bound)	not run						
C-b0.75	fo_mean, biased sampling	not run						
D-b0	fo_mean without zeroth-order updates, uniform sampling: partial-layer first-order training only	not run						
D-b0.75	fo_mean without zeroth-order updates, biased sampling	not run						
Forward-equivalents: one forward pass through all blocks for one batch; a backward pass costs two (accounting model, not a measurement).

3. H1a: is the gain of biased sampling a matter of dilution?
seed	G_C = C(b0) - A(b0)	G_beta = A(b0.75) - A(b0)	G_lr = max(Ax2, Ax3) - A(b0)
42	+1.78	-0.01	+2.59
mean	+1.78	-0.01	+2.59
Rule of the brief: supported when mean G_C >= 0.7 x mean G_beta and mean G_lr < 0.5 x mean G_C. CANNOT_JUDGE (preliminary: 1 of 3 seeds): biased sampling gave no gain over uniform sampling, so there is nothing to recover.

4. H1: how much do the zeroth-order updates contribute?
Model level, A (zeroth-order on) minus B (off, same weight 1/K):

sampling	seeds	A - B per seed	mean	seed sd of A	verdict
b0	none			n/a	NOT_RUN
b0.75	none			n/a	NOT_RUN
Report level (probe): each client's actual update against its first-order counterfactual from the same state and batches. zo: all layers below the client's cut pooled; fo: all layers at or above it (self-check, close to 1). The null value is 1/sqrt(dimension), the scatter of the cosine of a random direction.

qwen-s42-A-b0:

round	tier	layers	clients	mean cosine	standard error	null 1/sqrt(d)	median norm ratio actual/first-order
1	T1	fo	1	0.9993	n/a	0.0011	1.001
1	T1	zo	1	0.0008618	n/a	0.0018	0.754
1	T1	zo_lora_A	1	-0.0013	n/a	0.0023	0.825
1	T1	zo_lora_B	1	0.0039	n/a	0.0030	0.666
1	T2	fo	4	0.9994	0.0001858	0.0015	1.000
1	T2	zo	4	-0.000344	0.0002397	0.0013	0.770
1	T2	zo_lora_A	4	-0.0009543	0.0006971	0.0016	0.855
1	T2	zo_lora_B	4	0.0004504	0.0012	0.0021	0.670
1	T3	fo	1	0.9997	n/a	0.0025	0.999
1	T3	zo	1	-0.0004465	n/a	0.0010	0.664
1	T3	zo_lora_A	1	-3.353e-05	n/a	0.0013	0.757
1	T3	zo_lora_B	1	-0.0009825	n/a	0.0017	0.562
20	T1	fo	3	0.9997	6.005e-05	0.0011	1.000
20	T1	zo	3	0.0002314	0.0014	0.0018	0.692
20	T1	zo_lora_A	3	0.0001775	0.0017	0.0023	0.712
20	T1	zo_lora_B	3	0.0003118	0.0015	0.0030	0.659
20	T2	fo	1	0.9998	n/a	0.0015	1.000
20	T2	zo	1	0.0017	n/a	0.0013	0.670
20	T2	zo_lora_A	1	0.0017	n/a	0.0016	0.646
20	T2	zo_lora_B	1	0.0017	n/a	0.0021	0.717
20	T3	fo	2	0.9996	7.424e-05	0.0025	1.008
20	T3	zo	2	-0.0006318	0.0002899	0.0010	0.648
20	T3	zo_lora_A	2	-0.0007057	0.000153	0.0013	0.663
20	T3	zo_lora_B	2	-0.0004834	0.0010	0.0017	0.625
40	T1	fo	3	0.9997	1.62e-05	0.0011	1.000
40	T1	zo	3	0.0018	0.0012	0.0018	0.712
40	T1	zo_lora_A	3	0.000755	0.0010	0.0023	0.734
40	T1	zo_lora_B	3	0.0034	0.0017	0.0030	0.680
40	T2	fo	2	0.9998	1.664e-05	0.0015	1.003
40	T2	zo	2	-0.0005191	0.0017	0.0013	0.640
40	T2	zo_lora_A	2	-0.00074	0.0018	0.0016	0.637
40	T2	zo_lora_B	2	-0.0001324	0.0014	0.0021	0.645
40	T3	fo	1	0.9997	n/a	0.0025	1.004
40	T3	zo	1	-0.0002503	n/a	0.0010	0.634
40	T3	zo_lora_A	1	-0.0025	n/a	0.0013	0.642
40	T3	zo_lora_B	1	0.0036	n/a	0.0017	0.622
80	T1	fo	1	0.9997	n/a	0.0011	1.000
80	T1	zo	1	-0.0024	n/a	0.0018	0.751
80	T1	zo_lora_A	1	-0.0010	n/a	0.0023	0.753
80	T1	zo_lora_B	1	-0.0046	n/a	0.0030	0.747
80	T2	fo	1	0.9990	n/a	0.0015	1.006
80	T2	zo	1	0.0019	n/a	0.0013	0.891
80	T2	zo_lora_A	1	0.0019	n/a	0.0016	0.895
80	T2	zo_lora_B	1	0.0019	n/a	0.0021	0.883
80	T3	fo	4	0.9992	0.0002208	0.0025	1.009
80	T3	zo	4	0.0003897	0.0006442	0.0010	0.710
80	T3	zo_lora_A	4	0.0006774	0.0005077	0.0013	0.704
80	T3	zo_lora_B	4	-0.0001105	0.0008885	0.0017	0.732
160	T1	fo	1	0.9993	n/a	0.0011	1.003
160	T1	zo	1	0.0007881	n/a	0.0018	0.855
160	T1	zo_lora_A	1	0.0031	n/a	0.0023	0.857
160	T1	zo_lora_B	1	-0.0032	n/a	0.0030	0.852
160	T2	fo	3	0.9979	0.0007415	0.0015	1.012
160	T2	zo	3	-0.0004569	0.0011	0.0013	0.848
160	T2	zo_lora_A	3	-0.0003023	0.0014	0.0016	0.843
160	T2	zo_lora_B	3	-0.0007275	0.0011	0.0021	0.857
160	T3	fo	2	0.9991	0.0003201	0.0025	1.007
160	T3	zo	2	-8.641e-05	0.0004929	0.0010	0.990
160	T3	zo_lora_A	2	0.0001574	0.0008453	0.0013	0.994
160	T3	zo_lora_B	2	-0.0004763	9.007e-05	0.0017	0.986
qwen-s42-A-b0.75:

round	tier	layers	clients	mean cosine	standard error	null 1/sqrt(d)	median norm ratio actual/first-order
1	T1	fo	6	0.9994	7.541e-05	0.0011	1.000
1	T1	zo	6	0.0020	0.0006622	0.0018	0.784
1	T1	zo_lora_A	6	0.0010	0.0010	0.0023	0.863
1	T1	zo_lora_B	6	0.0033	0.0009487	0.0030	0.685
20	T1	fo	4	0.9998	1.886e-05	0.0011	1.000
20	T1	zo	4	-0.0005219	0.0009192	0.0018	0.722
20	T1	zo_lora_A	4	-0.0011	0.0006492	0.0023	0.746
20	T1	zo_lora_B	4	0.0004527	0.0014	0.0030	0.693
20	T3	fo	2	0.9996	0.0001815	0.0025	1.003
20	T3	zo	2	0.0004851	1.238e-05	0.0010	0.653
20	T3	zo_lora_A	2	-2.117e-05	0.0006007	0.0013	0.644
20	T3	zo_lora_B	2	0.0013	0.0011	0.0017	0.669
40	T1	fo	5	0.9997	4.417e-05	0.0011	1.001
40	T1	zo	5	0.0011	0.0012	0.0018	0.714
40	T1	zo_lora_A	5	0.0003309	0.0012	0.0023	0.728
40	T1	zo_lora_B	5	0.0022	0.0019	0.0030	0.694
40	T3	fo	1	0.9995	n/a	0.0025	1.010
40	T3	zo	1	0.0019	n/a	0.0010	0.729
40	T3	zo_lora_A	1	0.0020	n/a	0.0013	0.714
40	T3	zo_lora_B	1	0.0016	n/a	0.0017	0.757
80	T1	fo	5	0.9990	0.0002018	0.0011	0.999
80	T1	zo	5	0.0002179	0.0011	0.0018	0.910
80	T1	zo_lora_A	5	0.0010	0.0011	0.0023	0.910
80	T1	zo_lora_B	5	-0.0012	0.0018	0.0030	0.911
80	T3	fo	1	0.9988	n/a	0.0025	1.007
80	T3	zo	1	0.0006084	n/a	0.0010	0.756
80	T3	zo_lora_A	1	0.0006388	n/a	0.0013	0.731
80	T3	zo_lora_B	1	0.0005529	n/a	0.0017	0.805
160	T1	fo	6	0.9980	0.0010	0.0011	0.999
160	T1	zo	6	6.912e-05	0.0005907	0.0018	0.843
160	T1	zo_lora_A	6	0.0003069	0.0007511	0.0023	0.844
160	T1	zo_lora_B	6	-0.0003375	0.0007256	0.0030	0.844
Reading: H1 expects the zo cosine in the 1e-3 to 1e-2 range and the norm ratio between 0.3 and 3. With AdamW every coordinate moves by about the learning rate per step whatever the gradient, so the norm ratio says little; the cosine and A minus B carry the test. zo_lora_A at round 1 has no first-order direction to compare with (LoRA B starts at zero, so the gradient of A is zero at the first step).

5. Layer statistics (averages over rounds and over the layers of a segment)
S0 lies below every cut (no first-order source), S1 to S3 start at the three cuts. Norms are of the per-layer update vectors.

arm	seed	segment	first-order reports	zeroth-order reports	norm of a first-order component	norm of a zeroth-order component	norm of the first-order mean	norm of the zeroth-order mean	cosine FO-FO	cosine ZO-ZO	cosine FO-ZO	norm of the applied update	electorate rule
A-b0	42	S0	0	6	n/a	0.00510	n/a	0.00209	n/a	-9.692e-05	n/a	0.00209	all 100%
A-b0	42	S1	2.03	3.97	0.00714	0.00511	0.00595	0.00265	0.313	2.64e-05	5.895e-05	0.00271	all 100%
A-b0	42	S2	4.08	1.93	0.00729	0.00512	0.00484	0.00377	0.247	0.0003431	-9.225e-05	0.00350	all 100%
A-b0	42	S3	6	0	0.00735	n/a	0.00414	n/a	0.189	n/a	n/a	0.00414	all 100%
C-b0	42	S0	0	6	n/a	0.00510	n/a	0.00208	n/a	3.863e-05	n/a	0.00208	thin_all 100%
C-b0	42	S1	2.03	3.97	0.00674	0.00510	0.00532	0.00264	0.170	9.846e-05	0.0001153	0.00512	fo 94%, thin_all 6%
C-b0	42	S2	4.08	1.93	0.00685	0.00510	0.00418	0.00376	0.152	7.488e-05	-3.306e-05	0.00418	fo 100%
C-b0	42	S3	6	0	0.00684	n/a	0.00366	n/a	0.148	n/a	n/a	0.00366	fo 100%
A-b0.75	42	S0	0	6	n/a	0.00510	n/a	0.00208	n/a	2.335e-05	n/a	0.00208	all 100%
A-b0.75	42	S1	4.97	1.02	0.00678	0.00508	0.00394	0.00444	0.177	-5.366e-05	0.0001105	0.00339	all 100%
A-b0.75	42	S2	5.56	0.44	0.00694	0.00507	0.00348	0.00487	0.102	-6.556e-05	6.051e-05	0.00328	all 100%
A-b0.75	42	S3	6	0	0.00702	n/a	0.00356	n/a	0.122	n/a	n/a	0.00356	all 100%
Ax2-b0	42	S0	0	6	n/a	0.00510	n/a	0.00208	n/a	1.572e-05	n/a	0.00417	all 100%
Ax2-b0	42	S1	2.03	3.97	0.00665	0.00510	0.00522	0.00264	0.160	7.038e-06	5.241e-05	0.00498	all 100%
Ax2-b0	42	S2	4.08	1.93	0.00677	0.00511	0.00404	0.00377	0.134	8.134e-05	-4.771e-05	0.00600	all 100%
Ax2-b0	42	S3	6	0	0.00682	n/a	0.00349	n/a	0.121	n/a	n/a	0.00698	all 100%
Ax3-b0	42	S0	0	6	n/a	0.00510	n/a	0.00209	n/a	-7.596e-06	n/a	0.00626	all 100%
Ax3-b0	42	S1	2.03	3.97	0.00649	0.00511	0.00494	0.00264	0.091	-8.721e-06	9.823e-05	0.00722	all 100%
Ax3-b0	42	S2	4.08	1.93	0.00659	0.00512	0.00376	0.00377	0.090	2.116e-05	-6.862e-05	0.00843	all 100%
Ax3-b0	42	S3	6	0	0.00667	n/a	0.00322	n/a	0.088	n/a	n/a	0.00966	all 100%
6. First-order supply
Supply map (E1a, no training):

Supply map for qwen: three tiers of 10 clients, cuts [8, 16, 24], depth 28.
S1, S2, S3 are the segments that start at the three cuts; width = first-order reports per round.

dep_round_v1 beta=0
  expected width S1 / S2 / S3: 2.00 (brief 2.00) / 4.00 (brief 4.00) / 6.00 (brief 6.00)
  effective sources (N_eff) S1 / S2 / S3: 10.00 (brief 10.00) / 20.00 (brief 20.00) / 30.00 (brief 30.00)
  S1 largest single share: 10.0% (brief 10.0%); S1 share of the 6 largest sources: 60.0% (brief 60.0%); largest presence of a client: 20.0% (brief 20.0%)
  simulated rounds with S1 width 0 / 1 / <=2 / <3: 6.51% (brief 6.40%) / 25.90% (brief 26.50%) / 69.14% (brief 69.20%) / 69.14%
  participation bias term N*sum((p_i-1/N)^2): 0.000 (brief 0.000); sampling-weighted d_ZO/q: 311296 (brief 311296)

dep_round_v1 beta=0.75
  expected width S1 / S2 / S3: 5.00 (brief 5.00) / 5.50 (brief 5.50) / 6.00 (brief 6.00)
  effective sources (N_eff) S1 / S2 / S3: 6.49 (brief 6.49) / 7.81 (brief 7.81) / 9.23 (brief 9.23)
  S1 largest single share: 16.0% (brief 16.0%); S1 share of the 6 largest sources: 96.0% (brief 96.0%); largest presence of a client: 80.0% (brief 80.0%)
  simulated rounds with S1 width 0 / 1 / <=2 / <3: 0.00% (brief 0.00%) / 0.04% (brief 0.01%) / 0.41% (brief 0.40%) / 0.41%
  participation bias term N*sum((p_i-1/N)^2): 2.250 (brief 2.250); sampling-weighted d_ZO/q: 194560 (brief 194560)

tier_equalised beta=0.75
  expected width S1 / S2 / S3: 5.00 (brief 5.00) / 5.50 (brief 5.50) / 6.00 (brief 6.00)
  effective sources (N_eff) S1 / S2 / S3: 10.00 (brief 10.00) / 11.98 (brief 11.98) / 14.12 (brief 14.12)
  S1 largest single share: 10.0% (brief 10.0%); S1 share of the 6 largest sources: 60.0% (brief 60.0%); largest presence of a client: 50.0% (brief 50.0%)
  simulated rounds with S1 width 0 / 1 / <=2 / <3: 0.00% (brief 0.00%) / 0.02% (brief 0.02%) / 0.41% (brief 0.40%) / 0.41%
  participation bias term N*sum((p_i-1/N)^2): 1.125 (brief 1.125); sampling-weighted d_ZO/q: 194560 (brief 194560)

tier_quota 3,1,2
  expected width S1 / S2 / S3: 3.00 (brief 3.00) / 4.00 (brief 4.00) / 6.00 (brief 6.00)
  effective sources (N_eff) S1 / S2 / S3: 10.00 (brief 10.00) / 16.00 (brief 16.00) / 25.71 (brief 25.71)
  S1 largest single share: 10.0% (brief 10.0%); S1 share of the 6 largest sources: 60.0% (brief 60.0%); largest presence of a client: 30.0% (brief 30.0%)
  simulated rounds with S1 width 0 / 1 / <=2 / <3: 0.00% (brief 0.00%) / 0.00% (brief 0.00%) / 0.00% (brief 0.00%) / 0.00%
  participation bias term N*sum((p_i-1/N)^2): 0.167 (brief 0.167); sampling-weighted d_ZO/q: 285355 (brief 285355)

tier_quota 4,1,1
  expected width S1 / S2 / S3: 4.00 (brief 4.00) / 5.00 (brief 5.00) / 6.00 (brief 6.00)
  effective sources (N_eff) S1 / S2 / S3: 10.00 (brief 10.00) / 14.71 (brief 14.71) / 20.00 (brief 20.00)
  S1 largest single share: 10.0% (brief 10.0%); S1 share of the 6 largest sources: 60.0% (brief 60.0%); largest presence of a client: 40.0% (brief 40.0%)
  simulated rounds with S1 width 0 / 1 / <=2 / <3: 0.00% (brief 0.00%) / 0.00% (brief 0.00%) / 0.00% (brief 0.00%) / 0.00%
  participation bias term N*sum((p_i-1/N)^2): 0.500 (brief 0.500); sampling-weighted d_ZO/q: 233472 (brief 233472)

tier_quota 2,2,2
  expected width S1 / S2 / S3: 2.00 / 4.00 / 6.00
  effective sources (N_eff) S1 / S2 / S3: 10.00 / 20.00 / 30.00
  S1 largest single share: 10.0%; S1 share of the 6 largest sources: 60.0%; largest presence of a client: 20.0%
  simulated rounds with S1 width 0 / 1 / <=2 / <3: 0.00% / 0.00% / 100.00% / 100.00%
  participation bias term N*sum((p_i-1/N)^2): 0.000; sampling-weighted d_ZO/q: 311296
Realised in the runs (one run scatters around the expectations above; the largest share is biased upwards):

run	mean width S1 / S2 / S3	rounds with S1 width 0	S1 width <= 2	S1 sources	S1 N_eff	largest presence
qwen-s42-A-b0	2.03 / 4.08 / 6.00	6.2%	65.6%	10	9.82	25.6%
qwen-s42-C-b0	2.03 / 4.08 / 6.00	6.2%	65.6%	10	9.82	25.6%
qwen-s42-A-b0.75	4.97 / 5.56 / 6.00	0.0%	0.6%	10	6.45	83.8%
qwen-s42-Ax2-b0	2.03 / 4.08 / 6.00	6.2%	65.6%	10	9.82	25.6%
qwen-s42-Ax3-b0	2.03 / 4.08 / 6.00	6.2%	65.6%	10	9.82	25.6%
7. Queue, timing and GPU hours
Runs at a time: 2 (requested 2).

run	block	status	wall min	estimated min	note
qwen-s42-A-b0	1	COMPLETED	21.9	29.7	
qwen-s42-C-b0	1	COMPLETED	21.1	29.7	
qwen-s42-A-b0.75	1	COMPLETED	21.3	26.4	
qwen-s42-Ax2-b0	1	COMPLETED	21.0	26.9	
qwen-s42-Ax3-b0	1	COMPLETED	10.6	26.8	
Measured alone on the GPU (three-round preflight, or a full run when one run at a time is used):

cost profile	s per round	s per evaluation	start-up s	peak GPU memory GiB	estimated min per 160-round run	measured on
qwen/hofl	3.00	11.81	31.4	9.8	11.9	preflight
While sharing the GPU: qwen/hofl: 2.60 s per round with 2 runs at a time.

Wall time of all runs recorded under this root: 1.62 h (sum over runs; with several runs at a time the session is shorter).

8. What this means and what to do next
E0 check, A-b0: 86.00 against the paper's 85.51 (+0.49 points, 1.1 paper standard deviations, 1 of 3 seeds; the acceptance rule max(1.0, 2 sd) applies to the mean over all seeds).
E0 check, A-b0.75: 85.99 against the paper's 86.69 (-0.70 points, 1.3 paper standard deviations, 1 of 3 seeds; the acceptance rule max(1.0, 2 sd) applies to the mean over all seeds).
H1a cannot be judged: biased sampling gave no gain over uniform sampling, so there is nothing to recover.
Probe, A-b0 seed 42: pooled zeroth-order cosine between -0.0024 and +0.0019 over rounds and tiers (null scatter about 0.0014); first-order self-check cosine at least 0.9979.
Probe, A-b0.75 seed 42: pooled zeroth-order cosine between -0.0005 and +0.0020 over rounds and tiers (null scatter about 0.0015); first-order self-check cosine at least 0.9980.
Gate G1 needs the H1 and H1a results for the research lead's decision; bring this report back.
result bundle: download started
Downloading "feedback_P1_bundle.zip": 



$ /content/venv/bin/python experiments/p1/p1.py queue --root /content/work/p1 --cache-dir /content/cache --device cuda --model qwen --seeds 42 --blocks 2 --jobs 2 --deadline 1795233849.7437687
== queue (qwen, seeds [42], blocks [2], 4 runs of 160 rounds, up to 2 at a time, nvidia-rtx-pro-6000-blackwell-server-edition-torch2.7.1) ==
  [preflight qwen/hofl-zo-off] starting: ho_fl.train --model qwen --dataset ag_news --partition dirichlet_05 --rounds 3 --method hofl --beta 0.0 --zo-mode off --agg-electorate fo --agg-weight cohort --layer-stats --seed 42 --evaluate-every 3 --device cuda --cache-dir /content/cache --output /content/work/p1/preflight/nvidia-rtx-pro-6000-blackwell-server-edition-torch2.7.1/qwen/hofl-zo-off
    [preflight qwen/hofl-zo-off] eval @ round 0: accuracy=0.5579 task_loss=1.1715 t=16s
    [preflight qwen/hofl-zo-off] round 1 train_loss=1.1499 t=18s
    [preflight qwen/hofl-zo-off] eval @ round 3: accuracy=0.5568 task_loss=1.1697 t=33s
  [preflight qwen/hofl-zo-off] completed in 34s
  qwen/hofl-zo-off: 1.6s per round, 11.8s per evaluation, 5s start-up, peak GPU memory 8.6 GiB
  [preflight qwen/fedavg] starting: ho_fl.train --model qwen --dataset ag_news --partition dirichlet_05 --rounds 3 --method fedavg --seed 42 --evaluate-every 3 --device cuda --cache-dir /content/cache --output /content/work/p1/preflight/nvidia-rtx-pro-6000-blackwell-server-edition-torch2.7.1/qwen/fedavg
    [preflight qwen/fedavg] eval @ round 0: accuracy=0.5579 task_loss=1.1715 t=16s
    [preflight qwen/fedavg] round 1 train_loss=1.1478 t=18s
    [preflight qwen/fedavg] eval @ round 3: accuracy=0.5604 task_loss=1.1661 t=34s
  [preflight qwen/fedavg] completed in 35s
  qwen/fedavg: 2.0s per round, 11.6s per evaluation, 5s start-up, peak GPU memory 14.5 GiB
  [preflight qwen/decomfl] starting: ho_fl.train --model qwen --dataset ag_news --partition dirichlet_05 --rounds 3 --method decomfl --seed 42 --evaluate-every 3 --device cuda --cache-dir /content/cache --output /content/work/p1/preflight/nvidia-rtx-pro-6000-blackwell-server-edition-torch2.7.1/qwen/decomfl
    [preflight qwen/decomfl] eval @ round 0: accuracy=0.5579 task_loss=1.1715 t=16s
    [preflight qwen/decomfl] round 1 train_loss=1.1470 t=21s
    [preflight qwen/decomfl] eval @ round 3: accuracy=0.5572 task_loss=1.1718 t=41s
  [preflight qwen/decomfl] completed in 41s
  qwen/decomfl: 4.2s per round, 11.8s per evaluation, 5s start-up, peak GPU memory 4.4 GiB
  [qwen-s42-B-b0] starting: ho_fl.train --model qwen --dataset ag_news --partition dirichlet_05 --rounds 160 --method hofl --beta 0.0 --zo-mode off --agg-electorate fo --agg-weight cohort --layer-stats --seed 42 --device cuda --cache-dir /content/cache --output /content/work/p1/runs/qwen-s42-B-b0
  [qwen-s42-B-b0.75] starting: ho_fl.train --model qwen --dataset ag_news --partition dirichlet_05 --rounds 160 --method hofl --beta 0.75 --zo-mode off --agg-electorate fo --agg-weight cohort --layer-stats --seed 42 --device cuda --cache-dir /content/cache --output /content/work/p1/runs/qwen-s42-B-b0.75
    [qwen-s42-B-b0] eval @ round 0: accuracy=0.5579 task_loss=1.1715 t=25s
    [qwen-s42-B-b0] round 1 train_loss=1.1499 t=29s
    [qwen-s42-B-b0.75] eval @ round 0: accuracy=0.5579 task_loss=1.1715 t=30s
    [qwen-s42-B-b0.75] round 1 train_loss=1.3122 t=33s
    [qwen-s42-B-b0] round 10 train_loss=1.2168 t=59s
    [qwen-s42-B-b0.75] round 10 train_loss=1.2153 t=68s
    [qwen-s42-B-b0] eval @ round 10: accuracy=0.5783 task_loss=1.1510 t=84s
    [qwen-s42-B-b0.75] eval @ round 10: accuracy=0.6126 task_loss=1.1269 t=92s
    [qwen-s42-B-b0] round 20 train_loss=1.1034 t=118s
    [qwen-s42-B-b0.75] round 20 train_loss=1.0372 t=131s
    [qwen-s42-B-b0] eval @ round 20: accuracy=0.6293 task_loss=1.1055 t=141s
    [qwen-s42-B-b0.75] eval @ round 20: accuracy=0.6987 task_loss=1.0204 t=155s
    [qwen-s42-B-b0] round 30 train_loss=0.9716 t=177s
    [qwen-s42-B-b0.75] round 30 train_loss=0.8502 t=194s
    [qwen-s42-B-b0] eval @ round 30: accuracy=0.6750 task_loss=1.0464 t=200s
    [qwen-s42-B-b0.75] eval @ round 30: accuracy=0.7283 task_loss=0.8918 t=217s
    [qwen-s42-B-b0] round 40 train_loss=0.9051 t=236s
    [qwen-s42-B-b0] eval @ round 40: accuracy=0.7095 task_loss=0.9769 t=259s
    [qwen-s42-B-b0.75] round 40 train_loss=0.7555 t=256s
    [qwen-s42-B-b0.75] eval @ round 40: accuracy=0.7621 task_loss=0.7509 t=279s
    [qwen-s42-B-b0] round 50 train_loss=0.9083 t=295s
    [qwen-s42-B-b0] eval @ round 50: accuracy=0.7474 task_loss=0.8890 t=318s
    [qwen-s42-B-b0.75] round 50 train_loss=0.5873 t=318s
    [qwen-s42-B-b0.75] eval @ round 50: accuracy=0.7980 task_loss=0.6373 t=341s
    [qwen-s42-B-b0] round 60 train_loss=0.7962 t=354s
    [qwen-s42-B-b0] eval @ round 60: accuracy=0.7766 task_loss=0.8079 t=377s
    [qwen-s42-B-b0.75] round 60 train_loss=0.4893 t=381s
    [qwen-s42-B-b0.75] eval @ round 60: accuracy=0.8225 task_loss=0.5838 t=403s
    [qwen-s42-B-b0] round 70 train_loss=0.7311 t=413s
    [qwen-s42-B-b0] eval @ round 70: accuracy=0.8020 task_loss=0.7299 t=436s
    [qwen-s42-B-b0.75] round 70 train_loss=0.4297 t=442s
    [qwen-s42-B-b0.75] eval @ round 70: accuracy=0.8354 task_loss=0.5676 t=465s
    [qwen-s42-B-b0] round 80 train_loss=0.5926 t=471s
    [qwen-s42-B-b0] eval @ round 80: accuracy=0.8163 task_loss=0.6550 t=494s
    [qwen-s42-B-b0.75] round 80 train_loss=0.4532 t=505s
    [qwen-s42-B-b0] round 90 train_loss=0.5046 t=528s
    [qwen-s42-B-b0.75] eval @ round 80: accuracy=0.8409 task_loss=0.5501 t=528s
    [qwen-s42-B-b0] eval @ round 90: accuracy=0.8329 task_loss=0.6071 t=552s
    [qwen-s42-B-b0.75] round 90 train_loss=0.4349 t=569s
    [qwen-s42-B-b0] round 100 train_loss=0.4907 t=588s
    [qwen-s42-B-b0.75] eval @ round 90: accuracy=0.8426 task_loss=0.5508 t=592s
    [qwen-s42-B-b0] eval @ round 100: accuracy=0.8439 task_loss=0.5681 t=612s
    [qwen-s42-B-b0.75] round 100 train_loss=0.5342 t=629s
    [qwen-s42-B-b0] round 110 train_loss=0.6021 t=647s
    [qwen-s42-B-b0.75] eval @ round 100: accuracy=0.8500 task_loss=0.5295 t=653s
    [qwen-s42-B-b0] eval @ round 110: accuracy=0.8461 task_loss=0.5558 t=671s
    [qwen-s42-B-b0.75] round 110 train_loss=0.5015 t=691s
    [qwen-s42-B-b0] round 120 train_loss=0.5382 t=705s
    [qwen-s42-B-b0.75] eval @ round 110: accuracy=0.8483 task_loss=0.5054 t=715s
    [qwen-s42-B-b0] eval @ round 120: accuracy=0.8457 task_loss=0.5669 t=730s
    [qwen-s42-B-b0.75] round 120 train_loss=0.3596 t=754s
    [qwen-s42-B-b0] round 130 train_loss=0.3710 t=764s
    [qwen-s42-B-b0.75] eval @ round 120: accuracy=0.8579 task_loss=0.4847 t=778s
    [qwen-s42-B-b0] eval @ round 130: accuracy=0.8495 task_loss=0.5554 t=789s
    [qwen-s42-B-b0.75] round 130 train_loss=0.3740 t=816s
    [qwen-s42-B-b0] round 140 train_loss=0.4830 t=822s
    [qwen-s42-B-b0.75] eval @ round 130: accuracy=0.8567 task_loss=0.4842 t=841s
    [qwen-s42-B-b0] eval @ round 140: accuracy=0.8516 task_loss=0.5402 t=847s
    [qwen-s42-B-b0] round 150 train_loss=0.4452 t=881s
    [qwen-s42-B-b0.75] round 140 train_loss=0.4181 t=878s
    [qwen-s42-B-b0] eval @ round 150: accuracy=0.8542 task_loss=0.5256 t=906s
    [qwen-s42-B-b0.75] eval @ round 140: accuracy=0.8530 task_loss=0.4904 t=903s
    [qwen-s42-B-b0] round 160 train_loss=0.4623 t=938s
    [qwen-s42-B-b0.75] round 150 train_loss=0.3817 t=940s
    [qwen-s42-B-b0] eval @ round 160: accuracy=0.8582 task_loss=0.4975 t=963s
  [qwen-s42-B-b0] completed in 964s
  [qwen-s42-fedavg] starting: ho_fl.train --model qwen --dataset ag_news --partition dirichlet_05 --rounds 160 --method fedavg --seed 42 --device cuda --cache-dir /content/cache --output /content/work/p1/runs/qwen-s42-fedavg
    [qwen-s42-B-b0.75] eval @ round 150: accuracy=0.8542 task_loss=0.4911 t=962s
    [qwen-s42-fedavg] eval @ round 0: accuracy=0.5579 task_loss=1.1715 t=28s
    [qwen-s42-fedavg] round 1 train_loss=1.1478 t=32s
    [qwen-s42-B-b0.75] round 160 train_loss=0.4323 t=999s
    [qwen-s42-B-b0.75] eval @ round 160: accuracy=0.8595 task_loss=0.4513 t=1023s
  [qwen-s42-B-b0.75] completed in 1024s
  [qwen-s42-decomfl] starting: ho_fl.train --model qwen --dataset ag_news --partition dirichlet_05 --rounds 160 --method decomfl --seed 42 --device cuda --cache-dir /content/cache --output /content/work/p1/runs/qwen-s42-decomfl
    [qwen-s42-fedavg] round 10 train_loss=1.1464 t=66s
    [qwen-s42-fedavg] eval @ round 10: accuracy=0.6430 task_loss=1.0936 t=87s
    [qwen-s42-decomfl] eval @ round 0: accuracy=0.5579 task_loss=1.1715 t=30s
    [qwen-s42-decomfl] round 1 train_loss=1.1470 t=39s
    [qwen-s42-fedavg] round 20 train_loss=0.8708 t=127s
    [qwen-s42-fedavg] eval @ round 20: accuracy=0.7722 task_loss=0.8811 t=152s
    [qwen-s42-decomfl] round 10 train_loss=1.2373 t=117s
    [qwen-s42-fedavg] round 30 train_loss=0.6475 t=192s
    [qwen-s42-decomfl] eval @ round 10: accuracy=0.5570 task_loss=1.1725 t=141s
    [qwen-s42-fedavg] eval @ round 30: accuracy=0.8317 task_loss=0.6370 t=217s
    [qwen-s42-fedavg] round 40 train_loss=0.4488 t=257s
    [qwen-s42-fedavg] eval @ round 40: accuracy=0.8571 task_loss=0.5062 t=281s
    [qwen-s42-decomfl] round 20 train_loss=1.1682 t=228s
    [qwen-s42-decomfl] eval @ round 20: accuracy=0.5589 task_loss=1.1724 t=252s
    [qwen-s42-fedavg] round 50 train_loss=0.4756 t=322s
    [qwen-s42-fedavg] eval @ round 50: accuracy=0.8643 task_loss=0.4758 t=346s
    [qwen-s42-fedavg] round 60 train_loss=0.4676 t=386s
    [qwen-s42-decomfl] round 30 train_loss=1.0513 t=339s
    [qwen-s42-fedavg] eval @ round 60: accuracy=0.8688 task_loss=0.4538 t=410s
    [qwen-s42-decomfl] eval @ round 30: accuracy=0.5562 task_loss=1.1717 t=363s
    [qwen-s42-fedavg] round 70 train_loss=0.4022 t=452s
    [qwen-s42-fedavg] eval @ round 70: accuracy=0.8722 task_loss=0.4416 t=476s
    [qwen-s42-decomfl] round 40 train_loss=1.1117 t=450s
    [qwen-s42-fedavg] round 80 train_loss=0.2981 t=515s
    [qwen-s42-decomfl] eval @ round 40: accuracy=0.5576 task_loss=1.1719 t=475s
    [qwen-s42-fedavg] eval @ round 80: accuracy=0.8728 task_loss=0.4256 t=540s
    [qwen-s42-fedavg] round 90 train_loss=0.2925 t=580s
    [qwen-s42-fedavg] eval @ round 90: accuracy=0.8747 task_loss=0.4054 t=604s
    [qwen-s42-decomfl] round 50 train_loss=1.2220 t=562s
    [qwen-s42-fedavg] round 100 train_loss=0.3334 t=645s
    [qwen-s42-decomfl] eval @ round 50: accuracy=0.5571 task_loss=1.1715 t=586s
    [qwen-s42-fedavg] eval @ round 100: accuracy=0.8778 task_loss=0.3930 t=670s
    [qwen-s42-fedavg] round 110 train_loss=0.4421 t=710s
    [qwen-s42-fedavg] eval @ round 110: accuracy=0.8771 task_loss=0.3874 t=734s
    [qwen-s42-decomfl] round 60 train_loss=1.1164 t=673s
    [qwen-s42-decomfl] eval @ round 60: accuracy=0.5579 task_loss=1.1720 t=697s
    [qwen-s42-fedavg] round 120 train_loss=0.3511 t=775s
    [qwen-s42-fedavg] eval @ round 120: accuracy=0.8766 task_loss=0.3817 t=800s
    [qwen-s42-fedavg] round 130 train_loss=0.2400 t=839s
    [qwen-s42-decomfl] round 70 train_loss=1.0821 t=783s
    [qwen-s42-fedavg] eval @ round 130: accuracy=0.8771 task_loss=0.3814 t=864s
    [qwen-s42-decomfl] eval @ round 70: accuracy=0.5583 task_loss=1.1714 t=808s
    [qwen-s42-fedavg] round 140 train_loss=0.2890 t=904s
    [qwen-s42-fedavg] eval @ round 140: accuracy=0.8792 task_loss=0.3674 t=928s
    [qwen-s42-decomfl] round 80 train_loss=1.1615 t=894s
    [qwen-s42-fedavg] round 150 train_loss=0.3019 t=969s
    [qwen-s42-decomfl] eval @ round 80: accuracy=0.5579 task_loss=1.1716 t=918s
    [qwen-s42-fedavg] eval @ round 150: accuracy=0.8807 task_loss=0.3590 t=994s
    [qwen-s42-fedavg] round 160 train_loss=0.3098 t=1034s
    [qwen-s42-fedavg] eval @ round 160: accuracy=0.8833 task_loss=0.3532 t=1058s
  [qwen-s42-fedavg] completed in 1059s
    [qwen-s42-decomfl] round 90 train_loss=1.0155 t=999s
    [qwen-s42-decomfl] eval @ round 90: accuracy=0.5574 task_loss=1.1722 t=1011s
    [qwen-s42-decomfl] round 100 train_loss=1.0944 t=1053s
    [qwen-s42-decomfl] eval @ round 100: accuracy=0.5564 task_loss=1.1709 t=1065s
    [qwen-s42-decomfl] round 110 train_loss=1.1655 t=1107s
    [qwen-s42-decomfl] eval @ round 110: accuracy=0.5589 task_loss=1.1712 t=1118s
    [qwen-s42-decomfl] round 120 train_loss=1.0368 t=1160s
    [qwen-s42-decomfl] eval @ round 120: accuracy=0.5591 task_loss=1.1713 t=1172s
    [qwen-s42-decomfl] round 130 train_loss=1.1870 t=1214s
    [qwen-s42-decomfl] eval @ round 130: accuracy=0.5604 task_loss=1.1702 t=1226s
    [qwen-s42-decomfl] round 140 train_loss=1.2148 t=1267s
    [qwen-s42-decomfl] eval @ round 140: accuracy=0.5580 task_loss=1.1706 t=1279s
    [qwen-s42-decomfl] round 150 train_loss=1.2040 t=1321s
    [qwen-s42-decomfl] eval @ round 150: accuracy=0.5587 task_loss=1.1708 t=1333s
    [qwen-s42-decomfl] round 160 train_loss=1.1582 t=1375s
    [qwen-s42-decomfl] eval @ round 160: accuracy=0.5570 task_loss=1.1707 t=1386s
  [qwen-s42-decomfl] completed in 1387s

| run | status | wall min | estimate min | note |
| --- | --- | --- | --- | --- |
| qwen-s42-B-b0 | COMPLETED | 16.1 | 19.2 |  |
| qwen-s42-B-b0.75 | COMPLETED | 17.1 | 19.2 |  |
| qwen-s42-fedavg | COMPLETED | 17.6 | 22.7 |  |
| qwen-s42-decomfl | COMPLETED | 23.1 | 39.4 |  |

stage results: queue=ok
3.78 h of budget left
$ /content/venv/bin/python experiments/p1/p1.py feedback --root /content/work/p1 --cache-dir /content/cache --device cuda --model qwen --seeds 42 --quiet
== feedback ==
H1a: CANNOT_JUDGE; H1: b0 CANNOT_JUDGE, b0.75 CANNOT_JUDGE; 9 of 12 runs completed
written: /content/work/p1/feedback/feedback_P1.md, CSV files and /content/work/p1/feedback/feedback_P1_bundle.zip

stage results: feedback=ok
P1 feedback
Generated 2026-11-21T00:27:17+00:00 for revision cdea00b095d2 on nvidia-rtx-pro-6000-blackwell-server-edition-torch2.7.1. Model qwen, AG News Dirichlet 0.5, seeds [42], 160 rounds per run; 9 of 12 planned runs completed.

1. Baseline regression in this session
I1: {'decomfl': 'PASS_BITWISE', 'fedavg': 'PASS_BITWISE', 'hofl': 'PASS_BITWISE'} (golden mode: bitwise); I2 replay: PASS.

2. E1c: ablation ladder
arm	what it is	seeds	final accuracy % per seed	mean	eval loss	forward-eq. per round	uplink MB per round	paper (sd)
A-b0	HO-FL baseline, uniform sampling (equal-weight mean over all reports)	42	86.00	86.00	0.4968	90.0	11.34	85.51 (0.43)
C-b0	fo_mean, uniform sampling: each layer averages its first-order reports only	42	87.78	87.78	0.3884	90.0	11.34	
A-b0.75	HO-FL baseline, biased sampling (the HO-FL default)	42	85.99	85.99	0.4509	90.0	16.85	86.69 (0.56)
Ax2-b0	step-size control: baseline with the server step doubled	42	87.72	87.72	0.3798	90.0	11.34	
Ax3-b0	step-size control: baseline with the server step tripled	42	88.59	88.59	0.3504	90.0	11.34	
B-b0	fo_diluted without zeroth-order updates, uniform sampling: A minus B is the net contribution of the zeroth-order updates	42	85.82	85.82	0.4975	56.0	11.34	
B-b0.75	fo_diluted without zeroth-order updates, biased sampling	42	85.95	85.95	0.4513	68.7	16.85	
fedavg	FedAvg: every client trains every layer first-order (upper bound)	42	88.33	88.33	0.3532	90.0	26.15	88.15 (0.21)
decomfl	DeComFL: every client trains every layer zeroth-order (lower bound)	42	55.70	55.70	1.1707	150.0	0.00035	55.83 (0.08)
C-b0.75	fo_mean, biased sampling	not run						
D-b0	fo_mean without zeroth-order updates, uniform sampling: partial-layer first-order training only	not run						
D-b0.75	fo_mean without zeroth-order updates, biased sampling	not run						
Forward-equivalents: one forward pass through all blocks for one batch; a backward pass costs two (accounting model, not a measurement).

3. H1a: is the gain of biased sampling a matter of dilution?
seed	G_C = C(b0) - A(b0)	G_beta = A(b0.75) - A(b0)	G_lr = max(Ax2, Ax3) - A(b0)
42	+1.78	-0.01	+2.59
mean	+1.78	-0.01	+2.59
Rule of the brief: supported when mean G_C >= 0.7 x mean G_beta and mean G_lr < 0.5 x mean G_C. CANNOT_JUDGE (preliminary: 1 of 3 seeds): biased sampling gave no gain over uniform sampling, so there is nothing to recover.

4. H1: how much do the zeroth-order updates contribute?
Model level, A (zeroth-order on) minus B (off, same weight 1/K):

sampling	seeds	A - B per seed	mean	seed sd of A	verdict
b0	42	+0.18	+0.18	n/a	CANNOT_JUDGE (one seed gives no seed-to-seed standard deviation)
b0.75	42	+0.04	+0.04	n/a	CANNOT_JUDGE (one seed gives no seed-to-seed standard deviation)
Report level (probe): each client's actual update against its first-order counterfactual from the same state and batches. zo: all layers below the client's cut pooled; fo: all layers at or above it (self-check, close to 1). The null value is 1/sqrt(dimension), the scatter of the cosine of a random direction.

qwen-s42-A-b0:

round	tier	layers	clients	mean cosine	standard error	null 1/sqrt(d)	median norm ratio actual/first-order
1	T1	fo	1	0.9993	n/a	0.0011	1.001
1	T1	zo	1	0.0008618	n/a	0.0018	0.754
1	T1	zo_lora_A	1	-0.0013	n/a	0.0023	0.825
1	T1	zo_lora_B	1	0.0039	n/a	0.0030	0.666
1	T2	fo	4	0.9994	0.0001858	0.0015	1.000
1	T2	zo	4	-0.000344	0.0002397	0.0013	0.770
1	T2	zo_lora_A	4	-0.0009543	0.0006971	0.0016	0.855
1	T2	zo_lora_B	4	0.0004504	0.0012	0.0021	0.670
1	T3	fo	1	0.9997	n/a	0.0025	0.999
1	T3	zo	1	-0.0004465	n/a	0.0010	0.664
1	T3	zo_lora_A	1	-3.353e-05	n/a	0.0013	0.757
1	T3	zo_lora_B	1	-0.0009825	n/a	0.0017	0.562
20	T1	fo	3	0.9997	6.005e-05	0.0011	1.000
20	T1	zo	3	0.0002314	0.0014	0.0018	0.692
20	T1	zo_lora_A	3	0.0001775	0.0017	0.0023	0.712
20	T1	zo_lora_B	3	0.0003118	0.0015	0.0030	0.659
20	T2	fo	1	0.9998	n/a	0.0015	1.000
20	T2	zo	1	0.0017	n/a	0.0013	0.670
20	T2	zo_lora_A	1	0.0017	n/a	0.0016	0.646
20	T2	zo_lora_B	1	0.0017	n/a	0.0021	0.717
20	T3	fo	2	0.9996	7.424e-05	0.0025	1.008
20	T3	zo	2	-0.0006318	0.0002899	0.0010	0.648
20	T3	zo_lora_A	2	-0.0007057	0.000153	0.0013	0.663
20	T3	zo_lora_B	2	-0.0004834	0.0010	0.0017	0.625
40	T1	fo	3	0.9997	1.62e-05	0.0011	1.000
40	T1	zo	3	0.0018	0.0012	0.0018	0.712
40	T1	zo_lora_A	3	0.000755	0.0010	0.0023	0.734
40	T1	zo_lora_B	3	0.0034	0.0017	0.0030	0.680
40	T2	fo	2	0.9998	1.664e-05	0.0015	1.003
40	T2	zo	2	-0.0005191	0.0017	0.0013	0.640
40	T2	zo_lora_A	2	-0.00074	0.0018	0.0016	0.637
40	T2	zo_lora_B	2	-0.0001324	0.0014	0.0021	0.645
40	T3	fo	1	0.9997	n/a	0.0025	1.004
40	T3	zo	1	-0.0002503	n/a	0.0010	0.634
40	T3	zo_lora_A	1	-0.0025	n/a	0.0013	0.642
40	T3	zo_lora_B	1	0.0036	n/a	0.0017	0.622
80	T1	fo	1	0.9997	n/a	0.0011	1.000
80	T1	zo	1	-0.0024	n/a	0.0018	0.751
80	T1	zo_lora_A	1	-0.0010	n/a	0.0023	0.753
80	T1	zo_lora_B	1	-0.0046	n/a	0.0030	0.747
80	T2	fo	1	0.9990	n/a	0.0015	1.006
80	T2	zo	1	0.0019	n/a	0.0013	0.891
80	T2	zo_lora_A	1	0.0019	n/a	0.0016	0.895
80	T2	zo_lora_B	1	0.0019	n/a	0.0021	0.883
80	T3	fo	4	0.9992	0.0002208	0.0025	1.009
80	T3	zo	4	0.0003897	0.0006442	0.0010	0.710
80	T3	zo_lora_A	4	0.0006774	0.0005077	0.0013	0.704
80	T3	zo_lora_B	4	-0.0001105	0.0008885	0.0017	0.732
160	T1	fo	1	0.9993	n/a	0.0011	1.003
160	T1	zo	1	0.0007881	n/a	0.0018	0.855
160	T1	zo_lora_A	1	0.0031	n/a	0.0023	0.857
160	T1	zo_lora_B	1	-0.0032	n/a	0.0030	0.852
160	T2	fo	3	0.9979	0.0007415	0.0015	1.012
160	T2	zo	3	-0.0004569	0.0011	0.0013	0.848
160	T2	zo_lora_A	3	-0.0003023	0.0014	0.0016	0.843
160	T2	zo_lora_B	3	-0.0007275	0.0011	0.0021	0.857
160	T3	fo	2	0.9991	0.0003201	0.0025	1.007
160	T3	zo	2	-8.641e-05	0.0004929	0.0010	0.990
160	T3	zo_lora_A	2	0.0001574	0.0008453	0.0013	0.994
160	T3	zo_lora_B	2	-0.0004763	9.007e-05	0.0017	0.986
qwen-s42-A-b0.75:

round	tier	layers	clients	mean cosine	standard error	null 1/sqrt(d)	median norm ratio actual/first-order
1	T1	fo	6	0.9994	7.541e-05	0.0011	1.000
1	T1	zo	6	0.0020	0.0006622	0.0018	0.784
1	T1	zo_lora_A	6	0.0010	0.0010	0.0023	0.863
1	T1	zo_lora_B	6	0.0033	0.0009487	0.0030	0.685
20	T1	fo	4	0.9998	1.886e-05	0.0011	1.000
20	T1	zo	4	-0.0005219	0.0009192	0.0018	0.722
20	T1	zo_lora_A	4	-0.0011	0.0006492	0.0023	0.746
20	T1	zo_lora_B	4	0.0004527	0.0014	0.0030	0.693
20	T3	fo	2	0.9996	0.0001815	0.0025	1.003
20	T3	zo	2	0.0004851	1.238e-05	0.0010	0.653
20	T3	zo_lora_A	2	-2.117e-05	0.0006007	0.0013	0.644
20	T3	zo_lora_B	2	0.0013	0.0011	0.0017	0.669
40	T1	fo	5	0.9997	4.417e-05	0.0011	1.001
40	T1	zo	5	0.0011	0.0012	0.0018	0.714
40	T1	zo_lora_A	5	0.0003309	0.0012	0.0023	0.728
40	T1	zo_lora_B	5	0.0022	0.0019	0.0030	0.694
40	T3	fo	1	0.9995	n/a	0.0025	1.010
40	T3	zo	1	0.0019	n/a	0.0010	0.729
40	T3	zo_lora_A	1	0.0020	n/a	0.0013	0.714
40	T3	zo_lora_B	1	0.0016	n/a	0.0017	0.757
80	T1	fo	5	0.9990	0.0002018	0.0011	0.999
80	T1	zo	5	0.0002179	0.0011	0.0018	0.910
80	T1	zo_lora_A	5	0.0010	0.0011	0.0023	0.910
80	T1	zo_lora_B	5	-0.0012	0.0018	0.0030	0.911
80	T3	fo	1	0.9988	n/a	0.0025	1.007
80	T3	zo	1	0.0006084	n/a	0.0010	0.756
80	T3	zo_lora_A	1	0.0006388	n/a	0.0013	0.731
80	T3	zo_lora_B	1	0.0005529	n/a	0.0017	0.805
160	T1	fo	6	0.9980	0.0010	0.0011	0.999
160	T1	zo	6	6.912e-05	0.0005907	0.0018	0.843
160	T1	zo_lora_A	6	0.0003069	0.0007511	0.0023	0.844
160	T1	zo_lora_B	6	-0.0003375	0.0007256	0.0030	0.844
Reading: H1 expects the zo cosine in the 1e-3 to 1e-2 range and the norm ratio between 0.3 and 3. With AdamW every coordinate moves by about the learning rate per step whatever the gradient, so the norm ratio says little; the cosine and A minus B carry the test. zo_lora_A at round 1 has no first-order direction to compare with (LoRA B starts at zero, so the gradient of A is zero at the first step).

5. Layer statistics (averages over rounds and over the layers of a segment)
S0 lies below every cut (no first-order source), S1 to S3 start at the three cuts. Norms are of the per-layer update vectors.

arm	seed	segment	first-order reports	zeroth-order reports	norm of a first-order component	norm of a zeroth-order component	norm of the first-order mean	norm of the zeroth-order mean	cosine FO-FO	cosine ZO-ZO	cosine FO-ZO	norm of the applied update	electorate rule
A-b0	42	S0	0	6	n/a	0.00510	n/a	0.00209	n/a	-9.692e-05	n/a	0.00209	all 100%
A-b0	42	S1	2.03	3.97	0.00714	0.00511	0.00595	0.00265	0.313	2.64e-05	5.895e-05	0.00271	all 100%
A-b0	42	S2	4.08	1.93	0.00729	0.00512	0.00484	0.00377	0.247	0.0003431	-9.225e-05	0.00350	all 100%
A-b0	42	S3	6	0	0.00735	n/a	0.00414	n/a	0.189	n/a	n/a	0.00414	all 100%
C-b0	42	S0	0	6	n/a	0.00510	n/a	0.00208	n/a	3.863e-05	n/a	0.00208	thin_all 100%
C-b0	42	S1	2.03	3.97	0.00674	0.00510	0.00532	0.00264	0.170	9.846e-05	0.0001153	0.00512	fo 94%, thin_all 6%
C-b0	42	S2	4.08	1.93	0.00685	0.00510	0.00418	0.00376	0.152	7.488e-05	-3.306e-05	0.00418	fo 100%
C-b0	42	S3	6	0	0.00684	n/a	0.00366	n/a	0.148	n/a	n/a	0.00366	fo 100%
A-b0.75	42	S0	0	6	n/a	0.00510	n/a	0.00208	n/a	2.335e-05	n/a	0.00208	all 100%
A-b0.75	42	S1	4.97	1.02	0.00678	0.00508	0.00394	0.00444	0.177	-5.366e-05	0.0001105	0.00339	all 100%
A-b0.75	42	S2	5.56	0.44	0.00694	0.00507	0.00348	0.00487	0.102	-6.556e-05	6.051e-05	0.00328	all 100%
A-b0.75	42	S3	6	0	0.00702	n/a	0.00356	n/a	0.122	n/a	n/a	0.00356	all 100%
Ax2-b0	42	S0	0	6	n/a	0.00510	n/a	0.00208	n/a	1.572e-05	n/a	0.00417	all 100%
Ax2-b0	42	S1	2.03	3.97	0.00665	0.00510	0.00522	0.00264	0.160	7.038e-06	5.241e-05	0.00498	all 100%
Ax2-b0	42	S2	4.08	1.93	0.00677	0.00511	0.00404	0.00377	0.134	8.134e-05	-4.771e-05	0.00600	all 100%
Ax2-b0	42	S3	6	0	0.00682	n/a	0.00349	n/a	0.121	n/a	n/a	0.00698	all 100%
Ax3-b0	42	S0	0	6	n/a	0.00510	n/a	0.00209	n/a	-7.596e-06	n/a	0.00626	all 100%
Ax3-b0	42	S1	2.03	3.97	0.00649	0.00511	0.00494	0.00264	0.091	-8.721e-06	9.823e-05	0.00722	all 100%
Ax3-b0	42	S2	4.08	1.93	0.00659	0.00512	0.00376	0.00377	0.090	2.116e-05	-6.862e-05	0.00843	all 100%
Ax3-b0	42	S3	6	0	0.00667	n/a	0.00322	n/a	0.088	n/a	n/a	0.00966	all 100%
B-b0	42	S0	0	0	n/a	n/a	n/a	n/a	n/a	n/a	n/a	0.00000	empty 100%
B-b0	42	S1	2.03	0	0.00715	n/a	0.00595	n/a	0.311	n/a	n/a	0.00195	empty 6%, fo 94%
B-b0	42	S2	4.08	0	0.00729	n/a	0.00484	n/a	0.247	n/a	n/a	0.00324	fo 100%
B-b0	42	S3	6	0	0.00735	n/a	0.00414	n/a	0.189	n/a	n/a	0.00414	fo 100%
B-b0.75	42	S0	0	0	n/a	n/a	n/a	n/a	n/a	n/a	n/a	0.00000	empty 100%
B-b0.75	42	S1	4.97	0	0.00678	n/a	0.00394	n/a	0.178	n/a	n/a	0.00324	fo 100%
B-b0.75	42	S2	5.56	0	0.00694	n/a	0.00348	n/a	0.103	n/a	n/a	0.00322	fo 100%
B-b0.75	42	S3	6	0	0.00702	n/a	0.00356	n/a	0.121	n/a	n/a	0.00356	fo 100%
6. First-order supply
Supply map (E1a, no training):

Supply map for qwen: three tiers of 10 clients, cuts [8, 16, 24], depth 28.
S1, S2, S3 are the segments that start at the three cuts; width = first-order reports per round.

dep_round_v1 beta=0
  expected width S1 / S2 / S3: 2.00 (brief 2.00) / 4.00 (brief 4.00) / 6.00 (brief 6.00)
  effective sources (N_eff) S1 / S2 / S3: 10.00 (brief 10.00) / 20.00 (brief 20.00) / 30.00 (brief 30.00)
  S1 largest single share: 10.0% (brief 10.0%); S1 share of the 6 largest sources: 60.0% (brief 60.0%); largest presence of a client: 20.0% (brief 20.0%)
  simulated rounds with S1 width 0 / 1 / <=2 / <3: 6.51% (brief 6.40%) / 25.90% (brief 26.50%) / 69.14% (brief 69.20%) / 69.14%
  participation bias term N*sum((p_i-1/N)^2): 0.000 (brief 0.000); sampling-weighted d_ZO/q: 311296 (brief 311296)

dep_round_v1 beta=0.75
  expected width S1 / S2 / S3: 5.00 (brief 5.00) / 5.50 (brief 5.50) / 6.00 (brief 6.00)
  effective sources (N_eff) S1 / S2 / S3: 6.49 (brief 6.49) / 7.81 (brief 7.81) / 9.23 (brief 9.23)
  S1 largest single share: 16.0% (brief 16.0%); S1 share of the 6 largest sources: 96.0% (brief 96.0%); largest presence of a client: 80.0% (brief 80.0%)
  simulated rounds with S1 width 0 / 1 / <=2 / <3: 0.00% (brief 0.00%) / 0.04% (brief 0.01%) / 0.41% (brief 0.40%) / 0.41%
  participation bias term N*sum((p_i-1/N)^2): 2.250 (brief 2.250); sampling-weighted d_ZO/q: 194560 (brief 194560)

tier_equalised beta=0.75
  expected width S1 / S2 / S3: 5.00 (brief 5.00) / 5.50 (brief 5.50) / 6.00 (brief 6.00)
  effective sources (N_eff) S1 / S2 / S3: 10.00 (brief 10.00) / 11.98 (brief 11.98) / 14.12 (brief 14.12)
  S1 largest single share: 10.0% (brief 10.0%); S1 share of the 6 largest sources: 60.0% (brief 60.0%); largest presence of a client: 50.0% (brief 50.0%)
  simulated rounds with S1 width 0 / 1 / <=2 / <3: 0.00% (brief 0.00%) / 0.02% (brief 0.02%) / 0.41% (brief 0.40%) / 0.41%
  participation bias term N*sum((p_i-1/N)^2): 1.125 (brief 1.125); sampling-weighted d_ZO/q: 194560 (brief 194560)

tier_quota 3,1,2
  expected width S1 / S2 / S3: 3.00 (brief 3.00) / 4.00 (brief 4.00) / 6.00 (brief 6.00)
  effective sources (N_eff) S1 / S2 / S3: 10.00 (brief 10.00) / 16.00 (brief 16.00) / 25.71 (brief 25.71)
  S1 largest single share: 10.0% (brief 10.0%); S1 share of the 6 largest sources: 60.0% (brief 60.0%); largest presence of a client: 30.0% (brief 30.0%)
  simulated rounds with S1 width 0 / 1 / <=2 / <3: 0.00% (brief 0.00%) / 0.00% (brief 0.00%) / 0.00% (brief 0.00%) / 0.00%
  participation bias term N*sum((p_i-1/N)^2): 0.167 (brief 0.167); sampling-weighted d_ZO/q: 285355 (brief 285355)

tier_quota 4,1,1
  expected width S1 / S2 / S3: 4.00 (brief 4.00) / 5.00 (brief 5.00) / 6.00 (brief 6.00)
  effective sources (N_eff) S1 / S2 / S3: 10.00 (brief 10.00) / 14.71 (brief 14.71) / 20.00 (brief 20.00)
  S1 largest single share: 10.0% (brief 10.0%); S1 share of the 6 largest sources: 60.0% (brief 60.0%); largest presence of a client: 40.0% (brief 40.0%)
  simulated rounds with S1 width 0 / 1 / <=2 / <3: 0.00% (brief 0.00%) / 0.00% (brief 0.00%) / 0.00% (brief 0.00%) / 0.00%
  participation bias term N*sum((p_i-1/N)^2): 0.500 (brief 0.500); sampling-weighted d_ZO/q: 233472 (brief 233472)

tier_quota 2,2,2
  expected width S1 / S2 / S3: 2.00 / 4.00 / 6.00
  effective sources (N_eff) S1 / S2 / S3: 10.00 / 20.00 / 30.00
  S1 largest single share: 10.0%; S1 share of the 6 largest sources: 60.0%; largest presence of a client: 20.0%
  simulated rounds with S1 width 0 / 1 / <=2 / <3: 0.00% / 0.00% / 100.00% / 100.00%
  participation bias term N*sum((p_i-1/N)^2): 0.000; sampling-weighted d_ZO/q: 311296
Realised in the runs (one run scatters around the expectations above; the largest share is biased upwards):

run	mean width S1 / S2 / S3	rounds with S1 width 0	S1 width <= 2	S1 sources	S1 N_eff	largest presence
qwen-s42-A-b0	2.03 / 4.08 / 6.00	6.2%	65.6%	10	9.82	25.6%
qwen-s42-C-b0	2.03 / 4.08 / 6.00	6.2%	65.6%	10	9.82	25.6%
qwen-s42-A-b0.75	4.97 / 5.56 / 6.00	0.0%	0.6%	10	6.45	83.8%
qwen-s42-Ax2-b0	2.03 / 4.08 / 6.00	6.2%	65.6%	10	9.82	25.6%
qwen-s42-Ax3-b0	2.03 / 4.08 / 6.00	6.2%	65.6%	10	9.82	25.6%
qwen-s42-B-b0	2.03 / 4.08 / 6.00	6.2%	65.6%	10	9.82	25.6%
qwen-s42-B-b0.75	4.97 / 5.56 / 6.00	0.0%	0.6%	10	6.45	83.8%
7. Queue, timing and GPU hours
Runs at a time: 2 (requested 2).

run	block	status	wall min	estimated min	note
qwen-s42-A-b0	1	COMPLETED	21.9	29.7	
qwen-s42-C-b0	1	COMPLETED	21.1	29.7	
qwen-s42-A-b0.75	1	COMPLETED	21.3	26.4	
qwen-s42-Ax2-b0	1	COMPLETED	21.0	26.9	
qwen-s42-Ax3-b0	1	COMPLETED	10.6	26.8	
qwen-s42-B-b0	2	COMPLETED	16.1	19.2	
qwen-s42-B-b0.75	2	COMPLETED	17.1	19.2	
qwen-s42-fedavg	2	COMPLETED	17.6	22.7	
qwen-s42-decomfl	2	COMPLETED	23.1	39.4	
Measured alone on the GPU (three-round preflight, or a full run when one run at a time is used):

cost profile	s per round	s per evaluation	start-up s	peak GPU memory GiB	estimated min per 160-round run	measured on
qwen/decomfl	4.20	11.81	4.6	4.4	14.6	preflight
qwen/fedavg	2.00	11.61	4.8	14.5	8.7	preflight
qwen/hofl	3.00	11.81	31.4	9.8	11.9	preflight
qwen/hofl-zo-off	1.60	11.81	4.6	8.6	7.7	preflight
While sharing the GPU: qwen/decomfl: 8.41 s per round with 2 runs at a time; qwen/fedavg: 4.00 s per round with 2 runs at a time; qwen/hofl: 2.60 s per round with 2 runs at a time; qwen/hofl-zo-off: 3.80 s per round with 2 runs at a time.

Wall time of all runs recorded under this root: 2.88 h (sum over runs; with several runs at a time the session is shorter).

8. What this means and what to do next
E0 check, A-b0: 86.00 against the paper's 85.51 (+0.49 points, 1.1 paper standard deviations, 1 of 3 seeds; the acceptance rule max(1.0, 2 sd) applies to the mean over all seeds).
E0 check, A-b0.75: 85.99 against the paper's 86.69 (-0.70 points, 1.3 paper standard deviations, 1 of 3 seeds; the acceptance rule max(1.0, 2 sd) applies to the mean over all seeds).
E0 check, fedavg: 88.33 against the paper's 88.15 (+0.18 points, 0.9 paper standard deviations, 1 of 3 seeds; the acceptance rule max(1.0, 2 sd) applies to the mean over all seeds).
E0 check, decomfl: 55.70 against the paper's 55.83 (-0.13 points, 1.7 paper standard deviations, 1 of 3 seeds; the acceptance rule max(1.0, 2 sd) applies to the mean over all seeds).
H1a cannot be judged: biased sampling gave no gain over uniform sampling, so there is nothing to recover.
H1 (b0): switching the zeroth-order updates off changes the final accuracy by -0.18 points (CANNOT_JUDGE, one seed gives no seed-to-seed standard deviation).
H1 (b0.75): switching the zeroth-order updates off changes the final accuracy by -0.04 points (CANNOT_JUDGE, one seed gives no seed-to-seed standard deviation).
Probe, A-b0 seed 42: pooled zeroth-order cosine between -0.0024 and +0.0019 over rounds and tiers (null scatter about 0.0014); first-order self-check cosine at least 0.9979.
Probe, A-b0.75 seed 42: pooled zeroth-order cosine between -0.0005 and +0.0020 over rounds and tiers (null scatter about 0.0015); first-order self-check cosine at least 0.9980.
Gate G1 needs the H1 and H1a results for the research lead's decision; bring this report back.
result bundle: download started
Downloading "feedback_P1_bundle.zip": 

$ /content/venv/bin/python experiments/p1/p1.py queue --root /content/work/p1 --cache-dir /content/cache --device cuda --model qwen --seeds 42 --blocks 3 --jobs 2 --deadline 1795233849.7437687
== queue (qwen, seeds [42], blocks [3], 3 runs of 160 rounds, up to 2 at a time, nvidia-rtx-pro-6000-blackwell-server-edition-torch2.7.1) ==
  [qwen-s42-C-b0.75] starting: ho_fl.train --model qwen --dataset ag_news --partition dirichlet_05 --rounds 160 --method hofl --beta 0.75 --agg-electorate fo --agg-weight count --layer-stats --seed 42 --device cuda --cache-dir /content/cache --output /content/work/p1/runs/qwen-s42-C-b0.75
  [qwen-s42-D-b0] starting: ho_fl.train --model qwen --dataset ag_news --partition dirichlet_05 --rounds 160 --method hofl --beta 0.0 --zo-mode off --agg-electorate fo --agg-weight count --layer-stats --seed 42 --device cuda --cache-dir /content/cache --output /content/work/p1/runs/qwen-s42-D-b0
    [qwen-s42-C-b0.75] eval @ round 0: accuracy=0.5579 task_loss=1.1715 t=25s
    [qwen-s42-C-b0.75] round 1 train_loss=1.3112 t=30s
    [qwen-s42-D-b0] eval @ round 0: accuracy=0.5579 task_loss=1.1715 t=30s
    [qwen-s42-D-b0] round 1 train_loss=1.1499 t=33s
    [qwen-s42-D-b0] round 10 train_loss=1.1698 t=64s
    [qwen-s42-C-b0.75] round 10 train_loss=1.1986 t=76s
    [qwen-s42-D-b0] eval @ round 10: accuracy=0.6261 task_loss=1.1130 t=88s
    [qwen-s42-C-b0.75] eval @ round 10: accuracy=0.6233 task_loss=1.1163 t=101s
    [qwen-s42-D-b0] round 20 train_loss=0.9984 t=121s
    [qwen-s42-D-b0] eval @ round 20: accuracy=0.7126 task_loss=0.9970 t=145s
    [qwen-s42-C-b0.75] round 20 train_loss=1.0064 t=153s
    [qwen-s42-C-b0.75] eval @ round 20: accuracy=0.7087 task_loss=0.9924 t=176s
    [qwen-s42-D-b0] round 30 train_loss=0.8529 t=181s
    [qwen-s42-D-b0] eval @ round 30: accuracy=0.7659 task_loss=0.8515 t=204s
    [qwen-s42-C-b0.75] round 30 train_loss=0.7864 t=228s
    [qwen-s42-D-b0] round 40 train_loss=0.6473 t=239s
    [qwen-s42-C-b0.75] eval @ round 30: accuracy=0.7461 task_loss=0.8326 t=252s
    [qwen-s42-D-b0] eval @ round 40: accuracy=0.8089 task_loss=0.7082 t=263s
    [qwen-s42-D-b0] round 50 train_loss=0.6029 t=297s
    [qwen-s42-C-b0.75] round 40 train_loss=0.6746 t=304s
    [qwen-s42-D-b0] eval @ round 50: accuracy=0.8345 task_loss=0.5891 t=321s
    [qwen-s42-C-b0.75] eval @ round 40: accuracy=0.7863 task_loss=0.6774 t=329s
    [qwen-s42-D-b0] round 60 train_loss=0.5762 t=355s
    [qwen-s42-C-b0.75] round 50 train_loss=0.5285 t=381s
    [qwen-s42-D-b0] eval @ round 60: accuracy=0.8511 task_loss=0.5318 t=378s
    [qwen-s42-C-b0.75] eval @ round 50: accuracy=0.8183 task_loss=0.5864 t=404s
    [qwen-s42-D-b0] round 70 train_loss=0.5188 t=414s
    [qwen-s42-D-b0] eval @ round 70: accuracy=0.8588 task_loss=0.5157 t=437s
    [qwen-s42-C-b0.75] round 60 train_loss=0.4488 t=455s
    [qwen-s42-D-b0] round 80 train_loss=0.3901 t=472s
    [qwen-s42-C-b0.75] eval @ round 60: accuracy=0.8396 task_loss=0.5548 t=478s
    [qwen-s42-D-b0] eval @ round 80: accuracy=0.8530 task_loss=0.5053 t=495s
    [qwen-s42-C-b0.75] round 70 train_loss=0.4092 t=530s
    [qwen-s42-D-b0] round 90 train_loss=0.4050 t=527s
    [qwen-s42-C-b0.75] eval @ round 70: accuracy=0.8446 task_loss=0.5500 t=555s
    [qwen-s42-D-b0] eval @ round 90: accuracy=0.8651 task_loss=0.4765 t=553s
    [qwen-s42-D-b0] round 100 train_loss=0.3651 t=587s
    [qwen-s42-C-b0.75] round 80 train_loss=0.4276 t=606s
    [qwen-s42-D-b0] eval @ round 100: accuracy=0.8670 task_loss=0.4564 t=610s
    [qwen-s42-C-b0.75] eval @ round 80: accuracy=0.8464 task_loss=0.5319 t=630s
    [qwen-s42-D-b0] round 110 train_loss=0.5343 t=646s
    [qwen-s42-D-b0] eval @ round 110: accuracy=0.8678 task_loss=0.4591 t=669s
    [qwen-s42-C-b0.75] round 90 train_loss=0.4119 t=682s
    [qwen-s42-C-b0.75] eval @ round 90: accuracy=0.8454 task_loss=0.5353 t=705s
    [qwen-s42-D-b0] round 120 train_loss=0.4621 t=705s
    [qwen-s42-D-b0] eval @ round 120: accuracy=0.8613 task_loss=0.4820 t=728s
    [qwen-s42-C-b0.75] round 100 train_loss=0.5225 t=757s
    [qwen-s42-D-b0] round 130 train_loss=0.2927 t=762s
    [qwen-s42-C-b0.75] eval @ round 100: accuracy=0.8541 task_loss=0.5093 t=782s
    [qwen-s42-D-b0] eval @ round 130: accuracy=0.8612 task_loss=0.4697 t=787s
    [qwen-s42-D-b0] round 140 train_loss=0.3859 t=820s
    [qwen-s42-C-b0.75] round 110 train_loss=0.4770 t=833s
    [qwen-s42-D-b0] eval @ round 140: accuracy=0.8670 task_loss=0.4392 t=845s
    [qwen-s42-C-b0.75] eval @ round 110: accuracy=0.8534 task_loss=0.4820 t=857s
    [qwen-s42-D-b0] round 150 train_loss=0.3496 t=878s
    [qwen-s42-D-b0] eval @ round 150: accuracy=0.8646 task_loss=0.4287 t=902s
    [qwen-s42-C-b0.75] round 120 train_loss=0.3392 t=909s
    [qwen-s42-C-b0.75] eval @ round 120: accuracy=0.8604 task_loss=0.4622 t=932s
    [qwen-s42-D-b0] round 160 train_loss=0.3450 t=936s
    [qwen-s42-D-b0] eval @ round 160: accuracy=0.8778 task_loss=0.3893 t=960s
  [qwen-s42-D-b0] completed in 960s
  [qwen-s42-D-b0.75] starting: ho_fl.train --model qwen --dataset ag_news --partition dirichlet_05 --rounds 160 --method hofl --beta 0.75 --zo-mode off --agg-electorate fo --agg-weight count --layer-stats --seed 42 --device cuda --cache-dir /content/cache --output /content/work/p1/runs/qwen-s42-D-b0.75
    [qwen-s42-C-b0.75] round 130 train_loss=0.3575 t=979s
    [qwen-s42-D-b0.75] eval @ round 0: accuracy=0.5579 task_loss=1.1715 t=29s
    [qwen-s42-D-b0.75] round 1 train_loss=1.3122 t=34s
    [qwen-s42-C-b0.75] eval @ round 130: accuracy=0.8589 task_loss=0.4647 t=1003s
    [qwen-s42-D-b0.75] round 10 train_loss=1.1988 t=68s
    [qwen-s42-D-b0.75] eval @ round 10: accuracy=0.6255 task_loss=1.1160 t=91s
    [qwen-s42-C-b0.75] round 140 train_loss=0.3962 t=1055s
    [qwen-s42-C-b0.75] eval @ round 140: accuracy=0.8546 task_loss=0.4750 t=1079s
    [qwen-s42-D-b0.75] round 20 train_loss=1.0102 t=131s
    [qwen-s42-D-b0.75] eval @ round 20: accuracy=0.7092 task_loss=0.9929 t=155s
    [qwen-s42-C-b0.75] round 150 train_loss=0.3560 t=1132s
    [qwen-s42-C-b0.75] eval @ round 150: accuracy=0.8537 task_loss=0.4772 t=1155s
    [qwen-s42-D-b0.75] round 30 train_loss=0.7844 t=194s
    [qwen-s42-D-b0.75] eval @ round 30: accuracy=0.7462 task_loss=0.8327 t=217s
    [qwen-s42-C-b0.75] round 160 train_loss=0.4042 t=1208s
    [qwen-s42-D-b0.75] round 40 train_loss=0.6743 t=256s
    [qwen-s42-C-b0.75] eval @ round 160: accuracy=0.8620 task_loss=0.4305 t=1232s
  [qwen-s42-C-b0.75] completed in 1233s
    [qwen-s42-D-b0.75] eval @ round 40: accuracy=0.7850 task_loss=0.6780 t=274s
    [qwen-s42-D-b0.75] round 50 train_loss=0.5286 t=292s
    [qwen-s42-D-b0.75] eval @ round 50: accuracy=0.8182 task_loss=0.5872 t=303s
    [qwen-s42-D-b0.75] round 60 train_loss=0.4547 t=322s
    [qwen-s42-D-b0.75] eval @ round 60: accuracy=0.8363 task_loss=0.5554 t=333s
    [qwen-s42-D-b0.75] round 70 train_loss=0.4042 t=351s
    [qwen-s42-D-b0.75] eval @ round 70: accuracy=0.8453 task_loss=0.5491 t=363s
    [qwen-s42-D-b0.75] round 80 train_loss=0.4294 t=381s
    [qwen-s42-D-b0.75] eval @ round 80: accuracy=0.8468 task_loss=0.5300 t=393s
    [qwen-s42-D-b0.75] round 90 train_loss=0.4135 t=412s
    [qwen-s42-D-b0.75] eval @ round 90: accuracy=0.8453 task_loss=0.5338 t=423s
    [qwen-s42-D-b0.75] round 100 train_loss=0.5203 t=440s
    [qwen-s42-D-b0.75] eval @ round 100: accuracy=0.8532 task_loss=0.5099 t=452s
    [qwen-s42-D-b0.75] round 110 train_loss=0.4739 t=470s
    [qwen-s42-D-b0.75] eval @ round 110: accuracy=0.8543 task_loss=0.4823 t=482s
    [qwen-s42-D-b0.75] round 120 train_loss=0.3405 t=500s
    [qwen-s42-D-b0.75] eval @ round 120: accuracy=0.8611 task_loss=0.4619 t=512s
    [qwen-s42-D-b0.75] round 130 train_loss=0.3585 t=530s
    [qwen-s42-D-b0.75] eval @ round 130: accuracy=0.8607 task_loss=0.4655 t=542s
    [qwen-s42-D-b0.75] round 140 train_loss=0.4004 t=560s
    [qwen-s42-D-b0.75] eval @ round 140: accuracy=0.8542 task_loss=0.4753 t=571s
    [qwen-s42-D-b0.75] round 150 train_loss=0.3596 t=589s
    [qwen-s42-D-b0.75] eval @ round 150: accuracy=0.8541 task_loss=0.4767 t=601s
    [qwen-s42-D-b0.75] round 160 train_loss=0.4072 t=619s
    [qwen-s42-D-b0.75] eval @ round 160: accuracy=0.8616 task_loss=0.4319 t=631s
  [qwen-s42-D-b0.75] completed in 631s

| run | status | wall min | estimate min | note |
| --- | --- | --- | --- | --- |
| qwen-s42-C-b0.75 | COMPLETED | 20.5 | 29.7 |  |
| qwen-s42-D-b0 | COMPLETED | 16.0 | 19.2 |  |
| qwen-s42-D-b0.75 | COMPLETED | 10.5 | 20.0 |  |

stage results: queue=ok
3.34 h of budget left
$ /content/venv/bin/python experiments/p1/p1.py feedback --root /content/work/p1 --cache-dir /content/cache --device cuda --model qwen --seeds 42 --quiet
== feedback ==
H1a: CANNOT_JUDGE; H1: b0 CANNOT_JUDGE, b0.75 CANNOT_JUDGE; 12 of 12 runs completed
written: /content/work/p1/feedback/feedback_P1.md, CSV files and /content/work/p1/feedback/feedback_P1_bundle.zip

stage results: feedback=ok
P1 feedback
Generated 2026-11-21T00:53:55+00:00 for revision cdea00b095d2 on nvidia-rtx-pro-6000-blackwell-server-edition-torch2.7.1. Model qwen, AG News Dirichlet 0.5, seeds [42], 160 rounds per run; 12 of 12 planned runs completed.

1. Baseline regression in this session
I1: {'decomfl': 'PASS_BITWISE', 'fedavg': 'PASS_BITWISE', 'hofl': 'PASS_BITWISE'} (golden mode: bitwise); I2 replay: PASS.

2. E1c: ablation ladder
arm	what it is	seeds	final accuracy % per seed	mean	eval loss	forward-eq. per round	uplink MB per round	paper (sd)
A-b0	HO-FL baseline, uniform sampling (equal-weight mean over all reports)	42	86.00	86.00	0.4968	90.0	11.34	85.51 (0.43)
C-b0	fo_mean, uniform sampling: each layer averages its first-order reports only	42	87.78	87.78	0.3884	90.0	11.34	
A-b0.75	HO-FL baseline, biased sampling (the HO-FL default)	42	85.99	85.99	0.4509	90.0	16.85	86.69 (0.56)
Ax2-b0	step-size control: baseline with the server step doubled	42	87.72	87.72	0.3798	90.0	11.34	
Ax3-b0	step-size control: baseline with the server step tripled	42	88.59	88.59	0.3504	90.0	11.34	
B-b0	fo_diluted without zeroth-order updates, uniform sampling: A minus B is the net contribution of the zeroth-order updates	42	85.82	85.82	0.4975	56.0	11.34	
B-b0.75	fo_diluted without zeroth-order updates, biased sampling	42	85.95	85.95	0.4513	68.7	16.85	
fedavg	FedAvg: every client trains every layer first-order (upper bound)	42	88.33	88.33	0.3532	90.0	26.15	88.15 (0.21)
decomfl	DeComFL: every client trains every layer zeroth-order (lower bound)	42	55.70	55.70	1.1707	150.0	0.00035	55.83 (0.08)
C-b0.75	fo_mean, biased sampling	42	86.20	86.20	0.4305	90.0	16.85	
D-b0	fo_mean without zeroth-order updates, uniform sampling: partial-layer first-order training only	42	87.78	87.78	0.3893	56.0	11.34	
D-b0.75	fo_mean without zeroth-order updates, biased sampling	42	86.16	86.16	0.4319	68.7	16.85	
Forward-equivalents: one forward pass through all blocks for one batch; a backward pass costs two (accounting model, not a measurement).

3. H1a: is the gain of biased sampling a matter of dilution?
seed	G_C = C(b0) - A(b0)	G_beta = A(b0.75) - A(b0)	G_lr = max(Ax2, Ax3) - A(b0)
42	+1.78	-0.01	+2.59
mean	+1.78	-0.01	+2.59
Rule of the brief: supported when mean G_C >= 0.7 x mean G_beta and mean G_lr < 0.5 x mean G_C. CANNOT_JUDGE (preliminary: 1 of 3 seeds): biased sampling gave no gain over uniform sampling, so there is nothing to recover.

4. H1: how much do the zeroth-order updates contribute?
Model level, A (zeroth-order on) minus B (off, same weight 1/K):

sampling	seeds	A - B per seed	mean	seed sd of A	verdict
b0	42	+0.18	+0.18	n/a	CANNOT_JUDGE (one seed gives no seed-to-seed standard deviation)
b0.75	42	+0.04	+0.04	n/a	CANNOT_JUDGE (one seed gives no seed-to-seed standard deviation)
Report level (probe): each client's actual update against its first-order counterfactual from the same state and batches. zo: all layers below the client's cut pooled; fo: all layers at or above it (self-check, close to 1). The null value is 1/sqrt(dimension), the scatter of the cosine of a random direction.

qwen-s42-A-b0:

round	tier	layers	clients	mean cosine	standard error	null 1/sqrt(d)	median norm ratio actual/first-order
1	T1	fo	1	0.9993	n/a	0.0011	1.001
1	T1	zo	1	0.0008618	n/a	0.0018	0.754
1	T1	zo_lora_A	1	-0.0013	n/a	0.0023	0.825
1	T1	zo_lora_B	1	0.0039	n/a	0.0030	0.666
1	T2	fo	4	0.9994	0.0001858	0.0015	1.000
1	T2	zo	4	-0.000344	0.0002397	0.0013	0.770
1	T2	zo_lora_A	4	-0.0009543	0.0006971	0.0016	0.855
1	T2	zo_lora_B	4	0.0004504	0.0012	0.0021	0.670
1	T3	fo	1	0.9997	n/a	0.0025	0.999
1	T3	zo	1	-0.0004465	n/a	0.0010	0.664
1	T3	zo_lora_A	1	-3.353e-05	n/a	0.0013	0.757
1	T3	zo_lora_B	1	-0.0009825	n/a	0.0017	0.562
20	T1	fo	3	0.9997	6.005e-05	0.0011	1.000
20	T1	zo	3	0.0002314	0.0014	0.0018	0.692
20	T1	zo_lora_A	3	0.0001775	0.0017	0.0023	0.712
20	T1	zo_lora_B	3	0.0003118	0.0015	0.0030	0.659
20	T2	fo	1	0.9998	n/a	0.0015	1.000
20	T2	zo	1	0.0017	n/a	0.0013	0.670
20	T2	zo_lora_A	1	0.0017	n/a	0.0016	0.646
20	T2	zo_lora_B	1	0.0017	n/a	0.0021	0.717
20	T3	fo	2	0.9996	7.424e-05	0.0025	1.008
20	T3	zo	2	-0.0006318	0.0002899	0.0010	0.648
20	T3	zo_lora_A	2	-0.0007057	0.000153	0.0013	0.663
20	T3	zo_lora_B	2	-0.0004834	0.0010	0.0017	0.625
40	T1	fo	3	0.9997	1.62e-05	0.0011	1.000
40	T1	zo	3	0.0018	0.0012	0.0018	0.712
40	T1	zo_lora_A	3	0.000755	0.0010	0.0023	0.734
40	T1	zo_lora_B	3	0.0034	0.0017	0.0030	0.680
40	T2	fo	2	0.9998	1.664e-05	0.0015	1.003
40	T2	zo	2	-0.0005191	0.0017	0.0013	0.640
40	T2	zo_lora_A	2	-0.00074	0.0018	0.0016	0.637
40	T2	zo_lora_B	2	-0.0001324	0.0014	0.0021	0.645
40	T3	fo	1	0.9997	n/a	0.0025	1.004
40	T3	zo	1	-0.0002503	n/a	0.0010	0.634
40	T3	zo_lora_A	1	-0.0025	n/a	0.0013	0.642
40	T3	zo_lora_B	1	0.0036	n/a	0.0017	0.622
80	T1	fo	1	0.9997	n/a	0.0011	1.000
80	T1	zo	1	-0.0024	n/a	0.0018	0.751
80	T1	zo_lora_A	1	-0.0010	n/a	0.0023	0.753
80	T1	zo_lora_B	1	-0.0046	n/a	0.0030	0.747
80	T2	fo	1	0.9990	n/a	0.0015	1.006
80	T2	zo	1	0.0019	n/a	0.0013	0.891
80	T2	zo_lora_A	1	0.0019	n/a	0.0016	0.895
80	T2	zo_lora_B	1	0.0019	n/a	0.0021	0.883
80	T3	fo	4	0.9992	0.0002208	0.0025	1.009
80	T3	zo	4	0.0003897	0.0006442	0.0010	0.710
80	T3	zo_lora_A	4	0.0006774	0.0005077	0.0013	0.704
80	T3	zo_lora_B	4	-0.0001105	0.0008885	0.0017	0.732
160	T1	fo	1	0.9993	n/a	0.0011	1.003
160	T1	zo	1	0.0007881	n/a	0.0018	0.855
160	T1	zo_lora_A	1	0.0031	n/a	0.0023	0.857
160	T1	zo_lora_B	1	-0.0032	n/a	0.0030	0.852
160	T2	fo	3	0.9979	0.0007415	0.0015	1.012
160	T2	zo	3	-0.0004569	0.0011	0.0013	0.848
160	T2	zo_lora_A	3	-0.0003023	0.0014	0.0016	0.843
160	T2	zo_lora_B	3	-0.0007275	0.0011	0.0021	0.857
160	T3	fo	2	0.9991	0.0003201	0.0025	1.007
160	T3	zo	2	-8.641e-05	0.0004929	0.0010	0.990
160	T3	zo_lora_A	2	0.0001574	0.0008453	0.0013	0.994
160	T3	zo_lora_B	2	-0.0004763	9.007e-05	0.0017	0.986
qwen-s42-A-b0.75:

round	tier	layers	clients	mean cosine	standard error	null 1/sqrt(d)	median norm ratio actual/first-order
1	T1	fo	6	0.9994	7.541e-05	0.0011	1.000
1	T1	zo	6	0.0020	0.0006622	0.0018	0.784
1	T1	zo_lora_A	6	0.0010	0.0010	0.0023	0.863
1	T1	zo_lora_B	6	0.0033	0.0009487	0.0030	0.685
20	T1	fo	4	0.9998	1.886e-05	0.0011	1.000
20	T1	zo	4	-0.0005219	0.0009192	0.0018	0.722
20	T1	zo_lora_A	4	-0.0011	0.0006492	0.0023	0.746
20	T1	zo_lora_B	4	0.0004527	0.0014	0.0030	0.693
20	T3	fo	2	0.9996	0.0001815	0.0025	1.003
20	T3	zo	2	0.0004851	1.238e-05	0.0010	0.653
20	T3	zo_lora_A	2	-2.117e-05	0.0006007	0.0013	0.644
20	T3	zo_lora_B	2	0.0013	0.0011	0.0017	0.669
40	T1	fo	5	0.9997	4.417e-05	0.0011	1.001
40	T1	zo	5	0.0011	0.0012	0.0018	0.714
40	T1	zo_lora_A	5	0.0003309	0.0012	0.0023	0.728
40	T1	zo_lora_B	5	0.0022	0.0019	0.0030	0.694
40	T3	fo	1	0.9995	n/a	0.0025	1.010
40	T3	zo	1	0.0019	n/a	0.0010	0.729
40	T3	zo_lora_A	1	0.0020	n/a	0.0013	0.714
40	T3	zo_lora_B	1	0.0016	n/a	0.0017	0.757
80	T1	fo	5	0.9990	0.0002018	0.0011	0.999
80	T1	zo	5	0.0002179	0.0011	0.0018	0.910
80	T1	zo_lora_A	5	0.0010	0.0011	0.0023	0.910
80	T1	zo_lora_B	5	-0.0012	0.0018	0.0030	0.911
80	T3	fo	1	0.9988	n/a	0.0025	1.007
80	T3	zo	1	0.0006084	n/a	0.0010	0.756
80	T3	zo_lora_A	1	0.0006388	n/a	0.0013	0.731
80	T3	zo_lora_B	1	0.0005529	n/a	0.0017	0.805
160	T1	fo	6	0.9980	0.0010	0.0011	0.999
160	T1	zo	6	6.912e-05	0.0005907	0.0018	0.843
160	T1	zo_lora_A	6	0.0003069	0.0007511	0.0023	0.844
160	T1	zo_lora_B	6	-0.0003375	0.0007256	0.0030	0.844
Reading: H1 expects the zo cosine in the 1e-3 to 1e-2 range and the norm ratio between 0.3 and 3. With AdamW every coordinate moves by about the learning rate per step whatever the gradient, so the norm ratio says little; the cosine and A minus B carry the test. zo_lora_A at round 1 has no first-order direction to compare with (LoRA B starts at zero, so the gradient of A is zero at the first step).

5. Layer statistics (averages over rounds and over the layers of a segment)
S0 lies below every cut (no first-order source), S1 to S3 start at the three cuts. Norms are of the per-layer update vectors.

arm	seed	segment	first-order reports	zeroth-order reports	norm of a first-order component	norm of a zeroth-order component	norm of the first-order mean	norm of the zeroth-order mean	cosine FO-FO	cosine ZO-ZO	cosine FO-ZO	norm of the applied update	electorate rule
A-b0	42	S0	0	6	n/a	0.00510	n/a	0.00209	n/a	-9.692e-05	n/a	0.00209	all 100%
A-b0	42	S1	2.03	3.97	0.00714	0.00511	0.00595	0.00265	0.313	2.64e-05	5.895e-05	0.00271	all 100%
A-b0	42	S2	4.08	1.93	0.00729	0.00512	0.00484	0.00377	0.247	0.0003431	-9.225e-05	0.00350	all 100%
A-b0	42	S3	6	0	0.00735	n/a	0.00414	n/a	0.189	n/a	n/a	0.00414	all 100%
C-b0	42	S0	0	6	n/a	0.00510	n/a	0.00208	n/a	3.863e-05	n/a	0.00208	thin_all 100%
C-b0	42	S1	2.03	3.97	0.00674	0.00510	0.00532	0.00264	0.170	9.846e-05	0.0001153	0.00512	fo 94%, thin_all 6%
C-b0	42	S2	4.08	1.93	0.00685	0.00510	0.00418	0.00376	0.152	7.488e-05	-3.306e-05	0.00418	fo 100%
C-b0	42	S3	6	0	0.00684	n/a	0.00366	n/a	0.148	n/a	n/a	0.00366	fo 100%
A-b0.75	42	S0	0	6	n/a	0.00510	n/a	0.00208	n/a	2.335e-05	n/a	0.00208	all 100%
A-b0.75	42	S1	4.97	1.02	0.00678	0.00508	0.00394	0.00444	0.177	-5.366e-05	0.0001105	0.00339	all 100%
A-b0.75	42	S2	5.56	0.44	0.00694	0.00507	0.00348	0.00487	0.102	-6.556e-05	6.051e-05	0.00328	all 100%
A-b0.75	42	S3	6	0	0.00702	n/a	0.00356	n/a	0.122	n/a	n/a	0.00356	all 100%
Ax2-b0	42	S0	0	6	n/a	0.00510	n/a	0.00208	n/a	1.572e-05	n/a	0.00417	all 100%
Ax2-b0	42	S1	2.03	3.97	0.00665	0.00510	0.00522	0.00264	0.160	7.038e-06	5.241e-05	0.00498	all 100%
Ax2-b0	42	S2	4.08	1.93	0.00677	0.00511	0.00404	0.00377	0.134	8.134e-05	-4.771e-05	0.00600	all 100%
Ax2-b0	42	S3	6	0	0.00682	n/a	0.00349	n/a	0.121	n/a	n/a	0.00698	all 100%
Ax3-b0	42	S0	0	6	n/a	0.00510	n/a	0.00209	n/a	-7.596e-06	n/a	0.00626	all 100%
Ax3-b0	42	S1	2.03	3.97	0.00649	0.00511	0.00494	0.00264	0.091	-8.721e-06	9.823e-05	0.00722	all 100%
Ax3-b0	42	S2	4.08	1.93	0.00659	0.00512	0.00376	0.00377	0.090	2.116e-05	-6.862e-05	0.00843	all 100%
Ax3-b0	42	S3	6	0	0.00667	n/a	0.00322	n/a	0.088	n/a	n/a	0.00966	all 100%
B-b0	42	S0	0	0	n/a	n/a	n/a	n/a	n/a	n/a	n/a	0.00000	empty 100%
B-b0	42	S1	2.03	0	0.00715	n/a	0.00595	n/a	0.311	n/a	n/a	0.00195	empty 6%, fo 94%
B-b0	42	S2	4.08	0	0.00729	n/a	0.00484	n/a	0.247	n/a	n/a	0.00324	fo 100%
B-b0	42	S3	6	0	0.00735	n/a	0.00414	n/a	0.189	n/a	n/a	0.00414	fo 100%
B-b0.75	42	S0	0	0	n/a	n/a	n/a	n/a	n/a	n/a	n/a	0.00000	empty 100%
B-b0.75	42	S1	4.97	0	0.00678	n/a	0.00394	n/a	0.178	n/a	n/a	0.00324	fo 100%
B-b0.75	42	S2	5.56	0	0.00694	n/a	0.00348	n/a	0.103	n/a	n/a	0.00322	fo 100%
B-b0.75	42	S3	6	0	0.00702	n/a	0.00356	n/a	0.121	n/a	n/a	0.00356	fo 100%
C-b0.75	42	S0	0	6	n/a	0.00510	n/a	0.00209	n/a	-1.228e-05	n/a	0.00209	thin_all 100%
C-b0.75	42	S1	4.97	1.02	0.00669	0.00510	0.00378	0.00445	0.154	-0.0002664	7.434e-05	0.00378	fo 100%
C-b0.75	42	S2	5.56	0.44	0.00686	0.00512	0.00335	0.00491	0.088	0.0014	-7.913e-05	0.00335	fo 100%
C-b0.75	42	S3	6	0	0.00694	n/a	0.00346	n/a	0.112	n/a	n/a	0.00346	fo 100%
D-b0	42	S0	0	0	n/a	n/a	n/a	n/a	n/a	n/a	n/a	0.00000	empty 100%
D-b0	42	S1	2.03	0	0.00673	n/a	0.00531	n/a	0.168	n/a	n/a	0.00498	empty 6%, fo 94%
D-b0	42	S2	4.08	0	0.00684	n/a	0.00418	n/a	0.152	n/a	n/a	0.00418	fo 100%
D-b0	42	S3	6	0	0.00684	n/a	0.00366	n/a	0.148	n/a	n/a	0.00366	fo 100%
D-b0.75	42	S0	0	0	n/a	n/a	n/a	n/a	n/a	n/a	n/a	0.00000	empty 100%
D-b0.75	42	S1	4.97	0	0.00669	n/a	0.00378	n/a	0.154	n/a	n/a	0.00378	fo 100%
D-b0.75	42	S2	5.56	0	0.00686	n/a	0.00335	n/a	0.087	n/a	n/a	0.00335	fo 100%
D-b0.75	42	S3	6	0	0.00694	n/a	0.00346	n/a	0.112	n/a	n/a	0.00346	fo 100%
6. First-order supply
Supply map (E1a, no training):

Supply map for qwen: three tiers of 10 clients, cuts [8, 16, 24], depth 28.
S1, S2, S3 are the segments that start at the three cuts; width = first-order reports per round.

dep_round_v1 beta=0
  expected width S1 / S2 / S3: 2.00 (brief 2.00) / 4.00 (brief 4.00) / 6.00 (brief 6.00)
  effective sources (N_eff) S1 / S2 / S3: 10.00 (brief 10.00) / 20.00 (brief 20.00) / 30.00 (brief 30.00)
  S1 largest single share: 10.0% (brief 10.0%); S1 share of the 6 largest sources: 60.0% (brief 60.0%); largest presence of a client: 20.0% (brief 20.0%)
  simulated rounds with S1 width 0 / 1 / <=2 / <3: 6.51% (brief 6.40%) / 25.90% (brief 26.50%) / 69.14% (brief 69.20%) / 69.14%
  participation bias term N*sum((p_i-1/N)^2): 0.000 (brief 0.000); sampling-weighted d_ZO/q: 311296 (brief 311296)

dep_round_v1 beta=0.75
  expected width S1 / S2 / S3: 5.00 (brief 5.00) / 5.50 (brief 5.50) / 6.00 (brief 6.00)
  effective sources (N_eff) S1 / S2 / S3: 6.49 (brief 6.49) / 7.81 (brief 7.81) / 9.23 (brief 9.23)
  S1 largest single share: 16.0% (brief 16.0%); S1 share of the 6 largest sources: 96.0% (brief 96.0%); largest presence of a client: 80.0% (brief 80.0%)
  simulated rounds with S1 width 0 / 1 / <=2 / <3: 0.00% (brief 0.00%) / 0.04% (brief 0.01%) / 0.41% (brief 0.40%) / 0.41%
  participation bias term N*sum((p_i-1/N)^2): 2.250 (brief 2.250); sampling-weighted d_ZO/q: 194560 (brief 194560)

tier_equalised beta=0.75
  expected width S1 / S2 / S3: 5.00 (brief 5.00) / 5.50 (brief 5.50) / 6.00 (brief 6.00)
  effective sources (N_eff) S1 / S2 / S3: 10.00 (brief 10.00) / 11.98 (brief 11.98) / 14.12 (brief 14.12)
  S1 largest single share: 10.0% (brief 10.0%); S1 share of the 6 largest sources: 60.0% (brief 60.0%); largest presence of a client: 50.0% (brief 50.0%)
  simulated rounds with S1 width 0 / 1 / <=2 / <3: 0.00% (brief 0.00%) / 0.02% (brief 0.02%) / 0.41% (brief 0.40%) / 0.41%
  participation bias term N*sum((p_i-1/N)^2): 1.125 (brief 1.125); sampling-weighted d_ZO/q: 194560 (brief 194560)

tier_quota 3,1,2
  expected width S1 / S2 / S3: 3.00 (brief 3.00) / 4.00 (brief 4.00) / 6.00 (brief 6.00)
  effective sources (N_eff) S1 / S2 / S3: 10.00 (brief 10.00) / 16.00 (brief 16.00) / 25.71 (brief 25.71)
  S1 largest single share: 10.0% (brief 10.0%); S1 share of the 6 largest sources: 60.0% (brief 60.0%); largest presence of a client: 30.0% (brief 30.0%)
  simulated rounds with S1 width 0 / 1 / <=2 / <3: 0.00% (brief 0.00%) / 0.00% (brief 0.00%) / 0.00% (brief 0.00%) / 0.00%
  participation bias term N*sum((p_i-1/N)^2): 0.167 (brief 0.167); sampling-weighted d_ZO/q: 285355 (brief 285355)

tier_quota 4,1,1
  expected width S1 / S2 / S3: 4.00 (brief 4.00) / 5.00 (brief 5.00) / 6.00 (brief 6.00)
  effective sources (N_eff) S1 / S2 / S3: 10.00 (brief 10.00) / 14.71 (brief 14.71) / 20.00 (brief 20.00)
  S1 largest single share: 10.0% (brief 10.0%); S1 share of the 6 largest sources: 60.0% (brief 60.0%); largest presence of a client: 40.0% (brief 40.0%)
  simulated rounds with S1 width 0 / 1 / <=2 / <3: 0.00% (brief 0.00%) / 0.00% (brief 0.00%) / 0.00% (brief 0.00%) / 0.00%
  participation bias term N*sum((p_i-1/N)^2): 0.500 (brief 0.500); sampling-weighted d_ZO/q: 233472 (brief 233472)

tier_quota 2,2,2
  expected width S1 / S2 / S3: 2.00 / 4.00 / 6.00
  effective sources (N_eff) S1 / S2 / S3: 10.00 / 20.00 / 30.00
  S1 largest single share: 10.0%; S1 share of the 6 largest sources: 60.0%; largest presence of a client: 20.0%
  simulated rounds with S1 width 0 / 1 / <=2 / <3: 0.00% / 0.00% / 100.00% / 100.00%
  participation bias term N*sum((p_i-1/N)^2): 0.000; sampling-weighted d_ZO/q: 311296
Realised in the runs (one run scatters around the expectations above; the largest share is biased upwards):

run	mean width S1 / S2 / S3	rounds with S1 width 0	S1 width <= 2	S1 sources	S1 N_eff	largest presence
qwen-s42-A-b0	2.03 / 4.08 / 6.00	6.2%	65.6%	10	9.82	25.6%
qwen-s42-C-b0	2.03 / 4.08 / 6.00	6.2%	65.6%	10	9.82	25.6%
qwen-s42-A-b0.75	4.97 / 5.56 / 6.00	0.0%	0.6%	10	6.45	83.8%
qwen-s42-Ax2-b0	2.03 / 4.08 / 6.00	6.2%	65.6%	10	9.82	25.6%
qwen-s42-Ax3-b0	2.03 / 4.08 / 6.00	6.2%	65.6%	10	9.82	25.6%
qwen-s42-B-b0	2.03 / 4.08 / 6.00	6.2%	65.6%	10	9.82	25.6%
qwen-s42-B-b0.75	4.97 / 5.56 / 6.00	0.0%	0.6%	10	6.45	83.8%
qwen-s42-C-b0.75	4.97 / 5.56 / 6.00	0.0%	0.6%	10	6.45	83.8%
qwen-s42-D-b0	2.03 / 4.08 / 6.00	6.2%	65.6%	10	9.82	25.6%
qwen-s42-D-b0.75	4.97 / 5.56 / 6.00	0.0%	0.6%	10	6.45	83.8%
7. Queue, timing and GPU hours
Runs at a time: 2 (requested 2).

run	block	status	wall min	estimated min	note
qwen-s42-A-b0	1	COMPLETED	21.9	29.7	
qwen-s42-C-b0	1	COMPLETED	21.1	29.7	
qwen-s42-A-b0.75	1	COMPLETED	21.3	26.4	
qwen-s42-Ax2-b0	1	COMPLETED	21.0	26.9	
qwen-s42-Ax3-b0	1	COMPLETED	10.6	26.8	
qwen-s42-B-b0	2	COMPLETED	16.1	19.2	
qwen-s42-B-b0.75	2	COMPLETED	17.1	19.2	
qwen-s42-fedavg	2	COMPLETED	17.6	22.7	
qwen-s42-decomfl	2	COMPLETED	23.1	39.4	
qwen-s42-C-b0.75	3	COMPLETED	20.5	29.7	
qwen-s42-D-b0	3	COMPLETED	16.0	19.2	
qwen-s42-D-b0.75	3	COMPLETED	10.5	20.0	
Measured alone on the GPU (three-round preflight, or a full run when one run at a time is used):

cost profile	s per round	s per evaluation	start-up s	peak GPU memory GiB	estimated min per 160-round run	measured on
qwen/decomfl	4.20	11.81	4.6	4.4	14.6	preflight
qwen/fedavg	2.00	11.61	4.8	14.5	8.7	preflight
qwen/hofl	3.00	11.81	31.4	9.8	11.9	preflight
qwen/hofl-zo-off	1.60	11.81	4.6	8.6	7.7	preflight
While sharing the GPU: qwen/decomfl: 8.41 s per round with 2 runs at a time; qwen/fedavg: 4.00 s per round with 2 runs at a time; qwen/hofl: 5.20 s per round with 2 runs at a time; qwen/hofl-zo-off: 1.80 s per round with 2 runs at a time.

Wall time of all runs recorded under this root: 3.66 h (sum over runs; with several runs at a time the session is shorter).

8. What this means and what to do next
E0 check, A-b0: 86.00 against the paper's 85.51 (+0.49 points, 1.1 paper standard deviations, 1 of 3 seeds; the acceptance rule max(1.0, 2 sd) applies to the mean over all seeds).
E0 check, A-b0.75: 85.99 against the paper's 86.69 (-0.70 points, 1.3 paper standard deviations, 1 of 3 seeds; the acceptance rule max(1.0, 2 sd) applies to the mean over all seeds).
E0 check, fedavg: 88.33 against the paper's 88.15 (+0.18 points, 0.9 paper standard deviations, 1 of 3 seeds; the acceptance rule max(1.0, 2 sd) applies to the mean over all seeds).
E0 check, decomfl: 55.70 against the paper's 55.83 (-0.13 points, 1.7 paper standard deviations, 1 of 3 seeds; the acceptance rule max(1.0, 2 sd) applies to the mean over all seeds).
H1a cannot be judged: biased sampling gave no gain over uniform sampling, so there is nothing to recover.
H1 (b0): switching the zeroth-order updates off changes the final accuracy by -0.18 points (CANNOT_JUDGE, one seed gives no seed-to-seed standard deviation).
H1 (b0.75): switching the zeroth-order updates off changes the final accuracy by -0.04 points (CANNOT_JUDGE, one seed gives no seed-to-seed standard deviation).
Probe, A-b0 seed 42: pooled zeroth-order cosine between -0.0024 and +0.0019 over rounds and tiers (null scatter about 0.0014); first-order self-check cosine at least 0.9979.
Probe, A-b0.75 seed 42: pooled zeroth-order cosine between -0.0005 and +0.0020 over rounds and tiers (null scatter about 0.0015); first-order self-check cosine at least 0.9980.
Gate G1 needs the H1 and H1a results for the research lead's decision; bring this report back.
result bundle: download started
Downloading "feedback_P1_bundle.zip": 