$ /content/venv/bin/python experiments/p1/p1.py feedback --root /content/work/p1 --cache-dir /content/cache --device cuda --model qwen --seeds 42 --quiet
== feedback ==
H1a: CANNOT_JUDGE; H1: b0 CANNOT_JUDGE, b0.75 CANNOT_JUDGE; 12 of 12 runs completed
written: /content/work/p1/feedback/feedback_P1.md, CSV files and /content/work/p1/feedback/feedback_P1_bundle.zip

stage results: feedback=ok
P1 feedback
Generated 2026-11-21T00:53:58+00:00 for revision cdea00b095d2 on nvidia-rtx-pro-6000-blackwell-server-edition-torch2.7.1. Model qwen, AG News Dirichlet 0.5, seeds [42], 160 rounds per run; 12 of 12 planned runs completed.

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

report: /content/work/p1/feedback/feedback_P1.md
session time used: 2.16 h