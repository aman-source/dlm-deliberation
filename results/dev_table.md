# Dev sweep (protocol v1.1)

Mean off_label_mass. ECE: 15 bins. ECE-TS: one temperature per (model, cell), fit on dev pooled over datasets.

## arc_c

| model | cell | n | acc | ECE | ECE-TS | off_label | eos_pad (median) | NFE |
|---|---|---|---|---|---|---|---|---|
| llada-8b-instruct | C0 | 100 | 0.810 | 0.105 | 0.091 | 0.002 | - | 1 |
| llada-8b-instruct | C1 S=32 T=1 | 100 | 0.810 | 0.150 | 0.156 | 0.003 | 0.000 | 2 |
| llada-8b-instruct | C1 S=32 T=4 | 100 | 0.810 | 0.153 | 0.108 | 0.001 | 0.000 | 5 |
| llada-8b-instruct | C1 S=32 T=16 | 100 | 0.810 | 0.185 | 0.115 | 0.001 | 0.000 | 17 |
| llada-8b-instruct | C1 S=128 T=1 | 100 | 0.790 | 0.147 | 0.105 | 0.003 | 0.000 | 2 |
| llada-8b-instruct | C1 S=128 T=4 | 100 | 0.810 | 0.153 | 0.134 | 0.003 | 0.000 | 5 |
| llada-8b-instruct | C1 S=128 T=16 | 100 | 0.800 | 0.181 | 0.123 | 0.003 | 0.000 | 17 |
| llada-8b-instruct | C2 S=32 T=1 | 100 | 0.820 | 0.118 | 0.120 | 0.007 | 0.062 | 2 |
| llada-8b-instruct | C2 S=32 T=4 | 100 | 0.780 | 0.193 | 0.092 | 0.005 | 0.062 | 5 |
| llada-8b-instruct | C2 S=32 T=16 | 100 | 0.810 | 0.184 | 0.110 | 0.001 | 0.062 | 17 |
| llada-8b-instruct | C2 S=128 T=1 | 100 | 0.860 | 0.106 | 0.088 | 0.003 | 0.137 | 2 |
| llada-8b-instruct | C2 S=128 T=4 | 100 | 0.840 | 0.125 | 0.116 | 0.002 | 0.023 | 5 |
| llada-8b-instruct | C2 S=128 T=16 | 100 | 0.850 | 0.145 | 0.147 | 0.001 | 0.016 | 17 |
| qwen3-8b | C0 | 100 | 0.910 | 0.081 | 0.230 | 0.121 | - | 1 |

## boolq

| model | cell | n | acc | ECE | ECE-TS | off_label | eos_pad (median) | NFE |
|---|---|---|---|---|---|---|---|---|
| llada-8b-instruct | C0 | 100 | 0.780 | 0.137 | 0.067 | 0.021 | - | 1 |
| llada-8b-instruct | C1 S=32 T=1 | 100 | 0.830 | 0.126 | 0.079 | 0.035 | 0.000 | 2 |
| llada-8b-instruct | C1 S=32 T=4 | 100 | 0.820 | 0.160 | 0.057 | 0.019 | 0.000 | 5 |
| llada-8b-instruct | C1 S=32 T=16 | 100 | 0.830 | 0.176 | 0.093 | 0.011 | 0.000 | 17 |
| llada-8b-instruct | C1 S=128 T=1 | 100 | 0.760 | 0.165 | 0.066 | 0.048 | 0.000 | 2 |
| llada-8b-instruct | C1 S=128 T=4 | 100 | 0.770 | 0.210 | 0.066 | 0.041 | 0.000 | 5 |
| llada-8b-instruct | C1 S=128 T=16 | 100 | 0.800 | 0.202 | 0.035 | 0.021 | 0.000 | 17 |
| llada-8b-instruct | C2 S=32 T=1 | 100 | 0.810 | 0.160 | 0.065 | 0.036 | 0.062 | 2 |
| llada-8b-instruct | C2 S=32 T=4 | 100 | 0.820 | 0.159 | 0.075 | 0.027 | 0.062 | 5 |
| llada-8b-instruct | C2 S=32 T=16 | 100 | 0.790 | 0.196 | 0.042 | 0.013 | 0.062 | 17 |
| llada-8b-instruct | C2 S=128 T=1 | 100 | 0.820 | 0.126 | 0.069 | 0.029 | 0.098 | 2 |
| llada-8b-instruct | C2 S=128 T=4 | 100 | 0.820 | 0.147 | 0.094 | 0.025 | 0.023 | 5 |
| llada-8b-instruct | C2 S=128 T=16 | 100 | 0.780 | 0.236 | 0.142 | 0.025 | 0.016 | 17 |
| qwen3-8b | C0 | 100 | 0.760 | 0.227 | 0.157 | 0.000 | - | 1 |

## jagged

| model | cell | n | acc | ECE | ECE-TS | off_label | eos_pad (median) | NFE |
|---|---|---|---|---|---|---|---|---|
| llada-8b-instruct | C0 | 60 | 0.667 | 0.161 | 0.143 | 0.014 | - | 1 |
| llada-8b-instruct | C1 S=32 T=1 | 60 | 0.700 | 0.156 | 0.097 | 0.020 | 0.000 | 2 |
| llada-8b-instruct | C1 S=32 T=4 | 60 | 0.733 | 0.203 | 0.130 | 0.011 | 0.000 | 5 |
| llada-8b-instruct | C1 S=32 T=16 | 60 | 0.717 | 0.258 | 0.120 | 0.010 | 0.000 | 17 |
| llada-8b-instruct | C1 S=128 T=1 | 60 | 0.600 | 0.370 | 0.258 | 0.018 | 0.000 | 2 |
| llada-8b-instruct | C1 S=128 T=4 | 60 | 0.600 | 0.383 | 0.170 | 0.013 | 0.000 | 5 |
| llada-8b-instruct | C1 S=128 T=16 | 60 | 0.550 | 0.432 | 0.184 | 0.010 | 0.000 | 17 |
| llada-8b-instruct | C2 S=32 T=1 | 60 | 0.667 | 0.287 | 0.188 | 0.013 | 0.062 | 2 |
| llada-8b-instruct | C2 S=32 T=4 | 60 | 0.700 | 0.283 | 0.096 | 0.008 | 0.062 | 5 |
| llada-8b-instruct | C2 S=32 T=16 | 60 | 0.700 | 0.284 | 0.093 | 0.004 | 0.062 | 17 |
| llada-8b-instruct | C2 S=128 T=1 | 60 | 0.567 | 0.397 | 0.219 | 0.007 | 0.031 | 2 |
| llada-8b-instruct | C2 S=128 T=4 | 60 | 0.567 | 0.405 | 0.251 | 0.008 | 0.020 | 5 |
| llada-8b-instruct | C2 S=128 T=16 | 60 | 0.583 | 0.392 | 0.164 | 0.003 | 0.016 | 17 |
| qwen3-8b | C0 | 60 | 0.617 | 0.361 | 0.161 | 0.022 | - | 1 |

## strategyqa

| model | cell | n | acc | ECE | ECE-TS | off_label | eos_pad (median) | NFE |
|---|---|---|---|---|---|---|---|---|
| llada-8b-instruct | C0 | 100 | 0.610 | 0.129 | 0.090 | 0.023 | - | 1 |
| llada-8b-instruct | C1 S=32 T=1 | 100 | 0.610 | 0.136 | 0.063 | 0.046 | 0.000 | 2 |
| llada-8b-instruct | C1 S=32 T=4 | 100 | 0.650 | 0.259 | 0.138 | 0.029 | 0.000 | 5 |
| llada-8b-instruct | C1 S=32 T=16 | 100 | 0.690 | 0.255 | 0.146 | 0.013 | 0.000 | 17 |
| llada-8b-instruct | C1 S=128 T=1 | 100 | 0.570 | 0.198 | 0.097 | 0.061 | 0.000 | 2 |
| llada-8b-instruct | C1 S=128 T=4 | 100 | 0.580 | 0.257 | 0.118 | 0.048 | 0.000 | 5 |
| llada-8b-instruct | C1 S=128 T=16 | 100 | 0.660 | 0.295 | 0.156 | 0.023 | 0.000 | 17 |
| llada-8b-instruct | C2 S=32 T=1 | 100 | 0.670 | 0.133 | 0.062 | 0.054 | 0.062 | 2 |
| llada-8b-instruct | C2 S=32 T=4 | 100 | 0.640 | 0.254 | 0.096 | 0.042 | 0.062 | 5 |
| llada-8b-instruct | C2 S=32 T=16 | 100 | 0.670 | 0.306 | 0.129 | 0.008 | 0.062 | 17 |
| llada-8b-instruct | C2 S=128 T=1 | 100 | 0.650 | 0.212 | 0.111 | 0.029 | 0.270 | 2 |
| llada-8b-instruct | C2 S=128 T=4 | 100 | 0.590 | 0.252 | 0.109 | 0.029 | 0.430 | 5 |
| llada-8b-instruct | C2 S=128 T=16 | 100 | 0.650 | 0.318 | 0.115 | 0.015 | 0.016 | 17 |
| qwen3-8b | C0 | 100 | 0.770 | 0.204 | 0.112 | 0.001 | - | 1 |

## Fitted temperatures (dev)

| key | T | NLL | n_dev |
|---|---|---|---|
| llada-8b-instruct|C0|0|na | 1.783 | 0.561 | 360 |
| llada-8b-instruct|C1|128|1 | 3.031 | 0.678 | 360 |
| llada-8b-instruct|C1|128|16 | 4.407 | 0.687 | 360 |
| llada-8b-instruct|C1|128|4 | 3.417 | 0.692 | 360 |
| llada-8b-instruct|C1|32|1 | 2.224 | 0.611 | 360 |
| llada-8b-instruct|C1|32|16 | 4.082 | 0.623 | 360 |
| llada-8b-instruct|C1|32|4 | 3.233 | 0.606 | 360 |
| llada-8b-instruct|C2|128|1 | 2.522 | 0.601 | 360 |
| llada-8b-instruct|C2|128|16 | 3.512 | 0.682 | 360 |
| llada-8b-instruct|C2|128|4 | 2.756 | 0.628 | 360 |
| llada-8b-instruct|C2|32|1 | 2.170 | 0.598 | 360 |
| llada-8b-instruct|C2|32|16 | 3.253 | 0.628 | 360 |
| llada-8b-instruct|C2|32|4 | 2.994 | 0.625 | 360 |
| qwen3-8b|C0|0|na | 5.616 | 0.559 | 360 |

## Fitted tau (dev)

- tau: tau = 0.020, mean gap vs fixed frontier = 0.0046, mean NFE = 1.05
- tau_temperature_scaled: tau = 0.000, mean gap vs fixed frontier = 0.0000, mean NFE = 1.00

Selection rule: On dev, maximize the unweighted mean across datasets of (escalation accuracy - accuracy of the interpolated fixed-budget upper envelope) at the escalation rule's mean NFE. Ties break toward lower mean NFE, then lower tau. A second tau is fit the same way after per-cell temperature scaling. Section 7 did not name the selection objective; this one matches the dominance claim and is frozen before test.
