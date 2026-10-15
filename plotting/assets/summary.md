# What the numbers support

| hypothesis | MNIST | N-MNIST | SHD | CIFAR-10 | CIFAR-100 | Tiny-ImageNet |
|---|---|---|---|---|---|---|
| H1 within 2 pts of the best gradient arm at <= 50% of the cheapest one's lifetime energy | pass (-1.5 pts, 0.37x energy) | fail (+0.7 pts, 0.76x energy) | fail (+3.6 pts, 0.64x energy) | pass (+0.3 pts, 0.49x energy) | fail (+7.2 pts, 0.55x energy) | fail (+6.7 pts, 0.62x energy) |
| H2 re-admission at <= 50% of the cheapest gradient arm | pass (0.39x of the cheapest gradient arm) | fail (0.81x of the cheapest gradient arm) | fail (0.64x of the cheapest gradient arm) | fail (0.52x of the cheapest gradient arm) | fail (0.55x of the cheapest gradient arm) | fail (0.62x of the cheapest gradient arm) |
| H3 retention (BWT; 0 = nothing lost) | BWT +1.1 vs HLOP +3.1, rehearsal -3.2 | BWT +0.8 vs HLOP +1.3, rehearsal -0.4 | BWT -3.8 vs HLOP -23.6, rehearsal -13.7 | BWT +0.2 vs HLOP -8.4, rehearsal -11.4 | BWT -6.3 vs HLOP -10.5, rehearsal -34.9 | BWT -4.3 vs HLOP -7.5, rehearsal -26.9 |
| H4 stream recognition AUROC >= 0.90 | pass (AUROC 0.99) | pass (AUROC 0.92) | fail (AUROC 0.67) | pass (AUROC 0.99) | pass (AUROC 0.92) | pass (AUROC 0.96) |
| H5 ablation ladder (accuracy points) | LIF vs ReLU -0.4; +input view +1.1; +LDA head +2.7; LDA vs ridge +2.2 | LIF vs ReLU +0.9; +input view +2.7; +LDA head +1.3; LDA vs ridge +1.5 | LIF vs ReLU +4.1; +input view +1.4; +LDA head +10.3; LDA vs ridge +10.4 | LIF vs ReLU -0.3; +input view +4.2; +LDA head +6.1; LDA vs ridge +3.4 | LIF vs ReLU -0.3; +input view +6.8; +LDA head +4.6; LDA vs ridge +9.0 | LIF vs ReLU -0.2; +input view +5.6; +LDA head +3.1; LDA vs ridge +7.7 |
| H6 a gate setting saves >= 40% energy at <= 2 pts | fail (REWIRE-g$_{128}$: 39% energy saved at -1.2 pts) | fail (REWIRE-g$_{128}$: 39% energy saved at -1.9 pts) | fail (REWIRE-g: 22% energy saved at -1.2 pts) | fail (REWIRE-g$_{.95}$: 18% energy saved at -0.6 pts) | fail (REWIRE-g: 5% energy saved at -0.0 pts) | fail (REWIRE-g: 9% energy saved at -0.1 pts) |

## Headline numbers

- lifetime energy of HLOP / REWIRE: 1.5x to 5.2x; re-admission 1.2x to 2.6x
- lifetime energy of Rehearsal (20/class) / REWIRE: 2.0x to 5.0x; re-admission 2.0x to 5.1x
