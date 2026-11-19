[![make.fun](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/romdhoncom/gta6-trophy-rarity-forecast-01/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/romdhoncom/gta6-trophy-rarity-forecast-01/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01)

# GTA VI trophy rarity forecast

Forecasts every GTA VI (PS5) trophy's Exophase earn rate at a case's target,
as five quantiles (0.05, 0.25, 0.5, 0.75, 0.95) of the rate in percent.

## Model

An earn rate is a ratio — earners over players tracked — and Exophase
refreshes the two on its own schedule, so the program forecasts them
separately and divides at the end. For the players tracked and for each
trophy's earners it fits a weighted least-squares line of the log count
against log time since release (recent reads weighted more) and reads each
line at the target. A trophy's slope is shrunk toward the pooled slope
across trophies, so a rare trophy with a flat history borrows the common
trend instead of trusting its own noise.

The spread is honest about three things: how far each trophy's own history
strays from its line, a random walk that widens with the horizon, and the
jump a refresh makes (measured from the history's own refreshes, with a
prior when the history has seen none yet). Quantiles are the median's logit
shifted by the standard normal's quantiles times that spread.

## Run

```sh
mkdir -p out
MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out node main.ts
```

Node 22, standard library only, no network. Deterministic: two runs of a
case produce byte-identical output.
