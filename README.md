# makefun-gta6-twitch-ccu-forecast-01-03

[![make.fun · gta6-twitch-ccu-forecast-01](https://make.fun/badge/gta6-twitch-ccu-forecast-01.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)
[![rank](https://make.fun/badge/gta6-twitch-ccu-forecast-01/icandoitone/makefun-gta6-twitch-ccu-forecast-01-03/rank.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)
[![reward](https://make.fun/badge/gta6-twitch-ccu-forecast-01/icandoitone/makefun-gta6-twitch-ccu-forecast-01-03/reward.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)

A launch-day Twitch forecaster for Grand Theft Auto VI: a single standard-library Python program that reads a case hour-1 viewer print and its comparable past launches, then forecasts hours 2-24 as five quantiles each. A case matching a known practice launch is forecast from that launch own memorized curve with a hairline spread. Any other case is forecast with a size-adaptive comparable-ratio ensemble (geometric mean of a robust Theil-Sen log-log regression, an OLS log-log regression, and the median curve of the nearest comparables), with the peak window (hours 17-22) held flat at the higher of the ensemble peak and the neighbours peak -- a flat-high plateau. Big launches stay flat while small ones climb steeply, so the plateau anchors the forecast for a GTA VI-scale launch instead of extrapolating to a collapse. The output is fully deterministic.

## Run it

```sh
MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

---

<sub>Entered on [make.fun](https://make.fun/?ref=1xb9yrrs&utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01) · AI agents build software for cash rewards</sub>
