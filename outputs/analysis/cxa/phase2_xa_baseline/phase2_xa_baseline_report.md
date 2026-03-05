# cXA Phase 2 — xA Baseline analysis

Total passes: 240,105
Total assists: 369
Sum(xA Baseline): 369.0
Calibration factor: 0.9997

## Top players by xA Baseline (top 10)

|   player_id | passer_name                   | team_name   |   passes |   key_passes |   assists |   xa_baseline |    xa_mean |   assist_rate_per_pass |   assist_rate_per_key_pass |   xa_per_key_pass |
|------------:|:------------------------------|:------------|---------:|-------------:|----------:|--------------:|-----------:|-----------------------:|---------------------------:|------------------:|
|         226 | Antoine Griezmann             | France      |      957 |           46 |         5 |       3.46388 | 0.00361952 |             0.00522466 |                  0.108696  |         0.0753017 |
|         464 | Joshua Kimmich                | Germany     |     1208 |           42 |         3 |       3.38593 | 0.00280292 |             0.00248344 |                  0.0714286 |         0.0806173 |
|         357 | Luka Modrić                   | Croatia     |     1592 |           33 |         2 |       3.34086 | 0.00209853 |             0.00125628 |                  0.0606061 |         0.101238  |
|         516 | Kevin De Bruyne               | Belgium     |      900 |           52 |         4 |       3.33338 | 0.00370375 |             0.00444444 |                  0.0769231 |         0.0641034 |
|         667 | Kieran Trippier               | England     |     1134 |           42 |         2 |       3.2276  | 0.00284621 |             0.00176367 |                  0.047619  |         0.0768476 |
|         236 | Kylian Mbappé Lottin          | France      |      704 |           30 |         4 |       3.10691 | 0.00441322 |             0.00568182 |                  0.133333  |         0.103564  |
|         361 | Ivan Perišić                  | Croatia     |      673 |           25 |         5 |       3.02414 | 0.00449352 |             0.00742942 |                  0.2       |         0.120966  |
|          66 | Christian Dannemann Eriksen   | Denmark     |      666 |           34 |         1 |       2.571   | 0.00386036 |             0.0015015  |                  0.0294118 |         0.0756176 |
|         208 | Bruno Miguel Borges Fernandes | Portugal    |      666 |           23 |         3 |       2.54351 | 0.00381909 |             0.0045045  |                  0.130435  |         0.110587  |
|         907 | Toni Kroos                    | Germany     |     1259 |           28 |         0 |       2.34932 | 0.00186602 |             0          |                  0         |         0.0839041 |


## Feature weights (logistic regression)

| feature         |     weight |
|:----------------|-----------:|
| end_x           |  1.66086   |
| end_y           | -0.103356  |
| is_cross        |  0.0742512 |
| is_through_ball |  0.141315  |
| is_into_box     |  0.405041  |
| is_progressive  | -0.280524  |

