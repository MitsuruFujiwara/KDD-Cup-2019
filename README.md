# KDD-Cup-2019  
 This repository is my solution of KDD Cup 2019 Regular ML track
(Context-Aware Multi-Modal Transportation Recommendation).
See [Competition Website](https://dianshi.baidu.com/competition/29/rule)
for details.
In this competiton, I ranked 57th place at phase1 and 52nd at phase2.
## Phase1
See [phase1 final version](https://github.com/MitsuruFujiwara/KDD-Cup-2019/tree/7f538fd0785118cd6e8fd120023152872357023e) for full model source.
### Result
57th place of 1702 teams.
- LB score: 0.69917984
- Local cv score: 0.678330

![lb_phase1](img/lb_phase1.png)

### Model Pipeline
![phase1_model_pipeline](img/phase1_model_pipeline.png)

### Key Findings
#### Sub Model
I used 2 sub model trained by queries and queries & profiles.
By adding these predictions to features,
#### Post Processing

## Phase2
### Result
52nd place of 100teams.
- LB score: 0.69362814
- Local cv score: 0.657519

![lb_phase2](img/lb_phase2.png)

### Model Pipeline
![phase2_model_pipeline](img/phase2_model_pipeline.png)
### Features

### Post Processing

### Score
