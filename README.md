# KDD-Cup-2019  
 This repository is my solution of KDD Cup 2019 Regular ML track
(Context-Aware Multi-Modal Transportation Recommendation).
See [Competition Website](https://dianshi.baidu.com/competition/29/rule)
for the details.
In this competiton, I got the 57th place at phase1 and 52nd at phase2.
## Phase1
### Result
57th place of 1702 teams.
- LB score: 0.69917984
- Local cv score: 0.678330

![lb_phase1](img/lb_phase1.png)

### Model Pipeline
![phase1_model_pipeline](img/phase1_model_pipeline.png)
See [phase1 final version](https://github.com/MitsuruFujiwara/KDD-Cup-2019/tree/7f538fd0785118cd6e8fd120023152872357023e) for model details.

### Key Findings
- Sub Models  
I prepared two sub models, one with queries and the other with queries & profiles.
The LB score improved from 0.6925 to 0.6945 by adding their outputs to main model's features.
- Post Processing  
I adjusted the number of predicted classes by constant multiples.

## Phase2
### Result
52nd place of 100teams.
- LB score: 0.69362814
- Local cv score: 0.657519

![lb_phase2](img/lb_phase2.png)

### Model Pipeline
![phase2_model_pipeline](img/phase2_model_pipeline.png)

### Key Findings
- Split Models by Cities  
I splitted main models by cities.
There were 3 cities in phase2 and the distribution of transport mode were diffirent.
After splitting models, LB score reached to 0.6900.
- Post Processing  
Applyed the same post processing as phase1.
Finally this scored 0.6936 on LB.
