## Motivation
OPD for diffusion → Regression on velocity/KL is the most efficient way. Policy-gradient term has no contribution.
Optimal Solution: the optimal solution is the convex combination of all teachers.
OPD from multi-teacher naively has two results: 1. Like DiffusionOPD, the in-domain performance is comparable with each in-domain teacher, but the out-of-domain performance may have no gain; 2. Use the average velocities of all teachers as the target. It results lower performance on in-domain data but better performance on OOD data.
To achieve both teacher-comparable in-domain performance while better OOD performance, we learn a weight matrix with teacher info, data domain info and temporal info.

## Method and Experiments
第一步，推导OPD的公式，分为Pathwise和Policy-gradient两项.结果：policy-gradient没有贡献。 这一部分DiffusionOPD已经做了推导，简单叙述即可。 三个实验：pathwise_only, policy-gradient_only, both.
第二步，依照DiffusionOPD的算法，每个data domain上只拟合对应的teacher, 结果是无法学到其它适用的reward例如pickscore，in-domain也被teacher bound； 如果直接取所有teacher的凸组合进行拟合，in-domain指标更低（被bound住），而OOD指标会更好一点点。两个实验，一个对齐DiffusionOPD, 一个是average_pathwise. → student能几乎完美地拟合teacher, 达到comparable的performance, 问题在于如何构建一个更好的teacher.
第三步，引入teacher, data-domain, temporal (最关键）的权重。多个FM的凸组合仍然是一个FM, 所以直接可以用现有RL算法（DiffusionNFT/Flow-GRPO)进行优化来调整参数，保证In-domain能力的同时提高OOD rewards bonus.
准备： 用Flow-DPPO训练三个Teacher (不带CFG），GenEval, OCR, Pickscore. ✅
主实验：DiffusionNFT和Flow-GRPO分别进行，选SOTA. (同时也是RL算法的消融） NFT收敛更快，更适合小参数量搜索✅
对最终权重的时序分析→ 可能的 结论类似：GenEval主宰前期，PickScore主宰后期。
消融：不同reward setting（如何设计bonus）（2~3个实验）, 不同的初始权重 （2~3个实验）.


- Flow-DPPO train teachers: GenEval, OCR, Pickscore

/Users/bowenping/Desktop/geneval-teacher-eval.png

eval/GenEval

Selected ckpt *checkpoint-600*: step 1200. Performance: 0.958

/Users/bowenping/Desktop/geneval-teacher-train.png

train/GenEval

/Users/bowenping/Desktop/ocr-teacher-eval.png

eval/OCR

Selected ckpt *checkpoint-220*: step 440. Performance: 0.953

/Users/bowenping/Desktop/ocr-teacher-train.png

train/OCR

/Users/bowenping/Desktop/pickscore-teacher-eval.png

eval/Pickscore. 0.9313 * 26 ~ 24.21

Selected ckpt *checkpoint-1500*: step 3000. Performance: 24.128

/Users/bowenping/Desktop/pickscore-teacher-train.png

train/Pickscore