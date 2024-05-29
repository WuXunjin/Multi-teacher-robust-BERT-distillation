# Multi-teacher-robust-BERT-distillation
通过多教师知识蒸馏学习一个小型的学生网络(distilbert)，该网络既具有作为对抗预训练教师网络(distilbert)的鲁棒性能，又具有作为正常预训练教师网络(bert-base)的正常性能。采用Mean Teacher算法更新对抗教师模型，PGD对抗训练算法更新对抗样本，使用TextFooler、TextBugger、BAE三种文本对抗攻击方法测试学生模型鲁棒性
