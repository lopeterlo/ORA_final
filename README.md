# ORA_final - SeqGAN for text summarization

In this article, we addressed the text summarization problem using SeqGAN. By slightly change the training process and policy gradient loss function, we improve the ROUGE score compared to the naive LSTM-based sequence-to-sequence model on samples of the XSum dataset. Follwing sections include the motivation of these project, the methodology of this GAN-based approach, the experiment result and the conclusion.

**Keywords:** Natural Language Processing, Sequence-to-Sequence tasks, Monte Carlo, Reinforcement Learning, Generative Adversarial Network

# Table of Contents
- [Motivation](#motivation)
- [Methodology](#methodology)
## Motivation <a name="motivation"></a>
In natural language processing research field (NLP), there are lots of tasks can be formulated as sequence-to-sequence problem, such as question answering, text summarization and machine translation. These tasks are actually close to daily life. Google use machine/deep learning technique to improve google translation service. Some companies use automatic summarization for their meeting to record meeting process.  

![](https://blog.keras.io/img/seq2seq/seq2seq-teacher-forcing.png) 
[Sequence-to-Sequence problem]
[ref : https://blog.keras.io/a-ten-minute-introduction-to-sequence-to-sequence-learning-in-keras.html)]

## Methodology <a name="methodology"></a>
Recently, GAN-Based approached are proposed to solve this type of problems. Naive GAN architecture is as following figure. Generator will be trained for generating fake samples which are as close as the real samples and letting discriminator misclassify. Discriminator will be trained to distinguish to whether the given samples are real data or not. 

![](https://augustushsu.github.io/uploads/GAN.png)
[Naive Gan architecture]
[ref : https://augustushsu.github.io/uploads/GAN.png]

The model can be furthur utilized to solve sequence-to-sequence task. However, there are two main difficulties :
1. The gradient from the discriminative net makes little sense since the output of the generator is discrete
2. GAN can only give the score for an entire sequence. It is non-trivial to balance how good as it is now(partially) and the future score(entirely)

To address these two main difficulties, SeqGAN (Sequence Generative Adversarial Nets with Policy Gradient) was proposed in 2017 AAAI by Yu. et. al. First, we can consider generative model as an agent of reinforcement learning. There are three different roles:
1. Policy : generative model (stochastic)
2. stateL the generated token so far
3. action: the next token to be generated

Next, for the previous two difficulties, we solve it with policy gradient and Monte Carlo seperately:
1. We train the policy via policy gradient, which naturally avoid the differentiation difficulty for discrete data
2. In the policy gradient, Monte Carlo(MC) search is adopted to approximate the state-action value

![](https://i.imgur.com/1BwlIDP.png)
SeqGAN architecture]
[ref : https://www.aaai.org/Conferences/AAAI/2017/PreliminaryPapers/12-Yu-L-14344.pdf]
