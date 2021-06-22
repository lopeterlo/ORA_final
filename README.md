# ORA_final - SeqGAN for text summarization

In this article, we addressed the text summarization problem using SeqGAN. By slightly change the training process and policy gradient loss function, we improve the ROUGE score compared to the naive LSTM-based sequence-to-sequence model on samples of the XSum dataset. Follwing sections include the motivation of these project, the methodology of this GAN-based approach, the experiment result and the conclusion.

**Keywords:** Natural Language Processing, Sequence-to-Sequence tasks, Monte Carlo, Reinforcement Learning, Generative Adversarial Network

# Table of Contents
- [Motivation](#motivation)
- [Methodology](#methodology)
- [Experiment](#experiment)
- [Concousion](#oncousion)
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
2. state: the generated token so far
3. action: the next token to be generated

For the previous two difficulties, we solve it with policy gradient and Monte Carlo seperately:
1. We train the policy via policy gradient, which naturally avoid the differentiation difficulty for discrete data
2. In the policy gradient, Monte Carlo(MC) search is adopted to approximate the state-action value

![](https://i.imgur.com/1BwlIDP.png)
SeqGAN architecture]
[ref : https://www.aaai.org/Conferences/AAAI/2017/PreliminaryPapers/12-Yu-L-14344.pdf]

Next the Reward and Policy gradient can be formulated as follow:

![image](https://user-images.githubusercontent.com/22761888/122669722-3ca98c00-d1f1-11eb-9bfe-4fca928f8174.png)

[Reward: To reduce the variance and get more accurate assessment of the action value, we run the roll-out policy starting
from current state till the end of the sequence for N times to
get a batch of output samples]

![image](https://user-images.githubusercontent.com/22761888/122669745-5a76f100-d1f1-11eb-9384-3a39e5164e5b.png)

[ an unbiased estimation for the gradient of the objective function]


## Experiment <a name="experiment"></a>

We sample data from XSum dataset, which are collected from BBC articles (2010 to 2017) and cover a wide variety of domains (e.g., News, Politics, Sports, Weather, Business, Technology). We sampled about 70000 training samples and 20000 testing data as our dataset.

### Evaluation:

In evaluation step, we use ROUGE-1, ROUGE-2 and ROUGE-L score to calaulate the model performance. The ROUGE is a score of overlapping words. ROUGE-N refers to overlapping n-grams.
Specially: 

![image](https://user-images.githubusercontent.com/22761888/122670033-7f1f9880-d1f2-11eb-8e5b-bdc2e2f209b7.png)

For example :
- S1. police killed the gunman (reference)
- S2. police kill the gunman (candidate 1)
- S3. the gunman kill police (candidate 2)

Explanation:
  1. S2 and S3 both have one overlapping bigram with the reference, so they have the same ROUGE-2 score.
  2. For ROUGE-L : we look for longest common subsequence. Therefore, S2 ROUGE-L:  3/4 , S1 ROUGE-L: 2/4, which means s2 is better than s1 in ROUGE-L.

### Training process:

The training process is as following picture. The generator is a normal LSTM-based Seq2Seq model that will output predicted summary given input full text. We will first pre-train the generator to make it capable to generate acceptable summarization. Next, we will iteratively train the discriminator and use policy gradient to train generator.

![image](https://user-images.githubusercontent.com/22761888/122670362-4e406300-d1f4-11eb-8820-d637d6480a8d.png)

### Initially result:

we follow the training process but got unsatisfactory result, therefore we rethink the procedure and make little change on training procedure and also the loss function of policy gradient:

![image](https://user-images.githubusercontent.com/22761888/122670483-ef2f1e00-d1f4-11eb-8d0a-9abed2553bec.png)
![image](https://user-images.githubusercontent.com/22761888/122670490-f6eec280-d1f4-11eb-8df3-1fc4ee640ac3.png)

Also, we revised PG loss function that we gave weight to different position in generated summary (gave more weight to the previous token).

![image](https://user-images.githubusercontent.com/22761888/122670512-0e2db000-d1f5-11eb-9372-5942f6c5c2af.png)


### Result:
![image](https://user-images.githubusercontent.com/22761888/122670538-2c93ab80-d1f5-11eb-89d1-8dc12a14e16e.png)
![image](https://user-images.githubusercontent.com/22761888/122670563-5056f180-d1f5-11eb-9ab1-64dcfca321c7.png)

## Conclusion <a name="conclusion"></a>

In this project, We address text summarization problem using GAN-based approach. We modified the training process and PG loss function to slightly improve performance. The perforance is not good as state of the art but PG loss help improve performance in our experiment.

![image](https://user-images.githubusercontent.com/22761888/122670617-898f6180-d1f5-11eb-94dd-74dd2b07c113.png)




