之前训了一大堆模型。由于更改了参数，现在需要重新训练。

- 对4+5的数据进行训练。

结果很糟糕啊。。。。。。。。

the score of task 0 is 0.09280455112457275

the score of task 1 is 0.06958369165658951

the score of task 2 is 0.0501910001039505

the score of task 3 is 0.11050532013177872

the score of task 4 is 5.631658554077148

the score of task 5 is 0.04925113916397095


- 结果太差。重新训练。加上周期性边界条件。。

发现在ms和mb上的效果比较好，但是在关于能量的三个量上效果不好。分别是 etot emix eform。

-对这三个量的训练结果

the score of task 0 is 0.08319880068302155

the score of task 1 is 0.019761087372899055

the score of task 2 is 0.009530344046652317


- 对ms和mb两个量进行训练。
ms和mb的效果不怎么样；；
the score of task 0 is 0.11524567753076553

the score of task 1 is 3.8138861656188965


-对rmsd单独进行训练。
效果竟然非常好。。。
mae in the test set is 0.01676536165177822



-对2+3的数据进行训练，效果也十分不错

 etot emix eform

the score of task 0 is 0.051602110266685486

the score of task 1 is 0.02535172738134861

the score of task 2 is 0.011544919572770596

- 对2+3的数据进行训练。

ms和mb

the score of task 0 is 0.04279293119907379

the score of task 1 is 0.42110970616340637


- 对2+3的数据进行训练。
rmsd
结果相当不错了已经。

- mae in the test set is 0.014557369984686375


- 对2+3数据中所有的性质进行训练

the score of task 0 is 0.09138226509094238

the score of task 1 is 0.06483005732297897

the score of task 2 is 0.02785431034862995

the score of task 3 is 0.0328696109354496

the score of task 4 is 0.5237263441085815

the score of task 5 is 0.07911623269319534
效果还不错