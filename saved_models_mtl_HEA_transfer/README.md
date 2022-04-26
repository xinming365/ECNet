对etot，emix，eform三个任务进行训练。我们认为他们三个性质都是能量，应该比较相近，所以同时预测：
>python trainer_heanet_mtl_HEA.py --task etot emix eform  --epochs 400 --batch_size 128 --weight_decay 1e-4 --hidden_channels 128 --n_filters 64 --n_interactions 3 --is_validate True --tower_h1 128
--tower_h2 64

测试：
目前来说，效果都还可以，比对6个性质一起预测的时候要高一些。
the score of task 0 is 0.25654199719429016
 
the score of task 1 is 0.040695395320653915

the score of task 2 is 0.015935348346829414

mae in the test set is 0.10439091362059116

对ms，mb，和rmsd的数据进行训练。
>python trainer_heanet_mtl_HEA.py --task ms mb rmsd  --epochs 400 --batch_size 128 --weight_decay 1e-4 --hidden_channels 128 --n_filters 64 --n_interactions 3 --is_validate True --tower_h1 128 --tow
er_h2 64


对相应的数据进行测试：

the score of task 0 is 0.1651424616575241

the score of task 1 is 8.923702239990234

the score of task 2 is 0.15600533783435822

对ms，mb，和rmsd的数据进行训练。此时增加了两个bn层，在最后的tower_layer中。下面是训练和测试。
>python trainer_heanet_mtl_HEA.py --task ms mb rmsd  --epochs 400 --batch_size 128 --weight_decay 1e-4 --hidden_channels 128 --n_filters 64 --n_interactions 3 --is_validate True --tower_h1 128 --tow
er_h2 64

测试集上的效果
略。效果不如6个一起训练的时候好。

- 训练2

这时候，修改了bn层的参数affine=False，下面发现在rmsd上面预测的结果比较好。

the score of task 0 is 0.1604684591293335

the score of task 1 is 37.739383697509766

the score of task 2 is 0.0927605852484703

- 训练3

观察到上面的训练对于mb的效果比较差，这个时候，我们之训练ms和rmsd两个结果。下面是测试集上的效果：
> python trainer_heanet_mtl_HEA.py --task ms rmsd  --epochs 400 --batch_size 128 --weight_decay 1e-4 --hidden_channels 128 --n_filters 64 --n_interactions 3 --is_validate True --tower_h1 128 --tower_
h2 64

训练集上面效果竟然这么好：（白激动了，是训练集上的效果）
>the score of task 0 is 0.08794175833463669

>the score of task 1 is 0.05381542071700096

查看一下测试集上的效果。

the score of task 0 is 0.1793096661567688

the score of task 1 is 0.16547197103500366

可以确定，数据存在问题。。。。。尤其是测试集上的rmsd和训练集上的rmsd有很大的不一致。

- 训练4

之前的训练，在tower层上面加上了bn层。之前很多的模型都是没有bn层的，现在去掉bn层，观察一下bn是否有必要
> python trainer_heanet_mtl_HEA.py --task ms rmsd  --epochs 400 --batch_size 128 --weight_decay 1e-4 --hidden_channels 128 --n_filters 64 --n_interactions 3 --is_validate True --tower_h1 128 --tower_
h2 64

竟然还好了一些，说明bn层对模型影响不大。

the score of task 0 is 0.15026657283306122

the score of task 1 is 0.14861947298049927

