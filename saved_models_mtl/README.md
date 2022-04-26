### 多目标学习

对模型进行改造，多目标学习。
- 训练1

>python trainer_heanet_mtl.py --task 'k'  'g' --transform 'log' 'log' --epochs 800 --batch_size 256 --weight_decay 1e-4 --hidden_channels 128 --n_filters 64 --n_interactions 3 --is_validate True --lr 1e-3

对k和g两个模量数据进行训练。测试命令如下：

>python trainer_heanet_mtl.py --task 'k'  'g' --transform 'log' 'log'  --hidden_channels 128 --n_filters 64 --n_interactions 3 --is_validate True

下面是测试集上的结果。
> 0.06166788563132286(k) ; 0.06550 (g) 


#### 对ef和eg数据进行多目标学习训练
- 训练2
 

>python trainer_heanet_mtl.py --task ef eg --transform scaling scaling --epochs 500 --batch_size 128 --weight_decay 1e-4 --hidden_channels 128 --n_filters 64 --n_interactions 3 --is_validate True --
lr 1e-3

基本上300epochs后不会改变了。
在测试集上进行测试,误差如下：
>  0.09986583888530731(Ef)  0.2919038087129593(Eg) 

- 训练3

我对最上面的tower层结构进行了更改，之前默认都是64 64.现在改成了128 64的结构。训练400 epochs。
> python trainer_heanet_mtl.py --task ef eg --transform scaling scaling --epochs 400 --batch_size 128 --weight_decay 1e-4 --hidden_channels 128 --n_filters 64 --n_interactions 3 --is_validate  --lr 1e-3 --tower_h1 128 --tower_h2 64 -t
> 
测试命令如下
>  python trainer_heanet_mtl.py --task ef eg --transform scaling scaling --hidden_channels 128 --n_filters 64 --n_interactions 3 --is_validate True --tower_h1 128 --tower_h2 64

模型保存在下面的位置中：
mtl_2_mp_ef_eg_128_64_400_best.pt
下面是测试的结果：
>0.08464111387729645 (ef)  0.22702938318252563(Eg)

更新2022.04.03。由于args的使用方式修改了，所以使用方式和以前有一些不同的地方。


#### 使用多目标学习的模型进行单目标学习
更新了arg的传入方式。以后使用下面更新后的方法。

- 训练 k 体模量
>   python trainer_heanet_mtl.py --task k --transform log --epochs 500 --batch_size 256 --weight_decay 1e-4 --hidden_channels 128 --n_filters 64 --n_interactions 3 --is_validate --lr 1e-3 --predict

mae in the test set is 0.0656079649925232


> --task g --transform log --epochs 100 --batch_size 256 --weight_decay 1e-4 --hidden_channels 128 --n_filters 64 --n_interactions 3 --is_validate --lr 1e-3 -p
> 
> --task ef eg --transform scaling scaling --hidden_channels 128 --n_filters 64 --n_interactions 3 --is_validate --tower_h1 128 --tower_h2 64 -t --epochs 500
> 
> 
> python  .\trainer_heanet_mtl.py --task k g --transform log log --hidden_channels 128 --n_filters 64 --n_interactions 3 --is_validate --tower_h1 128 --tower_h2 64 -t --epochs 500

默认使用了16的batch size 很小。之前的都比较大
效果不错！！！
准备放在论文中

the score of task 0 is 0.050499916076660156

the score of task 1 is 0.04647131264209747
mtl_2_mp_k_g_128_64_500_best.pt



完了，现在离奇的事情出现了。单任务训练效果也很好。下面是g的效果。

the score of task 0 is 0.04931531846523285


下面是k的效果。

the score of task 0 is **0.05054137483239174**

重新训练ef的预测。
> python  .\trainer_heanet_mtl.py --task ef  --transform scaling --hidden_channels 128 --n_filters 64 --n_interactions 3 --is_validate --tower_h1 128 --tower_h2 64 -p --epochs 500 --batch_size 256
> 
测试集上表现不好。
mae in the test set is 0.07618208229541779
最后一次训练的模型效果(500)也是不错：the score of task 0 is 0.07714813202619553


训练eg-nz 非零的带隙。
效果如下：

mae in the test set is **0.27105632424354553**


最后对ef和eg重新训练。因为这个时候训练集手动固定了，固定成为了60000.
下面是训练、测试命令
>  python  .\trainer_heanet_mtl.py --task ef eg --transform scaling scaling  --hidden_channels 128 --n_filters 64 --n_interactions 3 --is_validate --tower_h1 128 --tower_h2 64  --epochs 500 --batch_si
ze 256 -p

测试集上的结果：
the score of task 0 is 0.11145998537540436

the score of task 1 is 0.2451021373271942


