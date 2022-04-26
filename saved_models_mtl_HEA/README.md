保存的深度学习模型命令规则介绍：
- mtl: multi-target learning
- 1、2、3、...、6：表示任务数目
- HEA: high entropy alloy
- 300、500：表示训练的时候epochs大小。
- best: 验证集上效果最好的模型
- etot/ef...:特定任务
- 128: 元素特征向量的维度。
- 4+5/2+3: 是2+3组元还是4+5组元的任务
- a0/a1/a2/a3: 数据集的分割类型，参考load_hea_data代码。a使用提前预处理的文件，哪些数据作为训练，测试已经
提前预处理划分到了相应的文件种。 a0: 提前把所有数据根据比例划分为训练和测试的数据集。a1:234组元
数据训练，5组元数据测试。a2:23训练，45测试。  a3: 45组元训练，23组元测试。
- b0/b1/b2/b3： 数据集的分割类型。参考load_hea_data_single_file代码。b在代码中区分分训练、测试验证。
b0: 45多组元的合金中进行，根据比例划分训练和测试。b1: 23低组元合金中进行。b2:2+3+4+5组元合金中进行。


- 测试命令。

>根据实际使用到的训练后的模型，选择相应的模型参数。其中测试的代码可以是下面的样子：

```shell
# 对6个任务同时进行预测。
python trainer_heanet_mtl_HEA.py --task etot emix eform ms mb rmsd  --batch_size 128 --is_validate --split_type 2 -t
```
> 实际上，我们的做法是从相关性出发，找到相关的目标任务放在一起进行预测，这样子效果会好一些。

```shell
# 经过分析，我们把etot，emix，和eform三个能量相关的放在一起进行训练。发现效果还可以。
python trainer_heanet_mtl_HEA.py --task etot emix eform  --batch_size 128  --hidden_channels 128 --n_filters 64 --n_interactions 3 --tower_h1 128 --tower_h2 64 --is_validate -p

# 使用训练好的模型，测试。注意修改模型
python trainer_heanet_mtl_HEA.py --task etot emix eform  --batch_size 128  --is_validate  --split_type 2 -p
```

```shell
# 经过分析，我们把ms 和 mb 两个磁矩相关的性质的放在一起进行训练。发现效果还可以。
python trainer_heanet_mtl_HEA.py --task ms mb  --batch_size 128  --hidden_channels 128 --n_filters 64 --n_interactions 3 --tower_h1 128 --tower_h2 64 --is_validate -p

# 使用训练好的模型，测试。注意修改模型
 python trainer_heanet_mtl_HEA.py --task ms mb  --batch_size 128  --is_validate  --split_type 2 -p

```

```shell
# 经过分析，我们把rmsd性质单独训练。发现效果还可以。
python trainer_heanet_mtl_HEA.py --task ms mb  --batch_size 128  --hidden_channels 128 --n_filters 64 --n_interactions 3 --tower_h1 128 --tower_h2 64 --is_validate -p

# 使用训练好的模型，测试。注意修改模型
 python trainer_heanet_mtl_HEA.py --task rmsd  --batch_size 128  --is_validate  --split_type 2 -p

```
下面是简易的训练方法，都使用默认的参数。
```shell
 python trainer_heanet_mtl_HEA.py --epochs 500 --batch_size 128 --task etot emix eform  --batch_size 128  --is_validate  --split_type 2 -t

```
对能量性质的预测：

| 模型 | 训练集 | etot | emix| eform |  测试集 | etot    |   emix  | eform|
| ------ | ------|------ | ------| ------ |------|------|---|------|
| 2+3+4+5 | 291  |0.112 | 0.972   |      |    |     |
| 2+3+4+5 |  291  |0.091 | 0.025 | 0.012   |     |    0.122  | 0.022 | 0.009|
| 2+3     |  169  |0.052 | 0.025 |  0.012  |       |0.063    | 0.027 | 0.012|
| 4+5     |  121  |0.083 | 0.020 | 0.010   |    | 0.081   | 0.014 | 0.007|

对磁矩性质的预测：

| 模型 | 训练集 | ms | mb |  测试集 | ms    |   mb  |
| ------ | ------|------ | ------ |------|------|---|
| 2+3+4+5 | 291  |0.112 | 0.972   |      |    |     |
| 2+3+4+5 |  291  |0.065 | 0.936 |    |   0.058 |  0.386   |
| 2+3     |  169  |0.043 | 0.421 |    |   0.031 | 0.334    |
| 4+5     |  121  |0.115 | 3.813 |    |   0.121 | 4.307    |

对rmsd的性质进行预测：

| 模型 | 训练集(验证集) | rmsd |  测试集|rmsd测试|
| ------ | ------|------ | ---|----|
| 2+3+4+5 | 291(36)  |0.0105 |  37|  0.0010 |
| 2+3+4+5 |  291(36)  |0.0093 | 37 | 0.0086|
| 2+3     |  169(21)  |0.0146 | 22 | 0.0103 |
| 4+5     |  121(15)  |0.0168 | 16 |   0.0128 |


-训练高熵合金的性质。
使用验证集，并且使用多目标预测，下面进行训练
>python trainer_heanet_mtl_HEA.py --task etot emix eform ms mb rmsd  --epochs 500 --batch_size 128 --weight_decay 1e-4 --hidden_channels 128 --n_filters 64 --n_interactions 3 --is_validate True

此时tower的结构比较简单，是[hidden_layer, 64], [64,64], [64,1]
#### 修复bug
the score of task 0 is 0.2814330756664276

the score of task 1 is 0.04131270945072174

the score of task 2 is 0.017144465819001198

the score of task 3 is 0.157793790102005

the score of task 4 is 8.890557289123535

the score of task 5 is 0.17167237401008606

########## mae in the test set is 1.593318950695296


下面我增大tower塔尖层的复杂度。
此时tower的结构比较简单，是[hidden_layer, 128], [128,64], [64,1]
从训练过程就能看出来，效果竟然显著变好了，下面是训练参数：
>python trainer_heanet_mtl_HEA.py --task etot emix eform ms mb rmsd  --epochs 400 --batch_size 128 --weight_decay 1e-4 --hidden_channels 128 --n_filters 64 --n_interactions 3 --is_validate True --to
wer_h1 128 --tower_h2 64


### split_type=0
同样地，对这个数据集进行训练和测试。这个时候增大了训练的次数，增大为500epochs。

>python trainer_heanet_mtl_HEA.py --task etot emix eform ms mb rmsd --hidden_channels 128 --n_filters 64 --n_interactions 3 --is_validate True --tower_h1 128 --tower_h2 64 --batch_size 128 --epochs
500  --split_type 0

测试集上一般
the score of task 0 is 0.3290392756462097

the score of task 1 is 0.03797396272420883

the score of task 2 is 0.022503674030303955

the score of task 3 is 0.16712243854999542

the score of task 4 is 8.572538375854492

the score of task 5 is 0.16491642594337463

训练集上的结果还算不错。

the score of task 0 is 0.15344351530075073

the score of task 1 is 0.023218417540192604

the score of task 2 is 0.014109401032328606

the score of task 3 is 0.08770613372325897

the score of task 4 is 3.0841562747955322

the score of task 5 is 0.08048425614833832


### split_type=1
为了证明低组元的学习能够应用到高组元的数据中，我们把数据分割为了2/3/4组元作为训练。5作为测试。
5是高组元合金数据，数据量小。此外，第一种的数据测试集的rmsd数据存在问题，有些材料的rmsd存在不一致。
>python trainer_heanet_mtl_HEA.py --task etot emix eform ms mb rmsd  --epochs 400 --batch_size 128 --weight_decay 1e-4 --hidden_channels 128 --n_filters 64 --n_interactions 3 --is_validate True --to
wer_h1 128 --tower_h2 64 --split_type 1

下面是测试的结果。

the score of task 0 is 0.8331852555274963

the score of task 1 is 0.04745005816221237

the score of task 2 is 0.06375592947006226

the score of task 3 is 0.35436347126960754

the score of task 4 is 39.25432205200195

the score of task 5 is 0.2879592180252075

这效果也太差劲了。查看训练集的结果，是否存在该模型不足以拟合的问题？

the score of task 0 is 0.11849990487098694

the score of task 1 is 0.026699356734752655

the score of task 2 is 0.011234216392040253

the score of task 3 is 0.08308744430541992

the score of task 4 is 1.3306300640106201

the score of task 5 is 0.08232572674751282

看起来结果都是正常的。没办法顺利应用在5组元的数据集上。


### split_type=2
我将2+3作为训练集，4+5组元作为测试集。
下面是训练的命令。默认500epochs，默认的batch_size，默认的weight_decay.
> python trainer_heanet_mtl_HEA.py --task etot emix eform ms mb rmsd  --hidden_channels 128 --n_filters 64 --n_interactions 3 --is_validate True --tower_h1 128 --tower_h2 64 --split_type 2

在原始的训练集上的效果惊人的好！！！

the score of task 0 is 0.11048314720392227

the score of task 1 is 0.012880340218544006

the score of task 2 is 0.009200973436236382

the score of task 3 is 0.052410874515771866

the score of task 4 is 0.46211522817611694

the score of task 5 is 0.03959397226572037

>python trainer_heanet_mtl_HEA.py --task etot emix eform ms mb rmsd  --hidden_channels 128 --n_filters 64 --n_interactions 3 --is_validate True --tower_h1 128 --tower_h2 64   --batch_size 128 --spli
t_type 2

测试的结果：（测试结果差的离谱，基本上不可能正确预测出来这些性质。）

the score of task 0 is 3.618922233581543

the score of task 1 is 0.06309327483177185

the score of task 2 is 0.025392010807991028

the score of task 3 is 0.9394475221633911

the score of task 4 is 29.808237075805664

the score of task 5 is 0.21709346771240234

实际上能看出来，机器学习模型认为低组元和高组元的数据之间是存在着本质的差别，因此认为他们不可能迁移过去。

### split_type=3
在这里，我想尝试下，高组元能否迁移到低组元中。我准备使用4+5作为训练，2+3作为测试。看一下效果。

效果依然不好。没有记录结果
## 增加周期性边界条件
- 训练

不知道什么原因导致了低组元和高组元的输入产生如此大的差别。在这里我增加了周期性边界条件尝试一下。
> python trainer_heanet_mtl_HEA.py --task etot emix eform ms mb rmsd  --batch_size 128 --hidden_channels 128 --n_filters 64 --n_interactions 3 --is_validate True --tower_h1 128 --tower_h2 64 --split_
type 3 --use_pbc True

看来就是扯淡。无论有没有周期边界条件，结果都一样。低组元都没办法向高组元。高组元也没办法到低组元。

训练集的结果也一般，不是很好。

the score of task 0 is 0.11844135820865631

the score of task 1 is 0.062251005321741104

the score of task 2 is 0.017737215384840965

the score of task 3 is 0.06712132692337036

the score of task 4 is 4.86176872253418

the score of task 5 is 0.10558665543794632 (看图是比较差的)

测试的结果不忍直视

the score of task 0 is 4.034746170043945

the score of task 1 is 0.11195771396160126

the score of task 2 is 0.12126947194337845

the score of task 3 is 0.5435783863067627

the score of task 4 is 13.496316909790039

the score of task 5 is 0.4580230116844177

#### 直接在4+5组元的数据集上进行训练。效果似乎还可以
>  python trainer_heanet_mtl_HEA.py --task etot emix eform ms mb rmsd  --hidden_channels 128 --n_filters 64 --n_interactions 3 --is_validate True --tower_h1 128 --tower_h2 64 --split_type 0  --epochs
300 --batch_size 128 --processed_data
> 
测试集的结果：

the score of task 0 is 0.10168442130088806

the score of task 1 is 0.013700866140425205

the score of task 2 is 0.008893394842743874

the score of task 3 is 0.09138397872447968

the score of task 4 is 7.015091419219971

the score of task 5 is 0.035496532917022705

训练集的结果：

the score of task 0 is 0.08622842282056808

the score of task 1 is 0.017615659162402153

the score of task 2 is 0.01050182618200779

the score of task 3 is 0.07687162607908249

the score of task 4 is 5.725644588470459

the score of task 5 is 0.08318459987640381 （不太好。）

#### 直接在2+3组元的数据集上进行训练,之前训练的时候是对所有的2+3组员数据进行训练，现在分出一部分作为测试集。
>python trainer_heanet_mtl_HEA.py --task etot emix eform ms mb rmsd  --hidden_channels 128 --n_filters 64 --n_interactions 3 --is_validate True --tower_h1 128 --tower_h2 64   --batch_size 128 --epoc
hs 300 --processed_data --split_type 1

the score of task 0 is 0.14116768538951874

the score of task 1 is 0.02516362816095352

the score of task 2 is 0.009250178001821041

the score of task 3 is 0.0668555423617363

the score of task 4 is 1.0221158266067505

the score of task 5 is 0.07198646664619446 （rmsd不太好）

测试的效果一般。

#### 现在把2+3的训练模型迁移到4+5的训练上，看看效果如何。
进行了两个尝试，第一，把后面的塔尖结构的所有权值参数去掉，只保留到元素特征向量那一个结构。
第二，保留塔尖结构的权值参数，所有的都要重新微调训练。

训练命令都相同，不同的是导入的模型参数。在fine_tune函数中，手动选择是否remove这些参数。
>python trainer_heanet_mtl_HEA.py --task etot emix eform ms mb rmsd  --hidden_channels 128 --n_filters 64 --n_interactions 3 --is_validate True --tower_h1 128 --tower_h2 64   --batch_size 128 --epoc
hs 300 --processed_data --split_type 0

##### 去掉塔尖结构
测试集的结果。

the score of task 0 is 0.11412158608436584

the score of task 1 is 0.018291668966412544

the score of task 2 is 0.02164388634264469

the score of task 3 is 0.07504543662071228

the score of task 4 is 4.592616558074951

the score of task 5 is 0.02527153119444847


##### 保留塔尖结构
这时候，训练了500epochs。测试集的结果。

the score of task 0 is 0.05842915177345276

the score of task 1 is 0.012473389506340027

the score of task 2 is 0.005979549139738083

the score of task 3 is 0.06546909362077713

the score of task 4 is 3.065028667449951

the score of task 5 is 0.022149354219436646
看起来都还不错！！


#### 去掉塔尖结构
之前去掉塔尖结构，但是训练了300epochs，为了保证参数的一致性，我们这次把epochs也设置为了500
对比，然后看看两种迁移的方式对模型的影响有多大。
下面是测试的结果

the score of task 0 is 0.08569896221160889

the score of task 1 is 0.02691951021552086

the score of task 2 is 0.009620314463973045

the score of task 3 is 0.035979967564344406 （ms:好于上一种）

the score of task 4 is 2.6578307151794434 （mb:好于上一种迁移方式）

the score of task 5 is 0.04820121452212334

此外，这个还证实了，500epochs比300epochs能够达到更好的一种效果。

### 直接在4+5上进行训练。
之前也做过一次，效果不好，能够增加对比性，证明迁移之后的效果很好。但是由于训练的步数只有300，
上面的实验也证明了500的步数会更加好。因此，这个时候我把训练步数增大，看看直接训练的效果如何。
>  python trainer_heanet_mtl_HEA.py --task etot emix eform ms mb rmsd  --hidden_channels 128 --n_filters 64 --n_interactions 3 --is_validate True --tower_h1 128 --tower_h2 64   --batch_size 128 --epochs 500 --processed_data --split_type 0

妈的，迁移不迁移没有什么卵用。只要训练的步数足够长，依然效果很好！！

the score of task 0 is 0.0616629421710968 （弱于迁移）
 
the score of task 1 is 0.013013633899390697 （稍弱于迁移）

the score of task 2 is 0.011709757149219513 （很弱于迁移）

the score of task 3 is 0.057435084134340286 （稍强于迁移）
 
the score of task 4 is 2.495039463043213 （稍强于迁移）

the score of task 5 is 0.02593887411057949 （稍弱于迁移）



#### 尝试迁移Ef的参数。Ef先在materials project的数据集上进行训练。因此，需要训练一个单目标学习的模型
这个模型只用于预测高熵合金的Ef。
这个训练和测试的集合是随机从所有的数据分割出来的。所以训练的数据包括了2345组元的合金体系。
当然测试集也一样，这几种组元都有。即尽可能的对这个体系的所有数据都能够很好的模拟。

> python trainer_heanet_mtl_HEA.py --task eform --hidden_channels 128 --n_filters 64 --n_interactions 3 --is_validate True --tower_h1 128 --tower_h2 64 --batch_size 128 --epochs
500  --split_type 0

测试集上没那么好
the score of task 0 is 0.02398679405450821

训练集上还可以
 0.01502698939293623


但是之前在materials project的数据集上，元素特征向量的维度好像是256维度，这个时候还用了128.
因此重新训练一个。并且之前使用了scaling。
#### 其次就是，对2+3+4+5高熵合金都进行训练。也就是说使得该模型适用于这个体系的预测Cr-Fe-Co-Ni-Mn和Cr-Fe-Co-Ni-Pd体系的预测

> python trainer_heanet_mtl_HEA.py --task etot emix eform ms mb rmsd --hidden_channels 128 --n_filters 64 --n_interactions 3 --is_validate True --tower_h1 128 --tower_h2 64 --batch_size 128 --epochs
500  --split_type 0




#### 做一个单一的模型，只预测了Etot能量。
> python trainer_heanet_mtl_HEA.py --task etot --hidden_channels 128 --n_filters 64 --n_interactions 3 --is_validate True --tower_h1 128 --tower_h2 64 --batch_size 128 --epochs 500  --split_type 0


测试集上的结果：

mae in the test set is 0.29413479566574097 （太差了）


训练集上也太tm准确了。
mae in the test set is 0.053597282618284225


