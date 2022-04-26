在迁移学习的情况中，所有都是45组元中进行的。b0.

### 自行修改函数中的引入迁移学习的方式

python trainer_heanet_mtl_HEA.py --task etot emix eform --batch_size 128 --is_validate --split_type 0 -f


#### 测试ms 和mb；修改函数里要迁移的模型
 python trainer_heanet_mtl_HEA.py --task ms mb --batch_size 128 --is_validate --split_type 0 -f
 
#### 测试rmsd; 修改后：
python trainer_heanet_mtl_HEA.py --task rmsd --batch_size 128 --is_validate --split_type 0 -f


#### as unit的方式迁移学习
python trainer_heanet_mtl_HEA.py --task etot emix eform ms mb rmsd --batch_size 128 --is_validate --split_type 0 -f