# 诗词对联生成

- 根据上联生成下联，或根据上联和下联第一个字生成完整的对联
- 采用 sample 的生成方式

### 自定义 transformer网络

- encoder + decoder
- PositionEmbedding: 正余弦位置编码
- num_layers: 6
- num_heads: 8
- vocab_size: 7327
- hidden_size: 768
- 特殊token: {'\<PAD>': 0, '\<UNK>': 1, '\<START>': 2, '\<END>': 3}

- 训练数据构建
```
 prompt1 = '请根据给定的上联生成下联：'
 prompt2 = '请根据给定的上下联第一个字生成完整对联：'

 eg: 碧林青旧竹，绿沼翠新苔
 --> encoder input: '请根据给定的上联生成下联：碧林青旧竹'
     decoder input: '<START>绿沼翠新苔'
     decoder label: '绿沼翠新苔<END>'
     
     encoder input: '请根据给定的上下联第一个字生成完整对联：碧绿'
     decoder input: '<START>碧林青旧竹，绿沼翠新苔'
     decoder label: '碧林青旧竹，绿沼翠新苔<END>'
```

- 预测结果：
```
请根据给定的上联生成下联:冰壶见底未为清
下联： 玉树寒花不可攀
--------------------------------------------------
请根据给定的上下联第一个字生成完整对联:天地
上下联： 天寒不可见，地静有神仙

请根据给定的上联生成下联:桃花流水杳然去
下联： 水国流莺不见时
--------------------------------------------------
请根据给定的上下联第一个字生成完整对联:春秋
上下联： 春风吹落叶，秋水入寒山
```

### 迁移 T5-pegasus-small 进行训练

- 预测结果：
```
请根据给定的上联生成下联:冰壶见底未为清
下联： 玉箸如珠不可量
--------------------------------------------------
请根据给定的上下联第一个字生成完整对联:天地
上下联： 天涯有路无穷事，地脉无心是故乡

请根据给定的上联生成下联:桃花流水杳然去
下联： 不似春风一夜吹
--------------------------------------------------
请根据给定的上下联第一个字生成完整对联:春秋
上下联： 春风吹落叶，秋雨洒残花
```

- 参考：
https://github.com/ZhuiyiTechnology/t5-pegasus
https://github.com/renmada/t5-pegasus-pytorch


### 迁移 GPT2 进行训练

- 预测结果：
```
请根据给定的上联生成下联，上联：冰壶见底未为清
下联：草城霞花酒苦山
--------------------------------------------------
请根据给定的上下联第一个字生成完整对联，上下联第一个字：天地
上下联：天子白云玉，地晚自记贫

请根据给定的上联生成下联:桃花流水杳然去
下联： 江花秋湖晓秋风
--------------------------------------------------
请根据给定的上下联第一个字生成完整对联:春秋
上下联： 春来苍苍色，秋月满河山
```