# 诗词对联生成

- 根据上联生成下联，或根据上联和下联第一个字生成完整的对联

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

