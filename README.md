# gtrick: Bag of Tricks for Graph Neural Networks

> Trick is all you need

## Tricks
* VirtualNode: [Neural Message Passing for Quantum Chemistry](https://arxiv.org/pdf/1704.01212.pdf)
* Position Encoding
* FLAG
* DropEdge


## Benchmark

### Virtual Node

| Dataset     | GCN             | GCN + Virtual Node | GIN             | GIN + Virtual Node |
|-------------|-----------------|--------------------|-----------------|--------------------|
| ogbg-molhiv | 0.7569 ± 0.0185 | 0.7188 ± 0.0360    | 0.7038 ± 0.0289 | 0.7007 ± 0.0282    |