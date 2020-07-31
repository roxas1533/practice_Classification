# practice_Classification
Classification with Softmax Transfer and Cross Entropy Error  
中間層二個、入力1次元3クラス分類  
#### 最終目的
以下のデータクラスを3つに分類する  
<img src="https://user-images.githubusercontent.com/52588447/88926133-7a878400-d2b0-11ea-91ec-d0a8390ba0f9.png" width="320">
## 最尤推定法
いきなり3分類は荷が重いので以下のような1入力2分類を行う。  
<img src="https://user-images.githubusercontent.com/52588447/88926823-7ad44f00-d2b1-11ea-8ceb-3fda5c3f04f5.png" width="320">  
そこで、最尤推定と呼ばれる手法を利用する。  
最尤推定とは  
>最尤推定（さいゆうすいてい、英: maximum likelihood estimationという）や最尤法（さいゆうほう、英: method of maximum likelihood）とは、統計学において、与えられたデータからそれが従う確率分布の母数を点推定する方法である。(Wikipediaより)  

例を挙げて考えてみる。  
~~~
ここに一枚のコインがある。このコインの作りの精度は甘く、表と裏が出る確率は同じではないことがわかっている。
このコインを20回投げたところ16回表が出て4回は裏であった。では、このコインが表になる確率はいくらであるだろうか？
~~~
この問題を最尤推定を用いて解析的に解こう。  
20回投げて16回表が出ているのでこのコインが表が出る確率をwとすると次の式が成り立つ  
<img src="https://latex.codecogs.com/gif.latex?_{20}C_{16}w^{16}(1-w)^{4}">  
ではまずこのコインが実は裏と表が同じ確率<img src="https://latex.codecogs.com/gif.latex?w=\frac{1}{2}">として式に代入してみると  
<img src="https://latex.codecogs.com/gif.latex?_{20}C_{16}\left(\frac{1}{2}\right)^{16}\left(\frac{1}{2}\right&space;)^{4}=0.00462" title="_{20}C_{16}\left(\frac{1}{2}\right)^{16}\left(\frac{1}{2}\right )^{4}=0.00462" />  
では<img src="https://latex.codecogs.com/gif.latex?w=\frac{2}{3}">の時はどうだろうか？  
<img src="https://latex.codecogs.com/gif.latex?_{20}C_{16}\left(\frac{2}{3}\right)^{16}\left(\frac{2}{3}\right&space;)^{4}=0.09106" title="_{20}C_{16}\left(\frac{2}{3}\right)^{16}\left(\frac{2}{3}\right )^{4}=0.09106" />  
2/3の時のほうが確率が高いことからこちらのほうがより真の表が出る確率が高そうといえる、つまりこの式の値(=確率)が最も高くなるようなwを見つければそれが真の表が出る確率といえそうであることがわかる。ではこの式の出力がもっとも高くなる時のwを求めるには？そう、微分をすればよいのである。ちなみにこの式をプロットすると以下のようなグラフになる。(w=0.8の時に最大となっているように思われる。）  
<img src="https://user-images.githubusercontent.com/52588447/88933174-0520b100-d2ba-11ea-8c5e-d9d24496a512.png" width="320" align="left"><img src="https://latex.codecogs.com/gif.latex?\frac{d}{dx}P(w)=\frac{d}{dw}\&space;_{20}C_{16}\left(w\right)^{16}\left(1-w\right&space;)^{4}" title="\frac{d}{dx}P(w)=\frac{d}{dw}\ _{20}C_{16}\left(w\right)^{16}\left(1-w\right )^{4}" />  
計算しやすいように対数をとる(定数も計算に影響しないので取り除く)  
<img src="https://latex.codecogs.com/gif.latex?\begin{align*}\frac{d}{dw}(log(\left(w\right)^{16}\left(1-w\right&space;)^{4}))&=\frac{d}{dw}(16log(w)&plus;4log(1-w))\\&=\frac{16}{w}-\frac{4}{1-w}\\&=\frac{16-20w}{w-w^2}\end{align*}" title="\begin{align*}\frac{d}{dw}(log(\left(w\right)^{16}\left(1-w\right )^{4}))&=\frac{d}{dw}(16log(w)0+4log(1-w))\\&=\frac{16}{w}-\frac{4}{1-w}\\&=\frac{16-20w}{w-w^2}\end{align*}" />  
<img src="https://latex.codecogs.com/gif.latex?w>0\\16-20w=0\\w=\frac{16}{20}\\w=\frac{4}{5}" title="x>0\\16-20x=0\\x=\frac{16}{20}\\w=\frac{4}{5}" />  
以上より<img src="https://latex.codecogs.com/gif.latex?w=\frac{4}{5}=0.8" title="w=\frac{4}{5}=0.8" />であるのでこのコインの表が出る確率は0.8のようである。この時、<img src="https://latex.codecogs.com/gif.latex?w^{16}(1-w)^4" title="w^{16}(1-w)^4" />を尤度と呼び、尤度が最大となるwを探すことが最尤推定である。

それではこの最尤推定を使って二分類問題を解いていく。1と0が混ざっている4~6の値は  
[4.151 4.304 4.342 4.412 4.448 4.546 4.58  4.69  4.691 4.723 4.822 4.914 4.95  5.075 5.309 5.343 5.384 5.395 5.438 5.491 5.548 5.788 5.826 5.84 5.906]
この時の0,1の値は[1 1 1 1 1 1 1 1 0 1 1 1 1 1 0 0 1 1 1 1 1 1 0 0 1]この値がwの確率で出されたものとすると尤度は<img src="https://latex.codecogs.com/gif.latex?w^{20}(1-w)^5" title="w^{20}(1-w)^5" />である。尤度が最大となるwの値は0.8なので4～6で1になる確率は0.8であると推定できる。<img src="https://user-images.githubusercontent.com/52588447/89028606-aca7ed00-d367-11ea-8c52-328515290f53.png" width="320">これで、二分類ができた。


本題の3分類に入る。2分類の時は線形的に解いたが今回のモデルはロジスティック回帰モデルを用いて解くことにする。まず、以下のようなモデルを考えた。
