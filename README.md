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
<img src="https://user-images.githubusercontent.com/52588447/89030061-ab2bf400-d36a-11ea-8ad1-182ce1501401.png" width="320" align="left">入力一次元(ダミーニューロンで二次元)中間層三次元出力層三次元である。このモデルの式は以下の通り  
<img src="https://latex.codecogs.com/gif.latex?x=[[x_1,1],[x_2,1],[x_3,1],[x_4,1],...,[x_N,1]]" title="x=[[x_1,1],[x_2,1],[x_3,1],[x_4,1],...,[x_N,1]]" />  
<img src="https://latex.codecogs.com/gif.latex?\\Iy_1=W_{x11}x&plus;W_{x12}\\Iy_2=W_{x21}x&plus;W_{x22}\\Iy_3=W_{x31}x&plus;W_{x32}" title="\\Iy_1=W_{x11}x+W_{x12}\\Iy_2=W_{x21}x+W_{x22}\\Iy_3=W_{x31}x+W_{x32}" />、<img src="https://latex.codecogs.com/gif.latex?\\y_1=sig(Iy_1)\\y_2=sig(Iy_2)\\y_3=sig(Iy_3)" title="\\y_1=sig(Iy_1)\\y_2=sig(Iy_2)\\y_3=sig(Iy_3)" />  
<img src="https://latex.codecogs.com/gif.latex?\\Iz_1=W_{y11}y_1&plus;W_{y12}y_2&plus;W_{y13}y_3&plus;W_{y14}\\Iz_2=W_{y21}y_1&plus;W_{y22}y_2&plus;W_{y23}y_3&plus;W_{y24}\\Iz_3=W_{y31}y_1&plus;W_{y32}y_2&plus;W_{y33}y_3&plus;W_{y34}" title="\\Iz_1=W_{y11}y_1+W_{y12}y_2+W_{y13}y_3+W_{y14}\\Iz_2=W_{y21}y_1+W_{y22}y_2+W_{y23}y_3+W_{y24}\\Iz_3=W_{y31}y_1+W_{y32}y_2+W_{y33}y_3+W_{y34}" />、<img src="https://latex.codecogs.com/gif.latex?\\z_1=sig(Iz_1)\\z_2=sig(Iz_2)\\z_3=sig(Iz_3)" title="\\z_1=sig(Iz_1)\\z_2=sig(Iz_2)\\z_3=sig(Iz_3)" />  
<img src="https://latex.codecogs.com/gif.latex?\\Iq_1=W_{z11}z_1&plus;W_{z12}z_2&plus;W_{z13}z_3&plus;W_{z14}\\Iq_2=W_{z21}z_1&plus;W_{z22}z_2&plus;W_{z23}z_3&plus;W_{z24}\\Iq_3=W_{z31}z_1&plus;W_{z32}z_2&plus;W_{z33}z_3&plus;W_{z34}" title="\\Iq_1=W_{z11}z_1+W_{z12}z_2+W_{z13}z_3+W_{z14}\\Iq_2=W_{z21}z_1+W_{z22}z_2+W_{z23}z_3+W_{z24}\\Iq_3=W_{z31}z_1+W_{z32}z_2+W_{z33}z_3+W_{z34}" />  
<img src="https://latex.codecogs.com/gif.latex?\\Q_1=softmax(Iq_1)\\Q_2=softmax(Iq_2)\\Q_3=softmax(Iq_3)" title="\\Q_1=softmax(Iq_1)\\Q_2=softmax(Iq_2)\\Q_3=softmax(Iq_3)" />  
sig()はシグモイド関数、softmax()はソフトマックス関数を表す。この出力Qは[0.1,0.8,0.1]のように確率で表される。次に、尤度を求める。入力xに対してそれぞれの確率をQ1,Q2,Q3としてa,b,cを発生した個数(a+b+c=N)とすると<img src="https://latex.codecogs.com/gif.latex?(Q_1)^a(Q_2)^b(Q_3)^c" title="(Q_1)^a(Q_2)^b(Q_3)^c" />が尤度となる。しかし、abcと3変数用意するのはあまり好ましくないのでTという一つの変数を用意する。この中には例えば入力xに対して答えが1であったとしたらT=[0,1,0]、答えが2であったとしたらT=[0,0,1]といった正解の値のみ1が入っている。これを利用することで尤度は次のように書き換えることができる。  
<img src="https://latex.codecogs.com/gif.latex?P'(Q)=\prod_{k=1}^{N}Q_1^{T_{k1}}Q_2^{T_{k2}}Q_3^{T_{k3}}" title="P'(Q)=\prod_{k=1}^{N}Q_1^{T_{k1}}Q_2^{T_{k2}}Q_3^{T_{k3}}" />
このような正解のラベルの時に1として不正解のデータを0とする表現をone-hot表現と呼ぶ。この尤度が最大となるようなQを求めればよいがこのままでは多数の掛け算で値が非常に小さくなる恐れがある。そこで対数をとることで足し算に変更する。  
<img src="https://latex.codecogs.com/gif.latex?logP'(Q)=P(Q)=\sum_{k=1}^{N}T_{k1}logQ_1&plus;T_{k2}logQ_2&plus;T_{k3}logQ_3" title="logP'(Q)=P(Q)=\sum_{k=1}^{N}T_{k1}logQ_1+T_{k2}logQ_2+T_{k3}logQ_3" />  
これがこのモデルに対する損失関数の役割を果たす。しかし、一般的な損失関数は値が最小となるように重みを更新していくのでこの式に-1を掛けることで以下のようになりこの式を特に交差エントロピー誤差と呼ぶ。  
<img src="https://latex.codecogs.com/gif.latex?-\sum_{k=1}^{N}T_{k1}logQ_1&plus;T_{k2}logQ_2&plus;T_{k3}logQ_3" title="-\sum_{k=1}^{N}T_{k1}logQ_1+T_{k2}logQ_2+T_{k3}logQ_3" />  
ではこれらの式から重みの偏微分を求めていこう。まず、<img src="https://latex.codecogs.com/gif.latex?\frac{\partial&space;P}{\partial&space;W_{z11}}" title="\frac{\partial P}{\partial W_{z11}}" />。連鎖律の公式より<img src="https://latex.codecogs.com/gif.latex?\frac{\partial&space;P}{\partial&space;W_{z11}}=\frac{\partial&space;P}{\partial&space;Q}\times\frac{\partial&space;Q}{\partial&space;Iq_1}\times\frac{\partial&space;Iq_1}{\partial&space;W_{z11}}" title="\frac{\partial P}{\partial W_{z11}}=\frac{\partial P}{\partial Q}\times\frac{\partial Q}{\partial Iq_1}\times\frac{\partial Iq_1}{\partial W_{z11}}" />。<img src="https://latex.codecogs.com/gif.latex?\frac{\partial&space;P}{\partial&space;Q_i}=-\frac{T_{i}}{Q_i}" title="\frac{\partial P}{\partial Q_i}=-\frac{T_{i}}{Q_i}" />  
ソフトマックス関数の微分は以下のようになるため  
<img src="http://www.texrendr.com/cgi-bin/mimetex?\normalsize%20%5Cbegin%7Balign*%7D%5Cfrac%7B%5Cpartial%20y_i%7D%7B%5Cpartial%20x_i%7D%3D%5Cleft%5C%7B%20%5Cbegin%7Barray%7D%7Bl%7Dy_i(1-y_i)%26i%3Dk%5C%5C-y_iy_k%26i%5Cneq%20k%5Cend%7Barray%7D%5Cright.%5Cend%7Balign*%7D" />  
連鎖律の公式より  
<img src="https://latex.codecogs.com/gif.latex?\begin{align*}\frac{\partial&space;P}{\partial&space;Q}\times\frac{\partial&space;Q}{\partial&space;Iq_1}&=-\frac{T_1}{Q_1}\times&space;-Q_1(1-Q_1)-\frac{T_2}{Q_2}\times(-Q_1Q_2)-\frac{T_3}{Q_3}\times(-Q_1Q_3)\\&=-T_1(1-Q_1)&plus;T_2Q_1&plus;T_3Q_1\\&=Q_1(T_1&plus;T_2&plus;T_3)-T_1\end{align*}" title="\begin{align*}\frac{\partial P}{\partial Q}\times\frac{\partial Q}{\partial Iq_1}&=-\frac{T_1}{Q_1}\times -Q_1(1-Q_1)-\frac{T_2}{Q_2}\times(-Q_1Q_2)-\frac{T_3}{Q_3}\times(-Q_1Q_3)\\&=-T_1(1-Q_1)+T_2Q_1+T_3Q_1\\&=Q_1(T_1+T_2+T_3)-T_1\end{align*}" />  
ここで、Tはどれかが必ず1でその他は0であるためT1+T2+T3=1となるので<img src="https://latex.codecogs.com/gif.latex?\begin{align*}\frac{\partial&space;P}{\partial&space;Q}\times\frac{\partial&space;Q}{\partial&space;Iq_1}&=Q_1-T_1\end{align*}" title="\begin{align*}\frac{\partial P}{\partial Q}\times\frac{\partial Q}{\partial Iq_1}&=Q_1-T_1\end{align*}" />  
<img src="https://latex.codecogs.com/gif.latex?\frac{\partial&space;P}{\partial&space;W_{z11}}=\frac{\partial&space;P}{\partial&space;Q}\times\frac{\partial&space;Q}{\partial&space;Iq_1}\times\frac{\partial&space;Iq_1}{\partial&space;W_{z11}}=(Q_1-T_1)\times&space;z" title="\frac{\partial P}{\partial W_{z11}}=\frac{\partial P}{\partial Q}\times\frac{\partial Q}{\partial Iq_1}\times\frac{\partial Iq_1}{\partial W_{z11}}=(Q_1-T_1)\times z" />  
これをWxすべて行い、行列で表すと  
<img src="https://latex.codecogs.com/gif.latex?\frac{\partial&space;P}{\partial&space;W_{z}}=&space;\left[\begin{array}{ccc}Q_1-T_1\\Q_2-T_2\\Q_3-T_3\end{array}\right]\left[\begin{array}{cccc}z_1&z_2&z_3&1\end{array}\right]" title="\frac{\partial P}{\partial W_{z}}= \left[\begin{array}{ccc}Q_1-T_1\\Q_2-T_2\\Q_3-T_3\end{array}\right]\left[\begin{array}{cccc}z_1&z_2&z_3&1\end{array}\right]" />
次にWyを求めると  
<img src="https://latex.codecogs.com/gif.latex?\begin{align*}\frac{\partial&space;P}{\partial&space;W_{y11}}&=\frac{\partial&space;P}{\partial&space;Q}\times&space;\frac{\partial&space;Q}{\partial&space;Iq}\times\frac{\partial&space;Iq}{\partial&space;z_1}\times\frac{\partial&space;z_1}{\partial&space;W_{y11}}\\&=((z_1-T_1)W_{z11}&plus;(z_2-T_2)W_{z21}&plus;(z_3-T_3)W_{z31})(z_1(1-z_1))y_1\end{align*}" title="\begin{align*}\frac{\partial P}{\partial W_{y11}}&=\frac{\partial P}{\partial Q}\times \frac{\partial Q}{\partial Iq}\times\frac{\partial Iq}{\partial z_1}\times\frac{\partial z_1}{\partial W_{y11}}\\&=((z_1-T_1)W_{z11}+(z_2-T_2)W_{z21}+(z_3-T_3)W_{z31})(z_1(1-z_1))y_1\end{align*}" />  
同様に最後までもとめ行列で表すと  
<img src="http://www.texrendr.com/cgi-bin/mimetex?\normalsize%20%5Cfrac%7B%5Cpartial%20P%7D%7B%5Cpartial%20W_y%7D%3D%5Cleft%5B%5Cbegin%7Barray%7D%7Bccc%7DW_%7Bz11%7D%26W_%7Bz12%7D%26W_%7Bz13%7D%5C%5CW_%7Bz21%7D%26W_%7Bz22%7D%26W_%7Bz23%7D%5C%5CW_%7Bz31%7D%26W_%7Bz32%7D%26W_%7Bz33%7D%5Cend%7Barray%7D%5Cright%5D%5ET%5Cleft%5B%5Cbegin%7Barray%7D%7Bc%7DQ_1-T_1%5C%5CQ_2-T_2%5C%5CQ_3-T_3%5Cend%7Barray%7D%5Cright%5D%5Ccirc%5Cleft%5B%5Cbegin%7Barray%7D%7Bc%7Dz_1(1-z_1)%5C%5Cz_2(1-z_2)%5C%5Cz_3(1-z_3)%5Cend%7Barray%7D%5Cright%5D%5Cleft%5B%5Cbegin%7Barray%7D%7Bcccc%7Dy_1%26y_2%26y_3%261%5Cend%7Barray%7D%5Cright%5D" />  
最後にWxをもとめる。  
<img src="https://latex.codecogs.com/gif.latex?\begin{align*}\frac{\partial&space;P}{\partial&space;W_{x11}}&=\frac{\partial&space;P}{\partial&space;Q}\times&space;\frac{\partial&space;Q}{\partial&space;Iq}\times\frac{\partial&space;Iq}{\partial&space;z}\times\frac{\partial&space;z}{\partial&space;y_1}\times\frac{\partial&space;y_1}{\partial&space;Iy_1}\times\frac{\partial&space;Iy_1}{\partial&space;W_{x11}}\\&=(((z_1-T_1)W_{z11}&plus;(z_2-T_2)W_{z21}&plus;(z_3-T_3)W_{z31})(z_1(1-z_1))W_{y11}\\&&plus;((z_1-T_1)W_{z12}&plus;(z_2-T_2)W_{z22}&plus;(z_3-T_3)W_{z32})(z_2(1-z_2))W_{y21}\\&&plus;((z_1-T_1)W_{z13}&plus;(z_2-T_2)W_{z23}&plus;(z_3-T_3)W_{z33})(z_3(3-z_3))W_{y31})\\&(y_1(1-y_1))x\end{align*}" title="\begin{align*}\frac{\partial P}{\partial W_{x11}}&=\frac{\partial P}{\partial Q}\times \frac{\partial Q}{\partial Iq}\times\frac{\partial Iq}{\partial z}\times\frac{\partial z}{\partial y_1}\times\frac{\partial y_1}{\partial Iy_1}\times\frac{\partial Iy_1}{\partial W_{x11}}\\&=(((z_1-T_1)W_{z11}+(z_2-T_2)W_{z21}+(z_3-T_3)W_{z31})(z_1(1-z_1))W_{y11}\\&+((z_1-T_1)W_{z12}+(z_2-T_2)W_{z22}+(z_3-T_3)W_{z32})(z_2(1-z_2))W_{y21}\\&+((z_1-T_1)W_{z13}+(z_2-T_2)W_{z23}+(z_3-T_3)W_{z33})(z_3(3-z_3))W_{y31})\\&(y_1(1-y_1))x\end{align*}" />  
行列は  
<img src="https://latex.codecogs.com/gif.latex?\frac{\partial&space;P}{\partial&space;W_x}=\left(\left[\begin{array}{ccc}W_{y11}&W_{y12}&W_{y13}\\W_{y21}&W_{y22}&W_{y23}\\W_{y31}&W_{y32}&W_{y33}\end{array}\right]\circ\left[\begin{array}{c}z_1(1-z_1)\\z_2(1-z_2)\\z_3(1-z_3)\end{array}\right]\right)^T\left(\left[\begin{array}{ccc}W_{z11}&W_{z12}&W_{z13}\\W_{z21}&W_{z22}&W_{z23}\\W_{z31}&W_{z32}&W_{z33}\end{array}\right]^T\left[\begin{array}{c}Q_1-T_1\\Q_2-T_2\\Q_3-T_3\end{array}\right]\right)\circ\left[\begin{array}{c}y_1(1-y_1)\\y_2(1-y_2)\\y_3(1-y_3)\end{array}\right]\left[\begin{array}{cc}x&1\end{array}\right]" title="\frac{\partial P}{\partial W_x}=\left(\left[\begin{array}{ccc}W_{y11}&W_{y12}&W_{y13}\\W_{y21}&W_{y22}&W_{y23}\\W_{y31}&W_{y32}&W_{y33}\end{array}\right]\circ\left[\begin{array}{c}z_1(1-z_1)\\z_2(1-z_2)\\z_3(1-z_3)\end{array}\right]\right)^T\left(\left[\begin{array}{ccc}W_{z11}&W_{z12}&W_{z13}\\W_{z21}&W_{z22}&W_{z23}\\W_{z31}&W_{z32}&W_{z33}\end{array}\right]^T\left[\begin{array}{c}Q_1-T_1\\Q_2-T_2\\Q_3-T_3\end{array}\right]\right)\circ\left[\begin{array}{c}y_1(1-y_1)\\y_2(1-y_2)\\y_3(1-y_3)\end{array}\right]\left[\begin{array}{cc}x&1\end{array}\right]" />
