# xgboost

$$
\begin{array}{l}{\mathcal{L}(\phi)=\sum_{i} l\left(\hat{y}_{i}, y_{i}\right)+\sum_{k} \Omega\left(f_{k}\right)} \\ {\text { where } \Omega(f)=\gamma T+\frac{1}{2} \lambda\|w\|^{2}}\end{array}
$$

$$
\mathcal{L}^{(t)}=\sum_{i=1}^{n} l\left(y_{i}, \hat{y}_{i}^{(t-1)}+f_{t}\left(\mathbf{x}_{i}\right)\right)+\Omega\left(f_{t}\right)
$$

$$
\begin{array}{c}{\mathcal{L}^{(t)} \simeq \sum_{i=1}^{n}\left[l\left(y_{i}, \hat{y}^{(t-1)}\right)+g_{i} f_{t}\left(\mathrm{x}_{i}\right)+\frac{1}{2} h_{i} f_{t}^{2}\left(\mathrm{x}_{i}\right)\right]+\Omega\left(f_{t}\right)} \\ {\text { where } g_{i}=\partial_{\hat{y}^{(t-1)}} l\left(y_{i}, \hat{y}^{(t-1)}\right) \text { and } h_{i}=\partial_{\hat{y}^{(t-1)}}^{2} l\left(y_{i}, \hat{y}^{(t-1)}\right)}\end{array}
$$