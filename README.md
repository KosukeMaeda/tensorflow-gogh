# tensorflow-gogh
A Neural Algorithm of Artistic Style([Original](https://arxiv.org/abs/1508.06576))

|content|style|result|
|-----|----|----|
|<img src="https://raw.githubusercontent.com/KosukeMaeda/tensorflow-gogh/images/images/img1.jpg" width=256px>|<img src="https://raw.githubusercontent.com/KosukeMaeda/tensorflow-gogh/images/images/img2.jpg" width=256px>|<img src="https://raw.githubusercontent.com/KosukeMaeda/tensorflow-gogh/images/images/progress.gif" width=256px>


## Usage
1. Download [vgg16.npy](https://mega.nz/#!YU1FWJrA!O1ywiCS2IiOlUCtCpI6HTJOMrneN-Qdv3ywQP5poecM) file and move to `./data/`

2. Install tensorflow, scikit-image, PIL
```
pip install tensorflow scikit-image pillow
```
3. Run tensorflow-gogh
```
python tensorflow-gogh.py
```

## References
 - [A Neural Algorithm of Artistic Style](https://arxiv.org/abs/1508.06576)
 - https://research.preferred.jp/2015/09/chainer-gogh/
 - https://github.com/machrisaa/tensorflow-vgg
