# vscvs

Common Visual Semantic Vector Space

## Examples

### Create embeddings for a trained model

```bash
python main.py embed --dataset-name sketchy-sketches --embeddings-name hog-sketches --batch-size 64 --workers 16 --n-gpu 1 hog --in-channels 3 --cell-size 8 --bins 9 --signed-gradients False
```

### Retrieve an image given a set of embeddings

```bash
python main.py retrieve --query-image-filename sketchy/sketch/tx_000000000000/ape/n02470325_6919-1.png --query-dataset-name sketchy-sketches --queried-dataset-name sketchy-photos --queried-embeddings-name hog-photos --k 16 --n-gpu 1 hog --in-channels 3 --cell-size 8 --bins 9 --signed-gradients False
```

### Measure the class recall for a set of embeddings

```bash
python main.py measure recall --k 5 --n-gpu 1 same-class --dataset-name sketchy-sketches --embeddings-name hog-sketches --test-split 0.2
```

### Measure the cross modal retrieval class recall for a set of embeddings

```bash
python main.py measure cross-modal recall --k 5 --n-gpu 1 same-class --sketch-dataset-name sketchy-sketches --photo-dataset-name sketchy-photos --sketch-embeddings-name hog-sketches --photo-embeddings-name hog-photos
```

## Tensorboad

Training routines generate Tensorboard compatible logs, which can be viewed in the Tensorboard console using the following command:

```bash
tensorboard --logdir=data/logs
```

## Git submodules

The project depends on several git submodules that provide NLP functions:

1. [textpreprocess](https://github.com/fcoclavero/textpreprocess): text preprocessing functions, such as spell checking and lemmatization
2. [wordvectors](https://github.com/fcoclavero/wordvectors): word and document embedding creation

Some submodules have their own submodules. To update all of them recursively, run the following:

```bash
git submodule update --init --recursive
```

For individual updates, `cd` into the corresponding directory and run the following:

```bash
git submodule init
git submodule update --remote
```

To [effectively remove a submodule](https://gist.github.com/myusuf3/7f645819ded92bda6677), you need to:

1. Delete the relevant section from the `.gitmodules` file.
2. Stage the `.gitmodules changes git add `.gitmodules`
3. Delete the relevant section from `.git/config`.
4. Run `git rm --cached path_to_submodule` (no trailing slash).
5. Run `rm -rf .git/modules/path_to_submodule` (no trailing slash).
6. Commit `git commit -m "Removed submodule"`
7. Delete the now untracked submodule files `rm -rf path_to_submodule`
