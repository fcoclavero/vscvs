# scvs
Common Visual Semantic Vector Space

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
6. Commit `git commit -m "Removed submodule "`
7. Delete the now untracked submodule files `rm -rf path_to_submodule`
