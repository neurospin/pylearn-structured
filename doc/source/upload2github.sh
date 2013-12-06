#!/bin/bash

# This script has only been test on ubuntu

# build html page on your machine
make html
epydoc -v --html parsimony -o ./_build/html/epydoc_api


# build tmp directory
outdir="$(mktemp -d)"
curdir="$(pwd)"

# download the epac from github to upload pages
git clone git@github.com:neurospin/pylearn-parsimony.git $outdir

# checkout gh-pages which are the epac webpage on github and commit them to github
cd $outdir
git fetch origin
git checkout -b gh-pages origin/gh-pages
cp -r $curdir/_build/html/* $outdir
git add .
git commit -a -m "DOC: update pages"
git push origin gh-pages
cd $curdir
rm -rf $outdir
