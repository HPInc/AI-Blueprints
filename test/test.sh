#!/bin/bash

echo $1
echo $2
cd aistudio-samples/
python test/py_utils/main.py -s $1 -o .tmp_folder -w $2

python .tmp_folder/mnist_test.py
