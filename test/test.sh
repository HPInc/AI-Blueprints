#!/bin/bash
SCRIPT=$(readlink -f $0)
TMP_FOLDER="/home/jovyan/.tmp_folder"

SCRIPT_FOLDER=$(dirname ${SCRIPT})
echo $SCRIPT_FOLDER
cd $SCRIPT_FOLDER/..

python ${SCRIPT_FOLDER}/py_utils/main.py -s $1 -o $TMP_FOLDER -w $2

for TEST_FILE in $(ls $TMP_FOLDER/*test.py); do
    echo $TEST_FILE
done
#	python "${SCRIPT_FOLDER}/${TEST_FILE}"
#done
	
