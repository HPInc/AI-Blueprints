#!/bin/bash
echo ""
echo ""
echo "*-*-*-*-*-*-*-*-*-*-*-* Started test using the workspace: $2 *-*-*-*-*-*-*-*-*-*-*-* "	
echo ""
echo ""
SCRIPT=$(readlink -f $0)
SCRIPT_FOLDER=$(dirname ${SCRIPT})
cd $SCRIPT_FOLDER/..

#TMP_FOLDER="${SCRIPT_FOLDER}/.tmp_folder"
TMP_FOLDER="/home/jovyan/.tmp_folder"

# Install required dependencies
pip install astor

python ${SCRIPT_FOLDER}/py_utils/main.py -s $1 -o $TMP_FOLDER -w $2

for TEST_FILE in $(ls $TMP_FOLDER/*test.py ); do
    echo "******************************"	
    echo "Running tests on ${TEST_FILE}"	
    python $TEST_FILE
    echo "******************************"	
done

echo ""
echo ""
echo "*-*-*-*-*-*-*-*-*-*-*-* Finished test on workspace: $2 *-*-*-*-*-*-*-*-*-*-*-* "	
echo ""
echo ""