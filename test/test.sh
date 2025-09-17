#!/bin/bash
# This script is used to run automated tests on a (workspace)container. It takes 4 arguments:
# 1. The source script with metadata for the tests
# 2. The workspace to run the tests in
# 3. An argument to specify if temp folder should be mounted from the host (mount / nomount)
# 4. An optional argument to specify if the tests should be run in a virtual environment (venv / novenv)
echo ""
echo ""
echo "*-*-*-*-*-*-*-*-*-*-*-* Started test using the workspace: $2 *-*-*-*-*-*-*-*-*-*-*-* "
echo ""
echo ""
SCRIPT=$(readlink -f $0)
SCRIPT_FOLDER=$(dirname ${SCRIPT})
cd $SCRIPT_FOLDER/..

# Checks it the third argument is provided and equals to "mount"
if [[ -n "$3" ]] && [[ "$3" == "mount" ]]; then
    echo "Mounting temp folder on host - to be persisted after container is stopped"
    TMP_FOLDER="${SCRIPT_FOLDER}/.tmp_folder"
    mkdir -p $TMP_FOLDER
else
    echo "Using local temp folder"
    TMP_FOLDER="/home/jovyan/.tmp_folder"
fi

# Install required dependencies
pip install astor

python ${SCRIPT_FOLDER}/py_utils/main.py -s $1 -o $TMP_FOLDER -w $2

for TEST_FILE in $(find $TMP_FOLDER -type f -name '*test.pyt'); do
    echo "******************************"
    echo "Running tests on ${TEST_FILE}"
    echo "     ****************"

    ## Create a virtual environment for each test file with a random name

    ###If the fourth argument is provided and equals to "venv", create a virtual environment
    if [[ -n "$4" ]] && [[ "$4" == "venv" ]]; then
        echo "Creating a virtual environment for the test file"
        ENV_FOLDER=$(mktemp -d ${TMP_FOLDER}/env_XXXXXX)
        python -m venv ${ENV_FOLDER}/venv --system-site-packages
        source ${ENV_FOLDER}/venv/bin/activate
        if [[ -f "${TEST_FILE}_requirements.txt" ]]; then
            pip install -r ${TEST_FILE}_requirements.txt --quiet
        fi
        # Run the test file using IPython
        ipython $TEST_FILE
        ${ENV_FOLDER}/venv/bin/deactivate
    else
        pip install -r ${TEST_FILE}_requirements.txt
        ipython $TEST_FILE
    fi
    echo "******************************"

done

echo ""
echo ""
echo "*-*-*-*-*-*-*-*-*-*-*-* Finished test on workspace: $2 *-*-*-*-*-*-*-*-*-*-*-* "
echo ""
echo ""
