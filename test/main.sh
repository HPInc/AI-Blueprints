####Params
TEST_SCRIPT="test/first_script.yaml"

#####From image config
WORKSPACE="deeplearning"
IMAGE_REPO="public.ecr.aws/hpphoenixdsp/itg/base-images"
IMAGE_NAME="dljp-gpu"
IMAGE_TAG="0.11.5"
FULL_IMAGE=${IMAGE_REPO}/${IMAGE_NAME}:${IMAGE_TAG}

#### From app folder
ENVOY_LOCAL="/mnt/c/Users/borges/AppData/Local/HP/AIStudio/certs"

#### Container folders
ENVOY_CONTAINER="/etc/envoy/certs"
GIT_CONTAINER="/home/jovyan/aistudio-samples"
TEST_CONTAINER="/home/jovyan/test"
DATA_CONTAINER="/home/jovyan/datafabric/tutorial"

#### Code and data from user
GIT_LOCAL="/mnt/c/Users/borges/code/public/aistudio-samples"
DATA_LOCAL="/mnt/c/Users/borges/AppData/Local/HP/AIStudio/tutorial"

#### Do I need this????
TEST_LOCAL="/mnt/c/Users/borges/code/public/aistudio-samples/test"


echo ${FULL_IMAGE}
sudo nerdctl -n phoenix run --gpus all -it -v ${ENVOY_LOCAL}:${ENVOY_CONTAINER} -v ${GIT_LOCAL}:${GIT_CONTAINER} -v ${DATA_LOCAL}:${DATA_CONTAINER} -v ${TEST_LOCAL}:${TEST_CONTAINER} ${FULL_IMAGE} ./aistudio-samples/test/test.sh $TEST_SCRIPT $WORKSPACE
