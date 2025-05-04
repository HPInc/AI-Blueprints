WORKSPACE="deeplearning"
TEST_SCRIPT="test/first_script.yaml"
IMAGE_REPO="public.ecr.aws/hpphoenixdsp/itg/base-images"
IMAGE_NAME="dljp"
IMAGE_TAG="0.13.6"
#IMAGE_NAME="local-genai"
#IMAGE_TAG="0.4.5"
FULL_IMAGE=${IMAGE_REPO}/${IMAGE_NAME}:${IMAGE_TAG}
ENVOY_LOCAL="/mnt/c/Users/borges/AppData/Local/HP/AIStudio/certs"
ENVOY_CONTAINER="/etc/envoy/certs"
GIT_LOCAL="/mnt/c/Users/borges/code/public/aistudio-samples"
GIT_CONTAINER="/home/jovyan/aistudio-samples"
DATA_LOCAL="/mnt/c/Users/borges/AppData/Local/HP/AIStudio/tutorial"
DATA_CONTAINER="/home/jovyan/datafabric/tutorial"
TEST_LOCAL="/mnt/c/Users/borges/code/public/aistudio-samples/test"
TEST_CONTAINER="/home/jovyan/test"
echo ${FULL_IMAGE}
sudo nerdctl -n phoenix run --gpus all -it -v ${ENVOY_LOCAL}:${ENVOY_CONTAINER} -v ${GIT_LOCAL}:${GIT_CONTAINER} -v ${DATA_LOCAL}:${DATA_CONTAINER} -v ${TEST_LOCAL}:${TEST_CONTAINER} ${FULL_IMAGE} bash
