###Gets Local App Data from windows, removes carriage return and convert to wsl format
LOCALAPPDATA=$(cmd.exe /c echo %LOCALAPPDATA%)
LOCALAPPDATA=${LOCALAPPDATA::-1}
LOCALAPPDATA=$(wslpath $LOCALAPPDATA)

##Load other constants
TESTSCRIPT=$1
SCRIPT=$(readlink -f $0)
HOME_CONTAINER="/home/jovyan"
DATAFABRIC_FOLDER="${HOME_CONTAINER}/datafabric/"
AISTUDIO_FOLDER="${LOCALAPPDATA}/HP/AIStudio"
export NERDCTL_LOG_LEVEL=error
export NERDCTL_PROGRESS=quiet
if test -t 1; then
  RED=$(tput setaf 1); GREEN=$(tput setaf 2); YELLOW=$(tput setaf 3)
  BLUE=$(tput setaf 4); BOLD=$(tput bold); RESET=$(tput sgr0)
else
  RED=""; GREEN=""; YELLOW=""; BLUE=""; BOLD=""; RESET=""
fi
COMMAND_PREFIX="sudo nerdctl -n phoenix run --gpus all"
ENTRYPOINT="test/test.sh"

### Ensures yq is installed
if ! command -v yq; then
  if ! command -v snap; then
    sudo apt install snapd
  fi
  sudo snap install yq  
fi

# Count statistics and helper function
TOTAL=0; PASSED=0; FAILED=0
run_container () {
  local _image="$1"
  local _cmd="$2"
  local _ts="$3"
  TOTAL=$((TOTAL+1))
  if eval "${_cmd}" ; then
    PASSED=$((PASSED+1))
    printf "Test using image ${_image} - ${GREEN}${BOLD}✅  PASS${RESET}\n"
  else
    FAILED=$((FAILED+1))
    printf "${RED}${BOLD}❌  FAIL${RESET}\n"
  fi
  printf "\n"
}

### Ensures nerdctl is installed
NERDCTL_VERSION="2.0.5"
NERDCTL_FILE="nerdctl-full-${NERDCTL_VERSION}-linux-amd64.tar.gz"
NERDCTL_URL="https://github.com/containerd/nerdctl/releases/download/v${NERDCTL_VERSION}/${NERDCTL_FILE}"
if ! command -v nerdctl; then
  wget -q ${NERDCTL_URL}
  sudo tar Cxzf /usr/local ${NERDCTL_FILE}
  rm ${NERDCTL_FILE}
fi

### Defines parameter to mount Envoy certificates
ENVOY_HOST="${AISTUDIO_FOLDER}/certs"
ENVOY_CONTAINER="/etc/envoy/certs"
ENVOY_PARAM="-v ${ENVOY_HOST}:${ENVOY_CONTAINER}"

### Defines parameter to mount Github repo folder and and relative endpoints
SCRIPT_FOLDER=$(dirname ${SCRIPT})
GITHUB_HOST=$(readlink -f ${SCRIPT_FOLDER}/..)
GITHUB_REPO=$(basename $GITHUB_HOST)
GITHUB_PARAM="-v ${GITHUB_HOST}:${HOME_CONTAINER}/${GITHUB_REPO}"
CONTAINER_ENTRYPOINT="${HOME_CONTAINER}/${GITHUB_REPO}/${ENTRYPOINT}"
HOST_TESTSCRIPT=$(readlink -f $TESTSCRIPT)

### If "mount" is passed, this will allow persisting the tmp_folder used to save temporary results
if [[ -n "$2" ]] && [[ "$2" == "mount" ]]; then
  MOUNT_ARG="mount"
else
  MOUNT_ARG="nomount"
fi

### If "venv" is passed, this will allow using virtual environment
if [[ -n "$3" ]] && [[ "$3" == "venv" ]]; then
  VENV_ARG="venv"
else
  VENV_ARG="novenv"
fi

### This tests whether the test script is in a subfolder of the Github repo folder
if [[ "${HOST_TESTSCRIPT}" == ${GITHUB_HOST}* ]]; then
  RELATIVE_TESTSCRIPT=${HOST_TESTSCRIPT#"${GITHUB_HOST}/"}
  CONTAINER_TESTSCRIPT="${HOME_CONTAINER}/${GITHUB_REPO}/${RELATIVE_TESTSCRIPT}"
else
  echo "PROBLEM IS HERE"  
fi

### Defines parameter to mount assets
ASSETS_PARAM=""
for ASSET in $(yq ".assets|keys" $TESTSCRIPT); do
  if [[ $ASSET != "-" ]]; then
	ASSET_HOST=$(yq ".assets.$ASSET" $TESTSCRIPT)
	if [[ ${ASSET_HOST:1:1} == ":" ]]; then
		ASSET_HOST=$(wslpath $ASSET_HOST)
	fi	
	ASSET_CONTAINER="${DATAFABRIC_FOLDER}${ASSET}"
	ASSETS_PARAM="-v ${ASSETS_PARAM} ${ASSET_HOST}:${ASSET_CONTAINER}"
  fi
 done

### Run tests on each base image
for IMAGE in $(yq ".baseimages|keys" $TESTSCRIPT); do  
  if [[ -n "$IMAGE" ]] && [[ $IMAGE != "-" ]] && [[ $IMAGE != "registry" ]]; then
    echo ""
    echo ""
    printf "*-*-*-*-*-*-*-*-*-*-*-* ${BLUE}Starting container:${RESET} ${YELLOW}${IMAGE}${RESET} *-*-*-*-*-*-*-*-*-*-*-*"
    echo ""
    echo ""
	  IMAGE_REGISTRY=$(yq ".baseimages.registry" $TESTSCRIPT)
    IMAGE_NAME=$(yq .$IMAGE test/strings.yaml)
	  IMAGE_VERSION=$(yq ".baseimages.${IMAGE}" $TESTSCRIPT)
	  FULL_IMAGE="$IMAGE_REGISTRY/$IMAGE_NAME:$IMAGE_VERSION"
	  FULL_COMMAND="$COMMAND_PREFIX $ENVOY_PARAM $GITHUB_PARAM $ASSETS_PARAM"
	  FULL_COMMAND="$FULL_COMMAND $FULL_IMAGE $CONTAINER_ENTRYPOINT $CONTAINER_TESTSCRIPT $IMAGE $MOUNT_ARG"
    run_container "$IMAGE" "$FULL_COMMAND" "$TESTSCRIPT"
  fi
done

### If there is no NGC images, exit
if [[ -z $(yq 'has("ngcconfig")' $TESTSCRIPT) ]]; then
  echo "No NGC images to run"
  exit 0
else
  ### If there is NGC images, run tests on each of them
  for IMAGE_ENTRY in $(yq ".ngcconfig|keys" $TESTSCRIPT); do  
    if [[ $IMAGE_ENTRY != "-" ]]; then
      echo ""
      echo ""
      printf "*-*-*-*-*-*-*-*-*-*-*-* ${BLUE}Starting container:${RESET} ${YELLOW}${IMAGE_ENTRY}${RESET} *-*-*-*-*-*-*-*-*-*-*-*"
      echo ""
      echo ""
      IMAGE=${IMAGE_ENTRY%version*}
      IMAGE_URL=$(yq .$IMAGE test/strings.yaml)
      IMAGE_VERSION=$(yq ".ngcconfig.${IMAGE_ENTRY}" $TESTSCRIPT)
	    FULL_IMAGE="$IMAGE_URL:$IMAGE_VERSION"
	    FULL_COMMAND="$COMMAND_PREFIX $ENVOY_PARAM $GITHUB_PARAM $ASSETS_PARAM"
	    FULL_COMMAND="$FULL_COMMAND $FULL_IMAGE $CONTAINER_ENTRYPOINT $CONTAINER_TESTSCRIPT $IMAGE $MOUNT_ARG $VENV_ARG"
      run_container "$IMAGE_ENTRY" "$FULL_COMMAND"
    fi
  done
fi

# Final summary at end of file (just before the last fi closes script)
printf "${BOLD}Test summary:${RESET}  total=%d  ${GREEN}passed=%d${RESET}  ${RED}failed=%d${RESET}\n" \
       "$TOTAL" "$PASSED" "$FAILED"
printf "\n"
exit $FAILED
