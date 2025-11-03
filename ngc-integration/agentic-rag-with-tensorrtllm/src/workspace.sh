#!/bin/bash
export JUPYTERLAB_PORT=8888
export JUPYTERLAB_INTERNAL_PORT=8889
export SHELL=/bin/bash
export SETUP_FOLDER=/phoenix/setup
export ENVOY_FOLDER=${SETUP_FOLDER}/envoy
export IMAGE_SETUP_FOLDER=${SETUP_FOLDER}/image-setup
export OPENSSH_SETUP_FOLDER=${SETUP_FOLDER}/openssh
export BEFORE_NOTEBOOK=/usr/local/bin/before-notebook.d

echo -e "Using folders:\nSETUP_FOLDER=${SETUP_FOLDER}\nIMAGE_SETUP_FOLDER=${IMAGE_SETUP_FOLDER}\nBEFORE_NOTEBOOK=${BEFORE_NOTEBOOK}"

if ! ${IMAGE_SETUP_FOLDER}/check-os.sh; then
    echo "OS check failed"
    exit 1
fi

# Install OS dependencies
${IMAGE_SETUP_FOLDER}/install-os-deps.sh

# Install OpenSSH Server
echo "Setting up OpenSSH Server..."
${OPENSSH_SETUP_FOLDER}/setup-ssh.sh

# Start SSH Server
echo "Starting OpenSSH Server..."
/usr/sbin/sshd -D -f /usr/local/etc/sshd/sshd_config &

# Configure and Start Envoy
${ENVOY_FOLDER}/install.sh ${JUPYTERLAB_PORT} ${JUPYTERLAB_INTERNAL_PORT} ${ENVOY_FOLDER}
${ENVOY_FOLDER}/start.sh
# This must be run using source in order for the exit codes to early exit
# in the library validation case
source ${IMAGE_SETUP_FOLDER}/setup-venv-env.sh
# not sure if we want to keep this or not
# the prenotebook hooks are a feature of the jupyter lab images
# and won't run in arbitrary images
# ${IMAGE_SETUP_FOLDER}/check-network.sh


chmod 777 /home/jovyan
non_root_user=$(getent passwd | grep /home | cut -d: -f1)
echo "User name = ${non_root_user}"

/bin/bash "${SETUP_FOLDER}/jupyter.sh" "${JUPYTERLAB_PORT}" "${JUPYTERLAB_INTERNAL_PORT}" "${SETUP_FOLDER}"
#if [ -z "${non_root_user}" ]; then
#	/bin/bash "${SETUP_FOLDER}/jupyter.sh" "${JUPYTERLAB_PORT}" "${JUPYTERLAB_INTERNAL_PORT}" "${SETUP_FOLDER}"
#else
#	sudo -u "${non_root_user}" env PATH="$PATH" /bin/bash "${SETUP_FOLDER}/jupyter.sh" "${JUPYTERLAB_PORT}" "${JUPYTERLAB_INTERNAL_PORT}" "${SETUP_FOLDER}"
#fi
