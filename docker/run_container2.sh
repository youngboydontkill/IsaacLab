# run container with Isaac Sim 4.5 & IsaacLab 2.1.0 
#!/bin/bash
# 容器配置参数
CONTAINER_NAME="leju-kuavo-lab2"
# ENV_FILE=".env.base"
# DOCKERFILE_PATH="../docker/Dockerfile"  # 根据实际路径调整
ISAACLAB_BASE_IMAGE="nvcr.io/nvidia/isaac-lab:2.1.0"
DOCKER_ISAACLAB_EXTENSION_PATH="/workspace/leju-kuavo-lab"

# 检查容器是否存在
check_container_exists() {
    docker ps -a --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"
}

# 进入容器
enter_container() {
    echo "进入已存在的容器: $CONTAINER_NAME"
    docker exec -it "$CONTAINER_NAME" /bin/bash
}

# 创建新容器并执行Dockerfile操作
create_and_setup_container() {
    echo "创建新容器: $CONTAINER_NAME"
    
    # 1. 创建容器（保持后台运行）
    docker run -d \
        --name "$CONTAINER_NAME" \
        -e "PRIVACY_CONSENT=Y" \
        -v $HOME/.Xauthority:/root/.Xauthority \
        -v ~/docker/isaac-sim/cache/kit:/isaac-sim/kit/cache:rw \
        -v ~/docker/isaac-sim/cache/ov:/root/.cache/ov:rw \
        -v ~/docker/isaac-sim/cache/pip:/root/.cache/pip:rw \
        -v ~/docker/isaac-sim/cache/glcache:/root/.cache/nvidia/GLCache:rw \
        -v ~/docker/isaac-sim/cache/computecache:/root/.nv/ComputeCache:rw \
        -v ~/docker/isaac-sim/logs:/root/.nvidia-omniverse/logs:rw \
        -v ~/docker/isaac-sim/data:/root/.local/share/ov/data:rw \
        -v ~/docker/isaac-sim/documents:/root/Documents:rw \
        --gpus all \
        --network host \
        -e OMNI_KIT_ALLOW_ROOT=1 \
        -e "ACCEPT_EULA=Y" \
        -v /tmp/.X11-unix:/tmp/.X11-unix \
        -e DISPLAY=${DISPLAY} \
        -v "$(pwd)/..:$DOCKER_ISAACLAB_EXTENSION_PATH" \
        --workdir /workspace \
        "$ISAACLAB_BASE_IMAGE"
    # 2. 在容器内执行Dockerfile操作
    echo "在容器内执行Dockerfile构建步骤..."
    docker exec "$CONTAINER_NAME" /bin/bash -c "\
        set -ex && \
        cd '$DOCKER_ISAACLAB_EXTENSION_PATH' && \
        /workspace/isaaclab/_isaac_sim/kit/python/bin/python3 -m pip install -e ./exts/ext_template"
}

# 主逻辑
if check_container_exists; then
    # 启动停止状态的容器
    if [ "$(docker inspect -f '{{.State.Running}}' "$CONTAINER_NAME")" != "true" ]; then
        docker start "$CONTAINER_NAME"
    fi
    enter_container
else
    create_and_setup_container
fi