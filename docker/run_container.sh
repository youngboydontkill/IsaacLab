#!/bin/bash
# 容器配置参数
CONTAINER_NAME="leju-kuavo-lab"
# ENV_FILE=".env.base"
# DOCKERFILE_PATH="../docker/Dockerfile"  # 根据实际路径调整
ISAACLAB_BASE_IMAGE="isaac-lab-base"
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