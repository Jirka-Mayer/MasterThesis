#!\bin\bash

VERSION="2.7.1-gpu"

RAW_IMAGE_TAG="tensorflow-plus-packages"
IMAGE_TAG="$RAW_IMAGE_TAG:$VERSION"

DOCKERFILE="\
FROM tensorflow/tensorflow:$VERSION
RUN pip3 install --upgrade tensorflow-probability==0.15.0
RUN pip3 install --upgrade tensorflow-datasets==4.5.2
# RUN pip3 install --upgrade matplotlib==3.4.3
# RUN pip3 install --upgrade opencv-python==4.5.3.56
WORKDIR /app
CMD bash
"

if [ -z "$(docker images | grep -F $RAW_IMAGE_TAG)" ]; then
    echo "BUILDING $IMAGE_TAG DOCKER IMAGE..."
    echo "$DOCKERFILE" | docker build -t $IMAGE_TAG -
fi

docker run \
    -u $(id -u):$(id -g) \
    --rm \
    --gpus all \
    -it \
    -v $(realpath ~/Datasets):/Datasets:ro \
    -v $(pwd):/app \
    -w /app \
    $IMAGE_TAG \
    bash $@
