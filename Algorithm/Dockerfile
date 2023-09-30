# FROM petermcgor/nnunetv2:0.0.2
FROM pytorch/pytorch:2.0.0-cuda11.7-cudnn8-devel

RUN groupadd -r user && useradd -m --no-log-init -r -g user user

RUN mkdir -p /opt/app /input /output/images/mediastinal-lymph-node-segmentation/ \
    && chown -R user:user /opt/app /input /output

USER user
WORKDIR /opt/app

ENV PATH="/home/user/.local/bin:${PATH}"
ENV nnUNet_results="/opt/app/nnUNet_results"
ENV MKL_SERVICE_FORCE_INTEL = 1
ENV MKL_THREADING_LAYER = GNU

COPY --chown=user:user . /opt/app/

RUN pip install -e /opt/app/nnUNet -i https://pypi.tuna.tsinghua.edu.cn/simple

ENTRYPOINT [ "python3", "process.py" ]
