FROM pytorch/pytorch:2.6.0-cuda12.6-cudnn9-runtime

RUN groupadd -r user && useradd -m --no-log-init -r -g user user

RUN mkdir -p /opt/app /input /output \
    && chown user:user /opt/app /input /output

USER user
WORKDIR /opt/app

ENV PATH="/home/user/.local/bin:${PATH}"

RUN python -m pip install --user -U pip && python -m pip install --user pip-tools

# COPY --chown=user:user requirements.txt /opt/app/
# RUN python -m pip install --user -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

COPY --chown=user:user resources/nnUNet/ /opt/app/nnUNet
WORKDIR /opt/app/nnUNet/
RUN python -m pip install --user -e . -i https://pypi.tuna.tsinghua.edu.cn/simple
RUN python -m pip install --user evalutils==0.4.2 -i https://pypi.tuna.tsinghua.edu.cn/simple
RUN python -m pip install --user tensorboard monai -i https://pypi.tuna.tsinghua.edu.cn/simple

WORKDIR /opt/app

COPY --chown=user:user iac /opt/app/iac

COPY --chown=user:user nnunet_pred.py /opt/app/

ENTRYPOINT [ "python", "nnunet_pred.py" ]

# COPY --chown=user:user offical_pred.py /opt/app/

# ENTRYPOINT [ "python", "offical_pred.py" ]