FROM artifactory-kfs.habana-labs.com/docker-local/1.12.0/ubuntu22.04/habanalabs/pytorch-installer-2.0.1:1.12.0-180
RUN apt-get update && apt-get install -y openssh-server strace sudo screen rsync git
RUN echo 'root:visl' | chpasswd
#RUN ssh-keygen -t rsa -f /root/.ssh/id_rsa -q -P ""
RUN sed -i 's/#PermitRootLogin prohibit-password/PermitRootLogin yes/' /etc/ssh/sshd_config

# SSH login fix. Otherwise, user is kicked off after login
RUN sed 's@session\s*required\s*pam_loginuid.so@session optional pam_loginuid.so@g' -i /etc/pam.d/sshd

ENV NOTVISIBLE "in users profile"
RUN echo "export VISIBLE=now" >> /etc/profile

RUN pip install --upgrade pip
RUN pip install runhouse transformers

RUN git clone https://github.com/htang2012/langchain.git
WORKDIR /langchain
RUN git checkout gaudi
WORKDIR /langchain/libs/langchain
RUN pip install -e .
WORKDIR /root/.ssh
WORKDIR /langchain/dockers
COPY id_rsa /root/.ssh
COPY id_rsa.pub /root/.ssh
RUN cat /root/.ssh/id_rsa.pub >> /root/.ssh/authorized_keys
RUN chmod 700 /root/.ssh
RUN chmod 600 /root/.ssh/*
WORKDIR /langchain/gaudi_tests

#RUN  curl -sSL https://install.python-poetry.org | python3 -
#RUN /root/.local/bin/poetry install --no-ansi --no-interaction
#ENV PATH="/root/.local/bin:$PATH"
#RUN python -m venv --system-site-packages venv
#RUN . venv/bin/activate
#RUN /root/.local/bin/poetry env use $(which python)
#RUN /root/.local/bin/poetry install






EXPOSE 22
CMD ["/usr/sbin/sshd", "-D"]
