### 1、卸载所有的旧版本，旧版本的Docker被称为 docker 或 docker-engine
```shell
sudo yum remove docker \
                docker-client \
                docker-client-latest \
                docker-common \
                docker-latest \
                docker-latest-logrotate \
                docker-logrotate \
                docker-engine
```

### 2、安装
```shell
yum install -y yum-utils

#sudo yum-config-manager --add-repo https://download.docker.com/linux/centos/docker-ce.repo #国外下载源
sudo yum-config-manager --add-repo https://mirrors.aliyun.com/docker-ce/linux/centos/docker-ce.repo  #国内下载源

sudo yum install docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin

```

### 3、启动
```shell
sudo systemctl start docker
sudo systemctl enable docker
```

### 创建目录
```shell
sudo mkdir -p /etc/docker
```

### 写入镜像配置
```shell
sudo tee /etc/docker/daemon.json <<-'EOF'
{
    "data-root": "/data/docker",
    "registry-mirrors": [
        "https://hub.xdark.top",
        "https://hub.littlediary.cn",
        "https://dockerpull.org",
        "https://hub.crdz.gq",
        "https://docker.1panel.live",
        "https://docker.unsee.tech",
        "https://docker.m.daocloud.io",
        "https://docker.kejilion.pro",
        "https://registry.dockermirror.com",
        "https://hub.rat.dev",
        "https://dhub.kubesre.xyz",
        "https://docker.nastool.de",
        "https://docker.udayun.com",
        "https://docker.rainbond.cc",
        "https://hub.geekery.cn",
        "https://docker.1panelproxy.com"
    ]
}
EOF
```
### 重启docker服务
```shell
sudo systemctl daemon-reload
sudo systemctl restart docker
```