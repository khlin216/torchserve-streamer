# structure
In repo and after you run terraform envoloped in stars
```
├── eks
│   ├── k8s
│   │   ├── deployment.yaml
│   │   ├── readme.md
│   │   ├── ssm.yaml
│   │   └── svc_elb.yaml
│   └── terraform
│       ├── cluster
│       │   ├── ecr.tf
│       │   ├── eks.tf
│       │   ├── main.tf
│       │   ├── **terraform.tfstate**
│       │   ├── **terraform.tfstate.backup**
│       │   └── vars.tf
│       ├── helm
│       │   ├── helm.tf
│       │   ├── **terraform.tfstate**
│       │   ├── **terraform.tfstate.backup**
│       │   └── vars.tf
│       ├── iam_policy.json
│       ├── readme.md
│       ├── terraform-destroy.sh
│       └── terraform-run.sh
├── model_store
├── readme.md
├── stream
│   └── vlc_streamer.py
├── tests
│   ├── **influencer.png**
│   ├── report.ipynb
│   ├── swarm.py
│   └── worker.py
└── torchserve
    ├── configs.properties
    ├── coordinator.py
    ├── Dockerfile
    ├── **img.png**
    ├── methods
    │   ├── constants.py
    │   ├── face_det_init_cnn.py
    │   └── torchserve2mar.py
    ├── model_files
    ├── model_store
    ├── readme.md
    ├── requirements.txt
    ├── **serve_alldet**
    │   └── **all_det.mar**
    └── streamer.py

```
# How to run locally

Firstly we need to convert all of the source code into .mar file

just use 
```
python methods/torchserve2mar.py
```

then start torchserve:
```
torchserve --start --foreground --ts-config ./configs.properties
```

Checkout dockerfile to see how to install the required libraries 

Or just use the docker version

```
sudo docker build -f Dockerfile.gpu -t streamo .
```

For the cpu version
```
sudo docker build -f Dockerfile.gpu -t streamo .
```
and then to run it add network host option

```
sudo docker run --network host --gpus all streamo 
```


## Important links

- http://127.0.0.1:9003/metrics -- metrics and basic info
- http://127.0.0.1:9001/predictions/all_det -- to predict basically send an image and return the model response


### example of predicting something 

```
binary_image = open("test.png", "rb").read()
url = "http://127.0.0.1:9001/predictions/all_det"
r = requests.put(url, binary_image, timeout=timeout).content

```

after running torchserve you can actually run a stream from twitch using 

```
python streamer.py  https://www.twitch.tv/matteohs
```

or any other url. Make sure that you have a GPU otherwise the facedetection model will become super slow and the torchserve workers will return timeouts that I havent taken into account in the streaming code.

# Deploy on AWS

## Terraform

Build infra on aws using the following code or using the bash script in ```eks/terraform/terraform-run.sh```
```
export KUBE_CONFIG_PATH=~/.kube/config
terraform init
terraform plan
terraform apply
# terraform destroy
```

```
aws eks update-kubeconfig --name stream-torch --region us-east-2
cd ./eks/k8s
kubectl apply -f .
kubectl get svc -o wide
```

build a new image like this:
```
$ cd torchserve
$ python methods/torchserve2mar.py
$ docker build -t torchserve-repo -f Dockerfiles/Dockerfile.gpu .
$ docker tag torchserve-repo:latest 814594265042.dkr.ecr.us-east-2.amazonaws.com/torchserve-repo:latest
$ docker push 814594265042.dkr.ecr.us-east-2.amazonaws.com/torchserve-repo:latest
```

after you finish building the infrastructure (make sure what are you using a GPU cluster or a CPU cluster)

Build an image depending on the type of the cluster in ```torchserve/Dockerfiles/```

For example to build the CPU image you can run

```sudo docker build -f Dockerfiles/Dockerfile.cpu -t torchserve-repo . ```

After that use the push commands in AWS.ECR and push the image to ECR repo.

Now you can apply the configs of k8s in ```eks/k8s``` or ```eks/k8s.cpu```


# NOTES

to check if there is a gpu in the cluster

```
kubectl describe nodes  |  tr -d '\000' | sed -n -e '/^Name/,/Roles/p' -e '/^Capacity/,/Allocatable/p' -e '/^Allocated resources/,/Events/p'  | grep -e Name  -e  nvidia.com  | perl -pe 's/\n//'  |  perl -pe 's/Name:/\n/g' | sed 's/nvidia.com\/gpu:\?//g'  | sed '1s/^/Node Available(GPUs)  Used(GPUs)/' | sed 's/$/ 0 0 0/'  | awk '{print $1, $2, $3}'  | column -t

```


to get daemon sets

```
kubectl get ds --all-namespaces | grep -i nvidia
```

```
kubectl get pods -n kube-system | grep -i nvidia
```

# ISSUES
if you see this error:
```
│ Error: Kubernetes cluster unreachable: exec plugin: invalid apiVersion "client.authentication.k8s.io/v1alpha1"
```
try the following:
* update aws cli to latest version (to enable v1beta1 instead of v1alpha1)
* downgrade helm to 3.8 from 3.9
* remember to set `export KUBE_CONFIG_PATH=~/.kube/config`


### steps to add a new model:
first, create .mar for new model:
* copy `coordinator.py`
* edit `methods/torchserve2mar.py` to use new coordinator name
* from `torchserve` directory run: `$ python methods/torchserve2mar.py`
* edit `configs/configs.cpu.properties` to add new model using similar syntax
* push image to ecr and follow push instructions on there
* delete `torchserve` deployment and bring it up again

current model weights are:
  - fullyolo2/weights/best.pt
  - version13/checkpoints/epoch=45-step=4500.ckpt
