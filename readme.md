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
# How to run

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


# Terraform


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
* 